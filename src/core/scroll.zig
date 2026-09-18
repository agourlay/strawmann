//! §12 scroll: id-ordered pagination, and the order it pages through.
//!
//! Split out of `collection.zig` as the most separable of that
//! file's concerns: a published, snapshot-ordered id list with its own
//! reclamation rule, touching no distance kernel, no graph and no quantized
//! store. It still needs six fields off a collection (`alloc`, `id_space`,
//! `deleted`, `write_lock`, `active_searches`, `scroll_state`), so the
//! operations take a `*Collection`; the dependency runs one way, since nothing
//! here owns a collection back and `Scroll` is a plain field on it.
//!
//! The reclamation rule is why this is a file and not a section. It is
//! `active_searches`-gated quiescence -- the same discipline
//! `Collection.retired_graphs` uses and a *different instance* of it -- and
//! the two were interleaved across 3,700 lines where neither could be read
//! without tripping over the other.

const std = @import("std");
const ids = @import("ids.zig");
const collection = @import("collection.zig");
const hnsw = @import("../index/hnsw.zig");

const Collection = collection.Collection;
const SearchGuard = collection.SearchGuard;
const ExternalId = ids.ExternalId;

/// Total order on external ids, matching what `Scroll` must page through.
///
/// Numeric ids sort before UUIDs, then by value. Qdrant's `PointId` is a oneof
/// and the two variants are not mutually comparable, so *some* convention is
/// required; this one is stated rather than inherited so pagination is at least
/// stable and total. A collection mixing the two is unusual, and the choice
/// only becomes visible there.
pub fn idLess(a: ExternalId, b: ExternalId) bool {
    return switch (a) {
        .num => |x| switch (b) {
            .num => |y| x < y,
            .uuid => true,
        },
        .uuid => |x| switch (b) {
            .num => false,
            .uuid => |y| x < y,
        },
    };
}

const ScrollCtx = struct {
    space: *const ids.IdSpace,
    fn lessThan(self: ScrollCtx, a: u32, b: u32) bool {
        return idLess(self.space.external(a), self.space.external(b));
    }
};

/// The materialised id order. Heap-allocated so the published order is a
/// single pointer that can be swapped atomically.
pub const ScrollOrder = struct {
    items: []u32,
};

/// The published id order and the orders waiting to be freed.
///
/// One owner because the two are one invariant: an order is retired rather
/// than freed while a reader may be inside a page, and freed only once
/// `active_searches` is zero. That rule lived in `dropScrollOrder`, twelve
/// hundred lines from the two `Collection` fields it governed, and `deinit`
/// open-coded half of it. Nothing else on `Collection` could tell you the
/// two fields had to move together.
pub const Scroll = struct {
    published: std.atomic.Value(?*ScrollOrder) = .init(null),
    /// Orders replaced by a mutation. Mutated only under `write_lock`.
    retired: std.ArrayList(*ScrollOrder) = .empty,

    pub fn current(self: *const Scroll) ?*ScrollOrder {
        return self.published.load(.acquire);
    }

    pub fn publish(self: *Scroll, order: *ScrollOrder) void {
        self.published.store(order, .release);
    }

    /// Unpublish the current order, and free anything retired that no reader
    /// can still be inside. `readers` is `active_searches`, read *here*, after
    /// the swap: it used to arrive as a value sampled before it, and a scroll
    /// that took its guard and loaded the order in that gap had the order
    /// freed under its binary search. `reclaimRetired` reads its counter after
    /// publication for the same reason; this is the same rule.
    pub fn retire(self: *Scroll, alloc: std.mem.Allocator, readers: *const std.atomic.Value(usize)) void {
        if (self.published.swap(null, .acq_rel)) |old| {
            self.retired.append(alloc, old) catch {
                // Keep it alive rather than free it under a reader; `deinit`
                // cannot reach it from here, so this is a bounded leak on OOM,
                // which is the lesser evil.
                return;
            };
        }
        if (self.retired.items.len == 0) return;
        if (readers.load(.acquire) != 0) return;
        for (self.retired.items) |o| destroyScrollOrder(alloc, o);
        self.retired.clearRetainingCapacity();
    }

    /// Free everything, published included. Only when nothing else runs.
    pub fn deinit(self: *Scroll, alloc: std.mem.Allocator) void {
        if (self.published.swap(null, .acq_rel)) |o| destroyScrollOrder(alloc, o);
        for (self.retired.items) |o| destroyScrollOrder(alloc, o);
        self.retired.deinit(alloc);
    }
};

fn destroyScrollOrder(alloc: std.mem.Allocator, o: *ScrollOrder) void {
    alloc.free(o.items);
    alloc.destroy(o);
}

/// Build (or return) the id-ordered offset list.
///
/// The fast path is one acquire load: after the first page of an epoch every
/// scroll finds the order published and allocates nothing (§6.3). The build
/// itself runs under the write lock, which is what makes it safe to read the
/// id space and to publish: two readers cannot both build, and no upsert can
/// publish an offset between the count being read and the order being
/// stored, which used to leave an order one point short that no mutation
/// would ever drop. The cost is one sort's worth of ingest stall, once per
/// mutation epoch, on a workload that interleaves scrolling with writes.
///
/// The caller must hold `SearchGuard` for as long as it uses the result, and
/// must **not** hold it while this builds: `beginInPlace` drains
/// `active_searches` with the write lock held, so a guard-holder waiting for
/// the write lock here would deadlock against the first in-place upsert.
/// `scroll` does the dance; test callers with no concurrent writer need not.
pub fn scrollOrder(coll: *Collection) ![]const u32 {
    if (coll.scroll_state.current()) |o| return o.items;

    coll.write_lock.lock();
    defer coll.write_lock.unlock();
    // Another reader may have built it while this one waited for the lock.
    if (coll.scroll_state.current()) |o| return o.items;

    const n = coll.id_space.count();
    const items = try coll.alloc.alloc(u32, n);
    errdefer coll.alloc.free(items);
    for (items, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, items, ScrollCtx{ .space = &coll.id_space }, ScrollCtx.lessThan);
    const order = try coll.alloc.create(ScrollOrder);
    order.* = .{ .items = items };
    coll.scroll_state.publish(order);
    return items;
}

/// Retire the current order. Called with the write lock held (or from
/// `deinit`, when nothing else runs).
///
/// Never frees the order it unpublishes: a scroll on another worker may be
/// walking it. It goes on `Scroll.retired`, and everything on that list is
/// freed here the next time no reader is inside a search or a scroll, the
/// same quiescence rule `reclaimRetired` applies to graphs. In the common
/// case, a write with no concurrent scroll, that is immediately.
pub fn dropScrollOrder(coll: *Collection) void {
    coll.scroll_state.retire(coll.alloc, &coll.active_searches);
}

/// One page of `Scroll`.
pub const Page = struct {
    /// Ids in this page, borrowed from the caller's buffer.
    ids: []ExternalId,
    /// The id to resume from, or null when the collection is exhausted.
    next: ?ExternalId,
};

/// §12 phase 2: page through points in id order, skipping tombstones.
///
/// `out` bounds the page. `start` is inclusive, so a cursor returned by the
/// previous call resumes exactly where it left off with no point repeated and
/// none skipped.
pub fn scroll(coll: *Collection, start: ?ExternalId, out: []ExternalId) !Page {
    return scrollFiltered(coll, start, null, out);
}

/// `scroll` admitting only what `filter` admits, tombstones aside. The
/// cursor is the next *admitted* point, for the same reason it is the next
/// live one: a cursor onto a point the filter rejects yields an empty page.
pub fn scrollFiltered(coll: *Collection, start: ?ExternalId, filter: ?hnsw.Index.Filter, out: []ExternalId) !Page {
    // Held for the whole page: `dropScrollOrder` frees a retired order only
    // when this counter is zero. Taken *after* the order exists, see
    // `scrollOrder` for why the build must not run under it; the loop covers
    // a mutation landing between the build and the guard.
    const order: []const u32 = while (true) {
        const guard = SearchGuard.begin(coll);
        if (coll.scroll_state.current()) |o| break o.items;
        guard.end();
        _ = try scrollOrder(coll);
    };
    defer SearchGuard.end(.{ .coll = coll });
    // Built under the write lock, so it is exactly as long as the id space
    // was then; a later append drops it before publishing, so it is either
    // current or already unpublished, never short.
    std.debug.assert(order.len <= coll.id_space.count());

    // Binary search for the first id >= start. Linear scanning from the
    // beginning would make a full walk quadratic in the number of pages, which
    // is precisely the traversal W13's `sequential` mode measures.
    var lo: usize = 0;
    if (start) |s| {
        var hi = order.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (idLess(coll.id_space.external(order[mid]), s)) lo = mid + 1 else hi = mid;
        }
    }

    var n: usize = 0;
    var i = lo;
    while (i < order.len and n < out.len) : (i += 1) {
        const off = order[i];
        if (coll.deleted.isSet(off)) continue;
        if (filter) |f| if (!f.admits(off)) continue;
        out[n] = coll.id_space.external(off);
        n += 1;
    }

    // The cursor is the next *live* point, so a page boundary landing on a run
    // of tombstones does not hand back a cursor that yields an empty page.
    var next: ?ExternalId = null;
    while (i < order.len) : (i += 1) {
        if (coll.deleted.isSet(order[i])) continue;
        if (filter) |f| if (!f.admits(order[i])) continue;
        next = coll.id_space.external(order[i]);
        break;
    }

    return .{ .ids = out[0..n], .next = next };
}

// =========================================================================
// Tests
// =========================================================================

const testing = std.testing;
const makeCollection = collection.makeCollection;

test "a scroll order is retired, not freed, while a page is being read" {
    var c = try makeCollection(4, .dot, 64);
    defer c.deinit();
    var v = [_]f32{ 1, 0, 0, 0 };
    for (0..10) |i| _ = try c.upsert(.{ .num = 9 - i }, &v);

    const order = try scrollOrder(&c);
    try testing.expectEqual(@as(usize, 10), order.len);
    const published = c.scroll_state.current().?;

    // A page in flight on another worker.
    const guard = SearchGuard.begin(&c);
    // A write drops the order. It must stay allocated and readable.
    _ = try c.upsert(.{ .num = 100 }, &v);
    try testing.expect(c.scroll_state.current() == null);
    try testing.expectEqual(@as(usize, 1), c.scroll_state.retired.items.len);
    try testing.expectEqual(published, c.scroll_state.retired.items[0]);
    // Still the id order it was: offset of id 0 is 9.
    try testing.expectEqual(@as(u32, 9), published.items[0]);
    guard.end();

    // With no reader inside, the next mutation reclaims it.
    _ = try c.upsert(.{ .num = 101 }, &v);
    try testing.expectEqual(@as(usize, 0), c.scroll_state.retired.items.len);

    // And a fresh page sees the new points.
    var page_buf: [16]ExternalId = undefined;
    const page = try scroll(&c, null, &page_buf);
    try testing.expectEqual(@as(usize, 12), page.ids.len);
}
