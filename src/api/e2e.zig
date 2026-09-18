//! End-to-end tests: a real socket, a real HTTP/2 handshake, real gRPC frames.
//!
//! The client that drives them is `e2e_client.zig`. Everything below is a test.

const std = @import("std");
const linux = std.os.linux;
const strawmann_net = @import("../net/net.zig");
const proto = @import("../proto/proto.zig");
const api = @import("handlers.zig");
const dist = @import("../dist/dist.zig");
const core = @import("../core/core.zig");
const h2 = strawmann_net.h2;
const grpc = strawmann_net.grpc;
const server = strawmann_net.server;
const wire = proto.wire;
const msg = proto.messages;

const client_mod = @import("e2e_client.zig");
const Client = client_mod.Client;
const Harness = client_mod.Harness;
const buildCreateCollection = client_mod.buildCreateCollection;
const buildCreateCollectionDt = client_mod.buildCreateCollectionDt;
const buildUpsert = client_mod.buildUpsert;
const buildQueryBatch = client_mod.buildQueryBatch;

// =========================================================================
// Tests
// =========================================================================

const testing = std.testing;

test "e2e: HealthCheck, the RPC on every connection's critical path" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();

    var c = try Client.connect(h.port);
    defer c.close();

    var out: [4096]u8 = undefined;
    const resp = try c.call("/qdrant.Qdrant/HealthCheck", &.{}, &out);

    try testing.expectEqual(grpc.Status.ok, resp.status);
    // §8.5 T0: the *shape* matters, not only the values. A success is HEADERS,
    // DATA, trailers, two header frames and one data frame.
    try testing.expectEqual(@as(usize, 2), resp.headers_frames);
    try testing.expectEqual(@as(usize, 1), resp.data_frames);
    try testing.expect(resp.saw_end_stream);

    // §2: qdrant-client compares the returned version against its own
    // major/minor, so it must be present and parseable.
    var r = wire.Reader.init(resp.body);
    var title: []const u8 = &.{};
    var version: []const u8 = &.{};
    while (!r.atEnd()) {
        const t = try r.tag();
        switch (t.field) {
            1 => title = try r.bytes(),
            2 => version = try r.bytes(),
            else => try r.skip(t.wire_type),
        }
    }
    try testing.expectEqualStrings("strawmann", title);
    try testing.expect(version.len > 0);
    // Must look like `major.minor.patch` or qdrant-client's check misparses.
    try testing.expect(std.mem.count(u8, version, ".") >= 1);
}

test "e2e: the collection lifecycle bfb performs at startup" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();

    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [4096]u8 = undefined;
    var out: [4096]u8 = undefined;

    // CollectionExists on a collection that does not exist (--create-if-missing).
    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "bench");
        const resp = try c.call("/qdrant.Collections/CollectionExists", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);

        var r = wire.Reader.init(resp.body);
        const t = try r.tag();
        try testing.expectEqual(@as(u32, 1), t.field);
        var sub = try r.nested();
        var exists = false;
        while (!sub.atEnd()) {
            const st = try sub.tag();
            if (st.field == 1) exists = try sub.boolean() else try sub.skip(st.wire_type);
        }
        try testing.expect(!exists);
    }

    // Delete (bfb issues this unconditionally during setup).
    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "bench");
        const resp = try c.call("/qdrant.Collections/Delete", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }

    // Create, Cosine, d=8.
    {
        const body = try buildCreateCollection(&req_buf, "bench", 8, 1);
        const resp = try c.call("/qdrant.Collections/Create", body, &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }

    // Now it exists.
    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "bench");
        const resp = try c.call("/qdrant.Collections/CollectionExists", w.written(), &out);
        var r = wire.Reader.init(resp.body);
        _ = try r.tag();
        var sub = try r.nested();
        var exists = false;
        while (!sub.atEnd()) {
            const st = try sub.tag();
            if (st.field == 1) exists = try sub.boolean() else try sub.skip(st.wire_type);
        }
        try testing.expect(exists);
    }

    // §2: the `wait_index` poll loop reads CollectionInfo and requires
    // status == Green three consecutive times, with optimizer_status Ok.
    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "bench");
        const resp = try c.call("/qdrant.Collections/Get", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);

        var r = wire.Reader.init(resp.body);
        const t = try r.tag();
        try testing.expectEqual(@as(u32, 1), t.field);
        var info = try r.nested();

        var status: u64 = 0;
        var optimizer_ok = false;
        var saw_optimizer = false;
        var points: ?u64 = null;
        var segments: u64 = 0;
        while (!info.atEnd()) {
            const it = try info.tag();
            switch (it.field) {
                1 => status = try info.varint(),
                2 => {
                    saw_optimizer = true;
                    var os = try info.nested();
                    while (!os.atEnd()) {
                        const ot = try os.tag();
                        if (ot.field == 1) optimizer_ok = try os.boolean() else try os.skip(ot.wire_type);
                    }
                },
                4 => segments = try info.varint(),
                9 => points = try info.varint(),
                else => try info.skip(it.wire_type),
            }
        }
        try testing.expectEqual(@as(u64, 1), status); // Green
        try testing.expect(saw_optimizer);
        try testing.expect(optimizer_ok);
        // Present and zero, not absent: `points_count` is `optional uint64`
        // and Qdrant always sends it. The assertion used to compare against
        // the decoder's own initialiser and could not fail.
        try testing.expectEqual(@as(?u64, 0), points);
        try testing.expectEqual(@as(u64, 1), segments); // §2's declared gap
    }
}

test "e2e: upsert then QueryBatch returns the true nearest neighbours" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();

    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    // Dot metric (3) so the expected ordering is easy to state exactly.
    {
        const body = try buildCreateCollection(&req_buf, "bench", 4, 3);
        const resp = try c.call("/qdrant.Collections/Create", body, &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }

    const vecs = [_][4]f32{
        .{ 1, 0, 0, 0 },
        .{ 0.9, 0.1, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 0.5, 0.5, 0, 0 },
    };
    var slices: [5][]const f32 = undefined;
    for (&slices, 0..) |*s, i| s.* = &vecs[i];
    const ids = [_]u64{ 10, 11, 12, 13, 14 };

    {
        const body = try buildUpsert(&req_buf, "bench", true, &ids, &slices);
        const resp = try c.call("/qdrant.Points/Upsert", body, &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);

        // PointsOperationResponse { UpdateResult result = 1 { op_id, status } }
        var r = wire.Reader.init(resp.body);
        const t = try r.tag();
        try testing.expectEqual(@as(u32, 1), t.field);
        var ur = try r.nested();
        var status: u64 = 0;
        while (!ur.atEnd()) {
            const ut = try ur.tag();
            if (ut.field == 2) status = try ur.varint() else try ur.skip(ut.wire_type);
        }
        // wait: true -> Completed (2).
        try testing.expectEqual(@as(u64, 2), status);
    }

    // Query [1,0,0,0]: expect 10 (dot 1.0), 11 (0.9), 14 (0.5).
    {
        const q = [_]f32{ 1, 0, 0, 0 };
        var qs: [1][]const f32 = .{&q};
        const body = try buildQueryBatch(&req_buf, "bench", &qs, 3);
        const resp = try c.call("/qdrant.Points/QueryBatch", body, &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);

        // QueryBatchResponse { repeated BatchResult result = 1 }
        var r = wire.Reader.init(resp.body);
        const t = try r.tag();
        try testing.expectEqual(@as(u32, 1), t.field);
        var batch = try r.nested();

        var got_ids: [8]u64 = undefined;
        var got_scores: [8]f32 = undefined;
        var n: usize = 0;
        while (!batch.atEnd()) {
            const bt = try batch.tag();
            if (bt.field != 1) {
                try batch.skip(bt.wire_type);
                continue;
            }
            var sp = try batch.nested();
            while (!sp.atEnd()) {
                const st = try sp.tag();
                switch (st.field) {
                    1 => {
                        var idr = try sp.nested();
                        const pid = try msg.PointId.decode(&idr);
                        got_ids[n] = pid.num;
                    },
                    3 => got_scores[n] = try sp.float(),
                    else => try sp.skip(st.wire_type),
                }
            }
            n += 1;
        }

        try testing.expectEqual(@as(usize, 3), n);
        try testing.expectEqual(@as(u64, 10), got_ids[0]);
        try testing.expectEqual(@as(u64, 11), got_ids[1]);
        try testing.expectEqual(@as(u64, 14), got_ids[2]);
        try testing.expectApproxEqAbs(@as(f32, 1.0), got_scores[0], 1e-6);
        try testing.expectApproxEqAbs(@as(f32, 0.9), got_scores[1], 1e-6);
        try testing.expectApproxEqAbs(@as(f32, 0.5), got_scores[2], 1e-6);
    }
}

test "e2e: §8.3 Euclid returns the square root while ranking on the square" {
    // The trap that would otherwise show up as a structured error distribution
    // in the differ. Ranking must be by increasing distance and the returned
    // score must be the distance itself, not its square and not its negation.
    var h = try Harness.start(testing.allocator);
    defer h.stop();

    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "e", 4, 2), &out);

    const vecs = [_][4]f32{
        .{ 3, 4, 0, 0 }, // distance 5 from origin
        .{ 1, 0, 0, 0 }, // distance 1
        .{ 0, 0, 6, 8 }, // distance 10
    };
    var slices: [3][]const f32 = undefined;
    for (&slices, 0..) |*s, i| s.* = &vecs[i];
    const ids = [_]u64{ 1, 2, 3 };
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "e", true, &ids, &slices), &out);

    const q = [_]f32{ 0, 0, 0, 0 };
    var qs: [1][]const f32 = .{&q};
    const resp = try c.call("/qdrant.Points/QueryBatch", try buildQueryBatch(&req_buf, "e", &qs, 3), &out);
    try testing.expectEqual(grpc.Status.ok, resp.status);

    var r = wire.Reader.init(resp.body);
    _ = try r.tag();
    var batch = try r.nested();
    var got_ids: [4]u64 = undefined;
    var got_scores: [4]f32 = undefined;
    var n: usize = 0;
    while (!batch.atEnd()) {
        const bt = try batch.tag();
        if (bt.field != 1) {
            try batch.skip(bt.wire_type);
            continue;
        }
        var sp = try batch.nested();
        while (!sp.atEnd()) {
            const st = try sp.tag();
            switch (st.field) {
                1 => {
                    var idr = try sp.nested();
                    got_ids[n] = (try msg.PointId.decode(&idr)).num;
                },
                3 => got_scores[n] = try sp.float(),
                else => try sp.skip(st.wire_type),
            }
        }
        n += 1;
    }

    try testing.expectEqual(@as(usize, 3), n);
    // Nearest first, by actual distance.
    try testing.expectEqual(@as(u64, 2), got_ids[0]);
    try testing.expectEqual(@as(u64, 1), got_ids[1]);
    try testing.expectEqual(@as(u64, 3), got_ids[2]);
    // Returned scores are distances, not squared distances.
    try testing.expectApproxEqAbs(@as(f32, 1.0), got_scores[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 5.0), got_scores[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 10.0), got_scores[2], 1e-6);
}

test "e2e: §1 non-goals return UNIMPLEMENTED naming the construct" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();

    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [4096]u8 = undefined;
    var out: [4096]u8 = undefined;

    // An RPC we do not implement, listed in §2 against a later phase.
    //
    // §12 asks for UNIMPLEMENTED "with a message naming the RPC", and the
    // message has to name *which* one: bfb reaches half a dozen unimplemented
    // paths and a generic string sends the reader back to the client's source
    // to work out which call failed.
    {
        const resp = try c.call("/qdrant.Points/Delete", &.{}, &out);
        try testing.expectEqual(grpc.Status.unimplemented, resp.status);
        try testing.expect(std.mem.indexOf(u8, resp.message, "qdrant.Points/Delete") != null);
        // Deferred, not refused. Sending a client to "§1 non-goals" for a
        // scheduled RPC is a wrong answer that reads like a right one.
        try testing.expect(std.mem.indexOf(u8, resp.message, "§2") != null);
        try testing.expect(std.mem.indexOf(u8, resp.message, "§1") == null);
        // Trailers-only: no DATA frame at all.
        try testing.expectEqual(@as(usize, 0), resp.data_frames);
        try testing.expectEqual(@as(usize, 1), resp.headers_frames);
    }

    // An RPC that is genuinely a §1 non-goal, not a deferred phase, must be
    // attributed to §1 instead.
    {
        const resp = try c.call("/qdrant.Snapshots/Create", &.{}, &out);
        try testing.expectEqual(grpc.Status.unimplemented, resp.status);
        try testing.expect(std.mem.indexOf(u8, resp.message, "qdrant.Snapshots/Create") != null);
        try testing.expect(std.mem.indexOf(u8, resp.message, "§1") != null);
    }

    // A supported RPC carrying an unsupported construct: a fusion query.
    {
        _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "u", 4, 3), &out);

        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "u");
        const qp = try w.beginNested(2, 3);
        try w.writeStringField(1, "u");
        const query = try w.beginNested(3, 3);
        try w.writeVarintFieldAlways(6, 0); // Query.fusion
        try w.endNested(query);
        try w.endNested(qp);

        const resp = try c.call("/qdrant.Points/QueryBatch", w.written(), &out);
        try testing.expectEqual(grpc.Status.unimplemented, resp.status);
        try testing.expectEqualStrings("fusion (RRF) queries", resp.message);
    }

    // Filtering is phase 3 and must not be silently ignored, ignoring it would
    // return more results than asked for and quietly invalidate W12. An
    // *empty* `Filter{}` carries no condition, though, and Qdrant answers it
    // as no filter; it used to be refused here too, and this test pinned that.
    for ([_]bool{ false, true }) |with_condition| {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "u");
        const qp = try w.beginNested(2, 3);
        try w.writeStringField(1, "u");
        {
            const query = try w.beginNested(3, 3);
            const nearest = try w.beginNested(1, 3);
            const dv = try w.beginNested(2, 3);
            try w.writePackedFloats(1, &[_]f32{ 1, 0, 0, 0 });
            try w.endNested(dv);
            try w.endNested(nearest);
            try w.endNested(query);
        }
        {
            const f = try w.beginNested(5, 2); // filter (VERIFIED: field 5)
            if (with_condition) {
                // `Filter.must = 2`, one `Condition { FieldCondition field = 1 }`
                // with a key: the smallest filter that actually filters.
                const cond = try w.beginNested(2, 2);
                const fc = try w.beginNested(1, 2);
                try w.writeStringField(1, "k");
                try w.endNested(fc);
                try w.endNested(cond);
            }
            try w.endNested(f);
        }
        try w.endNested(qp);

        const resp = try c.call("/qdrant.Points/QueryBatch", w.written(), &out);
        if (with_condition) {
            // A `FieldCondition` with a key and no `match` is not a filter
            // this engine can evaluate, and not one Qdrant accepts either.
            try testing.expectEqual(grpc.Status.invalid_argument, resp.status);
            try testing.expect(std.mem.indexOf(u8, resp.message, "FieldCondition without a match") != null);
        } else {
            try testing.expectEqual(grpc.Status.ok, resp.status);
        }
    }

    // `params.indexed_only` asks to skip the unindexed tail, which this
    // engine cannot do; ignoring it would return points the client asked not
    // to see, so it is refused by name like `hnsw_config.on_disk`.
    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "u");
        const qp = try w.beginNested(2, 3);
        try w.writeStringField(1, "u");
        {
            const query = try w.beginNested(3, 3);
            const nearest = try w.beginNested(1, 3);
            const dv = try w.beginNested(2, 3);
            try w.writePackedFloats(1, &[_]f32{ 1, 0, 0, 0 });
            try w.endNested(dv);
            try w.endNested(nearest);
            try w.endNested(query);
        }
        {
            const params = try w.beginNested(6, 2);
            try w.writeBoolField(4, true); // indexed_only
            try w.endNested(params);
        }
        try w.endNested(qp);
        const resp = try c.call("/qdrant.Points/QueryBatch", w.written(), &out);
        try testing.expectEqual(grpc.Status.unimplemented, resp.status);
        try testing.expectEqualStrings("params.indexed_only", resp.message);
    }

    // The legacy single-query spellings are aliases of QueryBatch, not §1
    // non-goals, and the message has to point a client at the right endpoint.
    for ([_][]const u8{ "/qdrant.Points/Search", "/qdrant.Points/SearchBatch", "/qdrant.Points/Query" }) |path| {
        const resp = try c.call(path, &.{}, &out);
        try testing.expectEqual(grpc.Status.unimplemented, resp.status);
        try testing.expect(std.mem.indexOf(u8, resp.message, path[1..]) != null);
        try testing.expect(std.mem.indexOf(u8, resp.message, "QueryBatch") != null);
        try testing.expect(std.mem.indexOf(u8, resp.message, "§1") == null);
    }
}

test "e2e: unknown collection is NOT_FOUND, not a hang" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1024]u8 = undefined;
    var out: [4096]u8 = undefined;
    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "nope");

    const resp = try c.call("/qdrant.Collections/Get", w.written(), &out);
    try testing.expectEqual(grpc.Status.not_found, resp.status);
}

test "e2e: many sequential RPCs on one connection (bfb's -c 1 default)" {
    // §6.1: "`-c 1` is the bfb default. One connection means all concurrency is
    // multiplexed over a single TCP stream." This exercises stream-slot reuse
    // and HPACK dynamic-table continuity across many requests.
    var h = try Harness.start(testing.allocator);
    defer h.stop();

    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "many", 4, 3), &out);

    // Well beyond `streams_per_conn = 4`, so slots must be recycled.
    for (0..64) |i| {
        const v = [_]f32{ @floatFromInt(i), 1, 2, 3 };
        var slices: [1][]const f32 = .{&v};
        const one = [_]u64{i};
        const resp = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "many", true, &one, &slices), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }

    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "many");
    const info = try c.call("/qdrant.Collections/Get", w.written(), &out);
    var r = wire.Reader.init(info.body);
    _ = try r.tag();
    var sub = try r.nested();
    var points: u64 = 0;
    while (!sub.atEnd()) {
        const t = try sub.tag();
        if (t.field == 9) points = try sub.varint() else try sub.skip(t.wire_type);
    }
    try testing.expectEqual(@as(u64, 64), points);
}

test "e2e: a batch of queries returns one BatchResult per query, in order" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "b", 4, 3), &out);

    const vecs = [_][4]f32{ .{ 1, 0, 0, 0 }, .{ 0, 1, 0, 0 }, .{ 0, 0, 1, 0 } };
    var slices: [3][]const f32 = undefined;
    for (&slices, 0..) |*s, i| s.* = &vecs[i];
    const ids = [_]u64{ 100, 200, 300 };
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "b", true, &ids, &slices), &out);

    // Three queries, each of which should return its own vector first.
    const q0 = [_]f32{ 1, 0, 0, 0 };
    const q1 = [_]f32{ 0, 1, 0, 0 };
    const q2 = [_]f32{ 0, 0, 1, 0 };
    var qs: [3][]const f32 = .{ &q0, &q1, &q2 };

    const resp = try c.call("/qdrant.Points/QueryBatch", try buildQueryBatch(&req_buf, "b", &qs, 1), &out);
    try testing.expectEqual(grpc.Status.ok, resp.status);

    var r = wire.Reader.init(resp.body);
    var first_ids: [8]u64 = undefined;
    var batches: usize = 0;
    while (!r.atEnd()) {
        const t = try r.tag();
        if (t.field != 1) {
            try r.skip(t.wire_type);
            continue;
        }
        var batch = try r.nested();
        while (!batch.atEnd()) {
            const bt = try batch.tag();
            if (bt.field != 1) {
                try batch.skip(bt.wire_type);
                continue;
            }
            var sp = try batch.nested();
            while (!sp.atEnd()) {
                const st = try sp.tag();
                if (st.field == 1) {
                    var idr = try sp.nested();
                    first_ids[batches] = (try msg.PointId.decode(&idr)).num;
                } else try sp.skip(st.wire_type);
            }
        }
        batches += 1;
    }

    try testing.expectEqual(@as(usize, 3), batches);
    try testing.expectEqual(@as(u64, 100), first_ids[0]);
    try testing.expectEqual(@as(u64, 200), first_ids[1]);
    try testing.expectEqual(@as(u64, 300), first_ids[2]);
}

/// Read every `BatchResult` of a `QueryBatchResponse`: `ids[k][..]` and
/// `scores[k][..]` for batch k, `counts[k]` how many. Returns the batch count.
fn readAllBatches(body: []const u8, ids: [][]u64, scores: [][]f32, counts: []usize) !usize {
    var r = wire.Reader.init(body);
    var b: usize = 0;
    while (!r.atEnd()) {
        const t = try r.tag();
        if (t.field != 1) {
            try r.skip(t.wire_type);
            continue;
        }
        var batch = try r.nested();
        var n: usize = 0;
        while (!batch.atEnd()) {
            const bt = try batch.tag();
            if (bt.field != 1) {
                try batch.skip(bt.wire_type);
                continue;
            }
            var sp = try batch.nested();
            var id: u64 = 0;
            var score: f32 = 0;
            while (!sp.atEnd()) {
                const st = try sp.tag();
                switch (st.field) {
                    1 => {
                        var idr = try sp.nested();
                        id = (try msg.PointId.decode(&idr)).num;
                    },
                    3 => score = try sp.float(),
                    else => try sp.skip(st.wire_type),
                }
            }
            if (b < ids.len and n < ids[b].len) {
                ids[b][n] = id;
                scores[b][n] = score;
            }
            n += 1;
        }
        if (b < counts.len) counts[b] = n;
        b += 1;
    }
    return b;
}

/// Create `name` (d=16, Euclid), upsert `n` random vectors under ids 0..n,
/// wait until the collection is Green, and return the stored vectors.
fn fillIndexed(alloc: std.mem.Allocator, c: *Client, name: []const u8, n: usize, seed: u64) ![]f32 {
    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;
    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, name, 16, 2), &out);
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const stored = try alloc.alloc(f32, n * 16);
    errdefer alloc.free(stored);
    for (stored) |*x| x.* = rnd.floatNorm(f32);
    var batch_ids: [50]u64 = undefined;
    var batch_vecs: [50][]const f32 = undefined;
    var i: usize = 0;
    while (i < n) : (i += 50) {
        const m = @min(50, n - i);
        for (0..m) |j| {
            batch_ids[j] = i + j;
            batch_vecs[j] = stored[(i + j) * 16 ..][0..16];
        }
        const resp = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, name, true, batch_ids[0..m], batch_vecs[0..m]), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }
    // Green three times, the way bfb polls; a build racing the queries
    // below would make graph and brute-force answers differ mid-test.
    var greens: usize = 0;
    var polls: usize = 0;
    while (greens < 3 and polls < 4000) : (polls += 1) {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, name);
        const resp = try c.call("/qdrant.Collections/Get", w.written(), &out);
        var r = wire.Reader.init(resp.body);
        _ = try r.tag();
        var info = try r.nested();
        var status: u64 = 0;
        while (!info.atEnd()) {
            const t = try info.tag();
            if (t.field == 1) status = try info.varint() else try info.skip(t.wire_type);
        }
        greens = if (status == 1) greens + 1 else 0;
    }
    try testing.expectEqual(@as(usize, 3), greens);
    return stored;
}

test "e2e: a QueryBatch fanned out across workers answers exactly what one query per request does" {
    // The dispatching worker used to run a batch's queries one after
    // another on its own core. Now queries 1..n-1 go back on the worker
    // queue and run wherever they are picked up (`handlers.BatchJob`), so
    // the property to pin is that nothing about the answer changed: same
    // ids, same scores, same order per query, and the batches in wire order.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    const n = 600;
    const stored = try fillIndexed(testing.allocator, &c, "fan", n, 0xfa17);
    defer testing.allocator.free(stored);

    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;
    const k = 7;
    const nq = 64;

    // One query per request: the reference.
    var ref_ids: [nq][k]u64 = undefined;
    var ref_scores: [nq][k]f32 = undefined;
    var ref_counts: [nq]usize = undefined;
    var qs: [nq][]const f32 = undefined;
    for (0..nq) |qi| {
        qs[qi] = stored[((qi * 13 + 5) % n) * 16 ..][0..16];
        var one: [1][]const f32 = .{qs[qi]};
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildQueryBatch(&req_buf, "fan", &one, k), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        var ids: [1][]u64 = .{&ref_ids[qi]};
        var scores: [1][]f32 = .{&ref_scores[qi]};
        var counts: [1]usize = undefined;
        try testing.expectEqual(@as(usize, 1), try readAllBatches(resp.body, &ids, &scores, &counts));
        ref_counts[qi] = counts[0];
        try testing.expectEqual(@as(usize, k), counts[0]);
        // The query is a stored vector, so it leads its own result.
        try testing.expectEqual(@as(u64, (qi * 13 + 5) % n), ref_ids[qi][0]);
    }

    // The same 64 as one batch, fanned out.
    const before = api.batches_fanned_out.load(.monotonic);
    const resp = try c.call("/qdrant.Points/QueryBatch", try buildQueryBatch(&req_buf, "fan", &qs, k), &out);
    try testing.expectEqual(grpc.Status.ok, resp.status);
    try testing.expectEqual(before + 1, api.batches_fanned_out.load(.monotonic));
    var got_ids: [nq][k]u64 = undefined;
    var got_scores: [nq][k]f32 = undefined;
    var got_counts: [nq]usize = undefined;
    var ids_v: [nq][]u64 = undefined;
    var scores_v: [nq][]f32 = undefined;
    for (0..nq) |i| {
        ids_v[i] = &got_ids[i];
        scores_v[i] = &got_scores[i];
    }
    try testing.expectEqual(@as(usize, nq), try readAllBatches(resp.body, &ids_v, &scores_v, &got_counts));
    for (0..nq) |qi| {
        try testing.expectEqual(ref_counts[qi], got_counts[qi]);
        try testing.expectEqualSlices(u64, &ref_ids[qi], &got_ids[qi]);
        try testing.expectEqualSlices(f32, &ref_scores[qi], &got_scores[qi]);
    }
    // And `time` is still on the wire (field 2, fixed64) for the detached
    // completion, appended by the finishing worker rather than the loop.
    {
        var r = wire.Reader.init(resp.body);
        var saw_time = false;
        while (!r.atEnd()) {
            const t = try r.tag();
            if (t.field == 2 and t.wire_type == .fixed64) {
                const secs: f64 = @bitCast(try r.fixed64());
                try testing.expect(secs > 0 and secs < 10);
                saw_time = true;
            } else try r.skip(t.wire_type);
        }
        try testing.expect(saw_time);
    }
}

test "e2e: a batch that does not fit its request buffer's spare bytes runs sequentially, unchanged" {
    // The job lives after the body in the stream's request buffer; a
    // buffer too small for it must not refuse the batch, only stop
    // parallelising it. 8 KiB leaves less than the job header, so nothing
    // fans out and the answers are the same.
    var h = try Harness.startWith(testing.allocator, 8192);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    const n = 200;
    const stored = try fillIndexed(testing.allocator, &c, "seq", n, 0x5e9);
    defer testing.allocator.free(stored);

    var req_buf: [8192]u8 = undefined;
    var out: [1 << 16]u8 = undefined;
    const nq = 8;
    var qs: [nq][]const f32 = undefined;
    for (0..nq) |qi| qs[qi] = stored[(qi * 17 % n) * 16 ..][0..16];
    const before = api.batches_fanned_out.load(.monotonic);
    const resp = try c.call("/qdrant.Points/QueryBatch", try buildQueryBatch(&req_buf, "seq", &qs, 3), &out);
    try testing.expectEqual(grpc.Status.ok, resp.status);
    try testing.expectEqual(before, api.batches_fanned_out.load(.monotonic));
    var ids: [nq][3]u64 = undefined;
    var scores: [nq][3]f32 = undefined;
    var counts: [nq]usize = undefined;
    var ids_v: [nq][]u64 = undefined;
    var scores_v: [nq][]f32 = undefined;
    for (0..nq) |i| {
        ids_v[i] = &ids[i];
        scores_v[i] = &scores[i];
    }
    try testing.expectEqual(@as(usize, nq), try readAllBatches(resp.body, &ids_v, &scores_v, &counts));
    for (0..nq) |qi| try testing.expectEqual(@as(u64, qi * 17 % n), ids[qi][0]);
}

test "e2e: one refused query fails the whole fanned-out batch, as Qdrant does" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    const n = 300;
    const stored = try fillIndexed(testing.allocator, &c, "bad", n, 0xbad);
    defer testing.allocator.free(stored);

    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 16]u8 = undefined;
    const nq = 40;
    var qs: [nq][]const f32 = undefined;
    for (0..nq) |qi| qs[qi] = stored[(qi % n) * 16 ..][0..16];
    // Query 20 has the wrong dimension: refused by name, whichever worker
    // meets it, and the batch carries no partial result.
    const short = [_]f32{ 1, 2, 3 };
    qs[20] = &short;
    const resp = try c.call("/qdrant.Points/QueryBatch", try buildQueryBatch(&req_buf, "bad", &qs, 5), &out);
    try testing.expectEqual(grpc.Status.invalid_argument, resp.status);
    try testing.expect(std.mem.indexOf(u8, resp.message, "dimension") != null);
    try testing.expectEqual(@as(usize, 0), resp.body.len);
}

test "e2e: fanned-out batches from concurrent connections interleave on the pool and stay correct" {
    // Two connections each send batches back to back while two workers
    // serve both; sub-requests of one batch queue behind pieces of the
    // other. Every answer must still lead with the query's own id.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    const n = 400;
    // Filled through a connection that is closed before the runners start:
    // the harness has two connection slots and each runner wants one.
    const stored = blk: {
        var c0 = try Client.connect(h.port);
        defer c0.close();
        break :blk try fillIndexed(testing.allocator, &c0, "mix", n, 0x1e5);
    };
    defer testing.allocator.free(stored);

    const Runner = struct {
        fn run(port: u16, vecs: []const f32, seed: usize, failures: *std.atomic.Value(u32)) void {
            var c = Client.connect(port) catch {
                _ = failures.fetchAdd(1, .monotonic);
                return;
            };
            defer c.close();
            var req_buf: [1 << 17]u8 = undefined;
            var out: [1 << 17]u8 = undefined;
            const nq = 24;
            var qs: [nq][]const f32 = undefined;
            var expect: [nq]u64 = undefined;
            for (0..40) |round| {
                for (0..nq) |qi| {
                    const id = (round * 31 + qi * 7 + seed) % n;
                    qs[qi] = vecs[id * 16 ..][0..16];
                    expect[qi] = id;
                }
                const body = buildQueryBatch(&req_buf, "mix", &qs, 3) catch {
                    _ = failures.fetchAdd(1, .monotonic);
                    return;
                };
                const resp = c.call("/qdrant.Points/QueryBatch", body, &out) catch {
                    _ = failures.fetchAdd(1, .monotonic);
                    return;
                };
                if (resp.status != .ok) {
                    _ = failures.fetchAdd(1, .monotonic);
                    return;
                }
                var ids: [nq][3]u64 = undefined;
                var scores: [nq][3]f32 = undefined;
                var counts: [nq]usize = undefined;
                var ids_v: [nq][]u64 = undefined;
                var scores_v: [nq][]f32 = undefined;
                for (0..nq) |i| {
                    ids_v[i] = &ids[i];
                    scores_v[i] = &scores[i];
                }
                const got = readAllBatches(resp.body, &ids_v, &scores_v, &counts) catch 0;
                if (got != nq) {
                    _ = failures.fetchAdd(1, .monotonic);
                    return;
                }
                for (0..nq) |qi| {
                    if (counts[qi] != 3 or ids[qi][0] != expect[qi]) {
                        _ = failures.fetchAdd(1, .monotonic);
                        return;
                    }
                }
            }
        }
    };
    var failures = std.atomic.Value(u32).init(0);
    const before = api.batches_fanned_out.load(.monotonic);
    var t1 = try std.Thread.spawn(.{}, Runner.run, .{ h.port, stored, @as(usize, 1), &failures });
    var t2 = try std.Thread.spawn(.{}, Runner.run, .{ h.port, stored, @as(usize, 2), &failures });
    t1.join();
    t2.join();
    try testing.expectEqual(@as(u32, 0), failures.load(.monotonic));
    try testing.expectEqual(before + 80, api.batches_fanned_out.load(.monotonic));
}

test "e2e: §2 wait_index, Yellow while building, Green once indexed" {
    // §2: "After upload, bfb polls `collection_info` once per second and
    // requires `status == Green` **three consecutive times**." The server must
    // therefore not report Green until the graph actually exists, §2 again:
    // "we must report `Green` only when actually indexed or the comparison is
    // meaningless."
    var h = try Harness.start(testing.allocator);
    defer h.stop();

    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "idx", 16, 2), &out);

    // Upload enough points that a graph is worth building.
    var prng = std.Random.DefaultPrng.init(0xbeef);
    const rnd = prng.random();
    const n = 600;
    var stored = try testing.allocator.alloc(f32, n * 16);
    defer testing.allocator.free(stored);
    for (stored) |*x| x.* = rnd.floatNorm(f32);

    {
        var batch_ids: [50]u64 = undefined;
        var batch_vecs: [50][]const f32 = undefined;
        var i: usize = 0;
        while (i < n) : (i += 50) {
            for (0..50) |j| {
                batch_ids[j] = i + j;
                batch_vecs[j] = stored[(i + j) * 16 ..][0..16];
            }
            const body = try buildUpsert(&req_buf, "idx", true, &batch_ids, &batch_vecs);
            const resp = try c.call("/qdrant.Points/Upsert", body, &out);
            try testing.expectEqual(grpc.Status.ok, resp.status);
        }
    }

    // Poll exactly the way bfb does: read status until Green three times.
    const readStatus = struct {
        fn f(cl: *Client, rb: []u8, ob: []u8) !struct { status: u64, indexed: u64 } {
            var w = wire.Writer.init(rb);
            try w.writeStringField(1, "idx");
            const resp = try cl.call("/qdrant.Collections/Get", w.written(), ob);
            var r = wire.Reader.init(resp.body);
            _ = try r.tag();
            var info = try r.nested();
            var status: u64 = 0;
            var indexed: u64 = 0;
            while (!info.atEnd()) {
                const t = try info.tag();
                switch (t.field) {
                    1 => status = try info.varint(),
                    10 => indexed = try info.varint(),
                    else => try info.skip(t.wire_type),
                }
            }
            return .{ .status = status, .indexed = indexed };
        }
    }.f;

    var greens: usize = 0;
    var polls: usize = 0;
    while (greens < 3 and polls < 2000) : (polls += 1) {
        const st = try readStatus(&c, &req_buf, &out);
        if (st.status == 1) {
            greens += 1;
            // Green must mean genuinely indexed, not merely "no build running".
            try testing.expectEqual(@as(u64, n), st.indexed);
        } else {
            greens = 0;
        }
    }
    try testing.expectEqual(@as(usize, 3), greens);
    // Yellow polls are informational: on a fast host the build can finish
    // before the first poll, so observing Yellow is not guaranteed. The
    // property §2 actually depends on is the one asserted in the loop, Green
    // never appears with `indexed < n`, and its complement: once Green with
    // nothing written since, it *stays* Green. bfb needs three in a row, and
    // a status that flickered back to Yellow would never satisfy it.
    for (0..20) |_| {
        const st = try readStatus(&c, &req_buf, &out);
        try testing.expectEqual(@as(u64, 1), st.status);
        try testing.expectEqual(@as(u64, n), st.indexed);
    }

    // And the indexed collection answers queries with high recall.
    const k = 10;
    var hits: usize = 0;
    const queries = 40;
    for (0..queries) |qi| {
        const q = stored[(qi * 7 % n) * 16 ..][0..16];
        var qs: [1][]const f32 = .{q};
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildQueryBatch(&req_buf, "idx", &qs, k), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);

        // Truth: the query is a stored vector, so its own id must come back.
        var r = wire.Reader.init(resp.body);
        _ = try r.tag();
        var batch = try r.nested();
        var first_id: u64 = std.math.maxInt(u64);
        var got: usize = 0;
        while (!batch.atEnd()) {
            const bt = try batch.tag();
            if (bt.field != 1) {
                try batch.skip(bt.wire_type);
                continue;
            }
            var sp = try batch.nested();
            while (!sp.atEnd()) {
                const st2 = try sp.tag();
                if (st2.field == 1) {
                    var idr = try sp.nested();
                    const pid = (try msg.PointId.decode(&idr)).num;
                    if (got == 0) first_id = pid;
                } else try sp.skip(st2.wire_type);
            }
            got += 1;
        }
        try testing.expectEqual(@as(usize, k), got);
        if (first_id == @as(u64, qi * 7 % n)) hits += 1;
    }
    // §8.6's self-retrieval canary, through the full stack. §8.6 is explicit
    // that "under ANN this becomes a recall canary rather than an assertion",
    // so this is a threshold rather than a demand for perfection, an occasional
    // miss is HNSW behaving normally, while a low rate means the graph is not
    // navigable.
    const self_rate = @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(queries));
    try testing.expect(self_rate >= 0.9);
}

test "e2e: exact search bypasses the index and matches brute force" {
    // §6.5: the exact path is the recall ground truth, so going through the
    // graph would make W9 measure the wrong thing.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;
    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "ex", 8, 3), &out);

    var prng = std.Random.DefaultPrng.init(0x1122);
    const rnd = prng.random();
    const n = 200;
    var stored = try testing.allocator.alloc(f32, n * 8);
    defer testing.allocator.free(stored);
    for (stored) |*x| x.* = rnd.floatNorm(f32);

    var ids: [200]u64 = undefined;
    var vecs: [200][]const f32 = undefined;
    for (0..n) |i| {
        ids[i] = i;
        vecs[i] = stored[i * 8 ..][0..8];
    }
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "ex", true, &ids, &vecs), &out);

    // Wait for the graph, or "bypasses the index" tests nothing: with no
    // index the exact path and the default path are the same brute force.
    _ = try waitGreen(&c, "ex", &req_buf, &out);
    try testing.expectEqual(@as(u64, n), (try getInfo(&c, "ex", &req_buf, &out)).indexed);
    try testing.expectEqual(core.collection.IndexState.ready, h.engine.find("ex").?.index_state.load(.acquire));

    // Build a QueryBatch with params.exact = true.
    const q = stored[0..8];
    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "ex");
    const qp = try w.beginNested(2, 3);
    try w.writeStringField(1, "ex");
    {
        const query = try w.beginNested(3, 3);
        const nearest = try w.beginNested(1, 3);
        const dv = try w.beginNested(2, 3); // VectorInput.dense (VERIFIED)
        try w.writePackedFloats(1, q);
        try w.endNested(dv);
        try w.endNested(nearest);
        try w.endNested(query);
    }
    {
        const params = try w.beginNested(6, 2);
        try w.writeBoolField(2, true); // exact
        try w.endNested(params);
    }
    try w.writeVarintFieldAlways(8, 5); // limit (VERIFIED: field 8)
    try w.endNested(qp);

    const resp = try c.call("/qdrant.Points/QueryBatch", w.written(), &out);
    try testing.expectEqual(grpc.Status.ok, resp.status);

    // Compute the true top-5 by dot product independently.
    var truth: [200]struct { id: u64, s: f64 } = undefined;
    for (0..n) |i| {
        var acc: f64 = 0;
        for (0..8) |d| acc += @as(f64, q[d]) * @as(f64, stored[i * 8 + d]);
        truth[i] = .{ .id = i, .s = acc };
    }
    std.mem.sort(@TypeOf(truth[0]), &truth, {}, struct {
        fn lt(_: void, a: @TypeOf(truth[0]), b: @TypeOf(truth[0])) bool {
            return a.s > b.s;
        }
    }.lt);

    var r = wire.Reader.init(resp.body);
    _ = try r.tag();
    var batch = try r.nested();
    var idx: usize = 0;
    while (!batch.atEnd()) {
        const bt = try batch.tag();
        if (bt.field != 1) {
            try batch.skip(bt.wire_type);
            continue;
        }
        var sp = try batch.nested();
        while (!sp.atEnd()) {
            const st = try sp.tag();
            if (st.field == 1) {
                var idr = try sp.nested();
                const pid = (try msg.PointId.decode(&idr)).num;
                try testing.expectEqual(truth[idx].id, pid);
            } else try sp.skip(st.wire_type);
        }
        idx += 1;
    }
    try testing.expectEqual(@as(usize, 5), idx);

    // The same query through the graph. Its answer is allowed to differ (it
    // is approximate, and that is the point of having an exact path to
    // measure it against), but the exact answer above was the truth *without*
    // consulting it, and the graph's answer must at least be a plausible one:
    // five results, led by the query's own vector.
    {
        var qs: [1][]const f32 = .{q};
        const aresp = try c.call("/qdrant.Points/QueryBatch", try buildQueryBatch(&req_buf, "ex", &qs, 5), &out);
        try testing.expectEqual(grpc.Status.ok, aresp.status);
        var aids: [5]u64 = undefined;
        var ascores: [5]f32 = undefined;
        try testing.expectEqual(@as(usize, 5), try readFirstBatch(aresp.body, &aids, &ascores));
        try testing.expectEqual(truth[0].id, aids[0]);
    }
}

test "e2e: §6.7 quantization is configured on the collection and used for search" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;

    // CreateCollection with a scalar quantization_config (field 8 on
    // CreateCollection, `ScalarQuantization` = field 1 of the oneof).
    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "q");
        {
            const vc = try w.beginNested(10, 2); // vectors_config (VERIFIED)
            const p = try w.beginNested(1, 2);
            try w.writeVarintField(1, 16); // size
            try w.writeVarintField(2, 3); // Dot
            {
                // Nested `hnsw_config { full_scan_threshold = 0 }`: 300
                // points would otherwise be plain-scanned, see
                // `buildCreateCollectionDt`.
                const hc = try w.beginNested(3, 2);
                try w.writeVarintFieldAlways(3, 0);
                try w.endNested(hc);
            }
            {
                const qc = try w.beginNested(4, 2); // quantization_config
                const sq = try w.beginNested(1, 2); // scalar
                try w.writeVarintField(1, 1); // QuantizationType::Int8
                try w.endNested(sq);
                try w.endNested(qc);
            }
            try w.endNested(p);
            try w.endNested(vc);
        }
        const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }

    // Upload, then wait for Green so the codes exist.
    var prng = std.Random.DefaultPrng.init(0x9911);
    const rnd = prng.random();
    const n = 300;
    const stored = try testing.allocator.alloc(f32, n * 16);
    defer testing.allocator.free(stored);
    for (stored) |*x| x.* = rnd.floatNorm(f32);

    {
        var ids: [300]u64 = undefined;
        var vecs: [300][]const f32 = undefined;
        for (0..n) |i| {
            ids[i] = i;
            vecs[i] = stored[i * 16 ..][0..16];
        }
        const resp = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "q", true, &ids, &vecs), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }

    var greens: usize = 0;
    var polls: usize = 0;
    while (greens < 3 and polls < 3000) : (polls += 1) {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "q");
        const resp = try c.call("/qdrant.Collections/Get", w.written(), &out);
        var r = wire.Reader.init(resp.body);
        _ = try r.tag();
        var info = try r.nested();
        var status: u64 = 0;
        while (!info.atEnd()) {
            const t = try info.tag();
            if (t.field == 1) status = try info.varint() else try info.skip(t.wire_type);
        }
        if (status == 1) greens += 1 else greens = 0;
    }
    try testing.expectEqual(@as(usize, 3), greens);

    // "Configured on the collection": the request's scalar config reached
    // the collection, and Green means the codes were actually built, so the
    // search below is the two-stage path (`coll.quant != null and !exact`
    // in `handlers.queryBatch`) and not the fp32 one.
    {
        const info = try getInfo(&c, "q", &req_buf, &out);
        try testing.expectEqual(@as(u64, 1), info.quant_kind);
        const coll = h.engine.find("q").?;
        try testing.expectEqual(@import("../quant/quant.zig").Mode.scalar, coll.quant_mode);
        try testing.expect(coll.quant.load(.acquire) != null);
    }

    // Query with oversampling + rescore, the configuration §6.7 recommends.
    {
        const q = stored[0..16];
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "q");
        const qp = try w.beginNested(2, 3);
        try w.writeStringField(1, "q");
        {
            const query = try w.beginNested(3, 3);
            const nearest = try w.beginNested(1, 3);
            const dv = try w.beginNested(2, 3);
            try w.writePackedFloats(1, q);
            try w.endNested(dv);
            try w.endNested(nearest);
            try w.endNested(query);
        }
        {
            const params = try w.beginNested(6, 2);
            try w.writeVarintField(1, 128); // hnsw_ef
            const qsp = try w.beginNested(3, 2);
            try w.writeBoolField(2, true); // rescore
            try w.writeDoubleField(3, 8.0); // oversampling
            try w.endNested(qsp);
            try w.endNested(params);
        }
        try w.writeVarintFieldAlways(8, 5);
        try w.endNested(qp);

        const resp = try c.call("/qdrant.Points/QueryBatch", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);

        var r = wire.Reader.init(resp.body);
        _ = try r.tag();
        var batch = try r.nested();
        var first: u64 = std.math.maxInt(u64);
        var got: usize = 0;
        while (!batch.atEnd()) {
            const bt = try batch.tag();
            if (bt.field != 1) {
                try batch.skip(bt.wire_type);
                continue;
            }
            var sp = try batch.nested();
            while (!sp.atEnd()) {
                const st = try sp.tag();
                if (st.field == 1) {
                    var idr = try sp.nested();
                    const pid = (try msg.PointId.decode(&idr)).num;
                    if (got == 0) first = pid;
                } else try sp.skip(st.wire_type);
            }
            got += 1;
        }
        try testing.expectEqual(@as(usize, 5), got);
        // SQ8 with rescore must still self-retrieve.
        try testing.expectEqual(@as(u64, 0), first);
    }
}

test "e2e: §6.7 turbo quantization is refused by name" {
    // "return a clear error; document the exclusion so nobody accidentally
    // compares against them". There is no `turbo` field in the proto, so the
    // parser-level rejection is exercised directly here.
    const quant = @import("../quant/quant.zig");
    switch (quant.Mode.parse("turbo-x8")) {
        .rejected => |why| try testing.expect(std.mem.indexOf(u8, why, "turbo") != null),
        .mode => return error.ShouldReject,
    }
}

test "e2e: a huge hnsw_ef is clamped, not fatal" {
    // `hnsw_ef` is client-controlled and feeds a heap whose storage is
    // preallocated at `Workspace.max_ef` (§6.3 forbids allocating on the query
    // path). Before it was clamped, a large value indexed past that storage:
    // an assertion failure in Debug and out-of-bounds writes in ReleaseFast,
    // from a single well-formed request.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;
    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "ef", 8, 3), &out);

    var prng = std.Random.DefaultPrng.init(0x3f3f);
    const rnd = prng.random();
    const n = 64;
    var stored: [64 * 8]f32 = undefined;
    for (&stored) |*x| x.* = rnd.floatNorm(f32);
    var ids: [n]u64 = undefined;
    var vecs: [n][]const f32 = undefined;
    for (0..n) |i| {
        ids[i] = i;
        vecs[i] = stored[i * 8 ..][0..8];
    }
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "ef", true, &ids, &vecs), &out);

    for ([_]u64{ 1, 128, 100_000, 4_000_000_000 }) |ef| {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "ef");
        const qp = try w.beginNested(2, 3);
        try w.writeStringField(1, "ef");
        {
            const query = try w.beginNested(3, 3);
            const nearest = try w.beginNested(1, 3);
            const dv = try w.beginNested(2, 3);
            try w.writePackedFloats(1, stored[0..8]);
            try w.endNested(dv);
            try w.endNested(nearest);
            try w.endNested(query);
        }
        {
            const params = try w.beginNested(6, 3);
            try w.writeVarintFieldAlways(1, ef);
            try w.endNested(params);
        }
        try w.writeVarintFieldAlways(8, 5);
        try w.endNested(qp);

        const resp = try c.call("/qdrant.Points/QueryBatch", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        // And the answer is still right, not merely non-fatal.
        var r = wire.Reader.init(resp.body);
        _ = try r.tag();
        var batch = try r.nested();
        var got: usize = 0;
        while (!batch.atEnd()) {
            const bt = try batch.tag();
            if (bt.field != 1) {
                try batch.skip(bt.wire_type);
                continue;
            }
            var sp = try batch.nested();
            while (!sp.atEnd()) {
                const st = try sp.tag();
                try sp.skip(st.wire_type);
            }
            got += 1;
        }
        try testing.expectEqual(@as(usize, 5), got);
    }
}

test "e2e: a collection larger than --max-dim is refused at create, not on upsert" {
    // §6.3 sizes the per-worker scratch once at startup. Without a bound at
    // creation, a well-formed CreateCollection with a large `size` produces a
    // collection whose very first upsert writes past that scratch, a
    // client-triggerable out-of-bounds, reported (if at all) from a place far
    // from its cause.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    // The harness builds workspaces at max_dim = 1536.
    const resp = try c.call(
        "/qdrant.Collections/Create",
        try buildCreateCollection(&req_buf, "toobig", 100_000, 3),
        &out,
    );
    try testing.expectEqual(grpc.Status.invalid_argument, resp.status);
    try testing.expect(std.mem.indexOf(u8, resp.message, "max-dim") != null);

    // And a dimension inside the bound still works.
    const ok_resp = try c.call(
        "/qdrant.Collections/Create",
        try buildCreateCollection(&req_buf, "fine", 1536, 3),
        &out,
    );
    try testing.expectEqual(grpc.Status.ok, ok_resp.status);
}

test "e2e: §12 Scroll pages the collection in id order" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [8192]u8 = undefined;
    var out: [8192]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "s", 2, 3), &out);

    // Uploaded out of id order, so a handler walking arrival order fails here.
    const ids = [_]u64{ 40, 10, 30, 20 };
    var vecs: [4][]const f32 = undefined;
    const v0 = [_]f32{ 1, 0 };
    for (&vecs) |*v| v.* = &v0;
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "s", true, &ids, &vecs), &out);

    // Page 1: limit 2, no cursor.
    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "s");
    try w.writeVarintField(4, 2); // limit, field 4, VERIFIED against 1.19
    const p1 = try c.call("/qdrant.Points/Scroll", w.written(), &out);
    try testing.expectEqual(grpc.Status.ok, p1.status);

    const page1 = try decodeScroll(p1.body);
    try testing.expectEqual(@as(usize, 2), page1.n);
    try testing.expectEqual(@as(u64, 10), page1.ids[0]);
    try testing.expectEqual(@as(u64, 20), page1.ids[1]);
    try testing.expectEqual(@as(u64, 30), page1.next.?);

    // Page 2: resume from the cursor. Nothing repeated, nothing skipped.
    var w2 = wire.Writer.init(&req_buf);
    try w2.writeStringField(1, "s");
    {
        const off = try w2.beginNested(3, 2); // offset, field 3, VERIFIED
        try w2.writeVarintFieldAlways(1, page1.next.?);
        try w2.endNested(off);
    }
    try w2.writeVarintField(4, 2);
    const p2 = try c.call("/qdrant.Points/Scroll", w2.written(), &out);
    const page2 = try decodeScroll(p2.body);
    try testing.expectEqual(@as(usize, 2), page2.n);
    try testing.expectEqual(@as(u64, 30), page2.ids[0]);
    try testing.expectEqual(@as(u64, 40), page2.ids[1]);
    // Collection exhausted: no cursor, which is how a client knows to stop.
    try testing.expectEqual(@as(?u64, null), page2.next);

    // `time` must land on field 3. The worker appends it generically and every
    // other response in §12 puts it at field 2, appending 2 here writes a
    // fixed64 into the middle of the repeated `result` list, so a conformant
    // decoder reads a garbage length for the next point. The page above decodes
    // only because the corruption lands after the last element we read.
    try testing.expectEqual(@as(?u32, 3), page1.time_field);
    try testing.expectEqual(@as(?u32, 3), page2.time_field);
}

test "e2e: Scroll refuses what it cannot answer rather than answering emptily" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [4096]u8 = undefined;
    var out: [4096]u8 = undefined;
    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "s", 2, 3), &out);

    // `with_vectors: true`. Returning points with the vectors field simply
    // absent would read as "these points have no vectors", a different and
    // wrong answer, and exactly the silent degradation §1 forbids.
    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "s");
    const wv = try w.beginNested(7, 2);
    try w.writeBoolField(1, true);
    try w.endNested(wv);
    const resp = try c.call("/qdrant.Points/Scroll", w.written(), &out);
    try testing.expectEqual(grpc.Status.unimplemented, resp.status);
    try testing.expectEqualStrings("vector retrieval in results (phase 2)", resp.message);
}

const ScrollPage = struct {
    ids: [64]u64 = undefined,
    n: usize = 0,
    next: ?u64 = null,
    /// Which field the `time` double actually arrived in.
    time_field: ?u32 = null,
};

fn decodeScroll(body: []const u8) !ScrollPage {
    var p = ScrollPage{};
    var r = wire.Reader.init(body);
    while (!r.atEnd()) {
        const t = try r.tag();
        switch (t.field) {
            1 => {
                var sub = try r.nested();
                p.next = (try msg.PointId.decode(&sub)).num;
            },
            2 => {
                var sub = try r.nested();
                while (!sub.atEnd()) {
                    const it = try sub.tag();
                    if (it.field == 1) {
                        var idr = try sub.nested();
                        p.ids[p.n] = (try msg.PointId.decode(&idr)).num;
                        p.n += 1;
                    } else try sub.skip(it.wire_type);
                }
            },
            else => {
                if (t.wire_type == .fixed64) p.time_field = t.field;
                try r.skip(t.wire_type);
            },
        }
    }
    return p;
}

test "e2e: an oversized request fails its stream and leaves the connection usable" {
    const small_limit = 64 * 1024;
    var h = try Harness.startWith(testing.allocator, small_limit);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [4096]u8 = undefined;
    var out: [4096]u8 = undefined;
    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "big", 4, 3), &out);

    // A body larger than the per-stream request buffer. This used to GOAWAY the
    // connection, so the client saw a transport error and every other in-flight
    // stream died with it.
    const huge = try testing.allocator.alloc(u8, small_limit + 4096);
    defer testing.allocator.free(huge);
    @memset(huge, 0);
    const resp = try c.call("/qdrant.Points/Upsert", huge, &out);
    try testing.expectEqual(grpc.Status.resource_exhausted, resp.status);
    // The message states the limit, so a client can pick a batch size instead
    // of bisecting one.
    try testing.expect(std.mem.indexOf(u8, resp.message, "KiB") != null);

    // The connection is still alive: the next request on it succeeds.
    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "big");
    const after = try c.call("/qdrant.Collections/CollectionExists", w.written(), &out);
    try testing.expectEqual(grpc.Status.ok, after.status);
}

test "e2e: connections beyond the slot count are refused and counted, not silently dropped" {
    // W4 died for a whole benchmark run on this: `-c 8` against 8 slots, where
    // the client's own version-handshake channel needed a ninth. The fd is
    // closed before the HTTP/2 preface, so there is no stream to carry a status
    // and the client can only report `transport error` with no detail. The
    // counter is what makes that diagnosable.
    var h = try Harness.startWith(testing.allocator, Harness.request_buffer);
    defer h.stop();

    const slots = h.srv.config.connections_per_io * h.srv.config.io_threads;

    var open: std.ArrayList(Client) = .empty;
    defer {
        for (open.items) |*c| c.close();
        open.deinit(testing.allocator);
    }

    // Fill every slot. Each must work.
    var out: [512]u8 = undefined;
    for (0..slots) |_| {
        var c = try Client.connect(h.port);
        const resp = try c.call("/qdrant.Qdrant/HealthCheck", &.{}, &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try open.append(testing.allocator, c);
    }
    try testing.expectEqual(@as(u64, 0), h.srv.totalStats().connections_refused);

    // One more than capacity: the connect may succeed at the TCP level, the
    // kernel completes the handshake from the backlog, but the request cannot,
    // and the server must record why.
    if (Client.connect(h.port)) |*extra| {
        var e = extra.*;
        defer e.close();
        _ = e.call("/qdrant.Qdrant/HealthCheck", &.{}, &out) catch {};
    } else |_| {}

    try testing.expect(h.srv.totalStats().connections_refused >= 1);
}

test "e2e: a uint8 collection stores bytes and searches them" {
    // The wire path for §5.4's storage datatypes: a client sets
    // `VectorParams.datatype` and the server stores one byte per component
    // rather than four, with Qdrant's saturating conversion.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    // Dot (3) with Uint8 (2).
    const resp_c = try c.call(
        "/qdrant.Collections/Create",
        try buildCreateCollectionDt(&req_buf, "bytes", 4, 3, 2),
        &out,
    );
    try testing.expectEqual(grpc.Status.ok, resp_c.status);

    const coll = h.engine.find("bytes").?;
    try testing.expectEqual(dist.Datatype.uint8, coll.config.datatype);
    // One byte per component, still padded to the cache line.
    try testing.expectEqual(@as(usize, 64), coll.space.stride);

    const vecs = [_][4]f32{
        .{ 10, 0, 0, 0 },
        .{ 200, 1, 0, 0 },
        // Out of range on both sides. Qdrant clamps rather than refusing, so
        // this asserts the clamp: §8.5's T1 compares values with the real
        // server, and being stricter is still a difference.
        .{ 300, -5, 2.9, 0 },
    };
    var slices: [3][]const f32 = undefined;
    for (&slices, 0..) |*sl, i| sl.* = &vecs[i];
    const ids = [_]u64{ 1, 2, 3 };
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "bytes", true, &ids, &slices), &out);

    try testing.expectEqualSlices(u8, &[_]u8{ 255, 0, 2, 0 }, coll.space.rowU8(2));

    const q = [_]f32{ 255, 0, 0, 0 };
    var qs: [1][]const f32 = .{&q};
    const resp = try c.call("/qdrant.Points/QueryBatch", try buildQueryBatch(&req_buf, "bytes", &qs, 2), &out);
    try testing.expectEqual(grpc.Status.ok, resp.status);

    var r = wire.Reader.init(resp.body);
    var top_id: u64 = 0;
    var top_score: f32 = 0;
    var seen: usize = 0;
    while (!r.atEnd()) {
        const t = try r.tag();
        if (t.field != 1) {
            try r.skip(t.wire_type);
            continue;
        }
        var batch = try r.nested();
        while (!batch.atEnd()) {
            const bt = try batch.tag();
            if (bt.field != 1) {
                try batch.skip(bt.wire_type);
                continue;
            }
            var sp = try batch.nested();
            var id: u64 = 0;
            var score: f32 = 0;
            while (!sp.atEnd()) {
                const st = try sp.tag();
                switch (st.field) {
                    1 => {
                        var idr = try sp.nested();
                        while (!idr.atEnd()) {
                            const it = try idr.tag();
                            if (it.field == 1) id = try idr.varint() else try idr.skip(it.wire_type);
                        }
                    },
                    3 => score = try sp.float(),
                    else => try sp.skip(st.wire_type),
                }
            }
            if (seen == 0) {
                top_id = id;
                top_score = score;
            }
            seen += 1;
        }
    }

    // The clamped third point is the nearest under dot, and the score is the
    // integer product 255*255 exactly: u8 storage computes in i32, so this is
    // an equality rather than a tolerance.
    try testing.expect(seen >= 1);
    try testing.expectEqual(@as(u64, 3), top_id);
    try testing.expectEqual(@as(f32, 255 * 255), top_score);
}

// =========================================================================
// §5.5 / Qdrant `VectorParams.memory`: where the vectors actually live
// =========================================================================

/// A scratch directory for the placement tests, removed on `deinit`.
const TmpDir = struct {
    path: [64]u8 = undefined,
    len: usize = 0,

    fn make(tag: []const u8) !TmpDir {
        var d = TmpDir{};
        const p = try std.fmt.bufPrint(&d.path, "/tmp/strawmann-place-{s}\x00", .{tag});
        d.len = p.len - 1;
        // The linux syscalls directly, as `storage.zig`'s own tests do: this
        // has to work the same way the code under test opens its files.
        _ = linux.mkdir(@ptrCast(d.path[0..].ptr), 0o755); // EEXIST is fine
        return d;
    }

    fn slice(self: *const TmpDir) []const u8 {
        return self.path[0..self.len];
    }

    /// A NUL-terminated path inside this directory.
    fn file(self: *const TmpDir, buf: []u8, name: []const u8) [:0]const u8 {
        const p = std.fmt.bufPrintZ(buf, "{s}/{s}", .{ self.slice(), name }) catch unreachable;
        return p;
    }

    /// The size of a file in this directory, or null if it does not exist.
    fn sizeOf(self: *const TmpDir, name: []const u8) ?u64 {
        var buf: [256]u8 = undefined;
        const path = self.file(&buf, name);
        var st: linux.Statx = undefined;
        const rc = linux.statx(linux.AT.FDCWD, path.ptr, 0, .{ .SIZE = true }, &st);
        if (linux.errno(rc) != .SUCCESS) return null;
        return st.size;
    }

    fn deinit(self: *TmpDir) void {
        // Only the files these tests create, then the directory itself.
        for ([_][]const u8{
            "warm.vectors.bin",   "chilly.vectors.bin",  "both.vectors.bin",
            "legacy.vectors.bin", "notdisk.vectors.bin", "g.vectors.bin",
        }) |n| {
            var buf: [256]u8 = undefined;
            _ = linux.unlink(self.file(&buf, n).ptr);
        }
        var z: [64]u8 = undefined;
        const p = std.fmt.bufPrintZ(&z, "{s}", .{self.slice()}) catch return;
        _ = linux.rmdir(p.ptr);
    }
};

/// Build a CreateCollection body with an explicit placement.
///
/// `memory` is `VectorParams.memory` (field 8, 0 = unset) and `on_disk` is the
/// deprecated field 5. Both are writable so the precedence between them can be
/// tested rather than assumed.
fn createWithPlacement(
    w: *wire.Writer,
    name: []const u8,
    dim: u64,
    memory: ?i32,
    on_disk: ?bool,
) !void {
    try w.writeStringField(1, name);
    const vc = try w.beginNested(10, 2);
    const p = try w.beginNested(1, 2);
    try w.writeVarintField(1, dim);
    try w.writeVarintField(2, 3); // Dot
    // Written as a raw tag rather than through `writeBoolField`, which elides
    // `false` as a proto3 default. `on_disk` is `optional bool` — explicit
    // presence — so a real client puts `5: 0` on the wire and means "Cached",
    // not "unstated". Eliding it here would make the test agree with the
    // server for the wrong reason.
    if (on_disk) |d| {
        try w.tag(5, .varint);
        try w.varint(if (d) 1 else 0);
    }
    if (memory) |m| try w.writeVarintField(8, @intCast(m));
    try w.endNested(p);
    try w.endNested(vc);
}

test "e2e: §5.5 a cached collection maps its arena from a file and still searches" {
    var tmp = try TmpDir.make("cached");
    defer tmp.deinit();

    var h = try Harness.startWithDataDir(testing.allocator, tmp.slice());
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;

    {
        var w = wire.Writer.init(&req_buf);
        try createWithPlacement(&w, "warm", 8, 2, null); // Memory::Cached
        const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }

    // The placement is what was asked for, and the arena is a mapping rather
    // than heap memory. Checking `region` and not only the enum: the enum is
    // what we were told, the region is what happened.
    const coll = h.engine.find("warm").?;
    try testing.expectEqual(core.storage.Placement.cached, coll.space.placement);
    try testing.expect(coll.space.region != null);
    try testing.expectEqual(core.storage.Placement.cached, coll.space.region.?.placement);

    // The file exists, and is preallocated to the full capacity rather than
    // grown as points arrive (§6.4).
    try testing.expectEqual(
        @as(?u64, core.storage.header_size + coll.space.stride * coll.config.capacity),
        tmp.sizeOf("warm.vectors.bin"),
    );

    // And it answers queries: a mapped arena is not a separate code path for
    // search, which is the entire point of doing it this way.
    var ids: [4]u64 = undefined;
    var vecs: [4][]const f32 = undefined;
    var data: [4][8]f32 = undefined;
    for (0..4) |i| {
        for (&data[i], 0..) |*x, j| x.* = if (j == @as(usize, i)) 1.0 else 0.0;
        ids[i] = i;
        vecs[i] = &data[i];
    }
    {
        const body = try buildUpsert(&req_buf, "warm", true, ids[0..], vecs[0..]);
        const resp = try c.call("/qdrant.Points/Upsert", body, &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }
    try testing.expectEqual(@as(usize, 4), coll.count());

    // The bytes went through the mapping and are visible in the file's rows:
    // row 2 is the third basis vector, so its third component is 1.
    const row = std.mem.bytesAsSlice(f32, coll.space.rowBytes(2));
    try testing.expectEqual(@as(f32, 1.0), row[2]);
}

test "e2e: a cold collection is not prefaulted, which is the whole difference" {
    var tmp = try TmpDir.make("cold");
    defer tmp.deinit();

    var h = try Harness.startWithDataDir(testing.allocator, tmp.slice());
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;

    {
        var w = wire.Writer.init(&req_buf);
        try createWithPlacement(&w, "chilly", 8, 1, null); // Memory::Cold
        const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }
    const coll = h.engine.find("chilly").?;
    try testing.expectEqual(core.storage.Placement.cold, coll.space.placement);
    try testing.expect(coll.space.region != null);
}

test "e2e: memory overrides the deprecated on_disk, as Memory::resolve does" {
    var tmp = try TmpDir.make("resolve");
    defer tmp.deinit();

    var h = try Harness.startWithDataDir(testing.allocator, tmp.slice());
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;

    // on_disk = true would be Cold on its own; memory = Cached must win.
    // Qdrant's `Memory::resolve` is `memory.or(legacy)`, documented as "the
    // explicit parameter always wins".
    {
        var w = wire.Writer.init(&req_buf);
        try createWithPlacement(&w, "both", 8, 2, true);
        const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }
    try testing.expectEqual(
        core.storage.Placement.cached,
        h.engine.find("both").?.space.placement,
    );

    // on_disk alone still decides, so a client too old to send `memory` is
    // honoured rather than ignored.
    {
        var w = wire.Writer.init(&req_buf);
        try createWithPlacement(&w, "legacy", 8, null, true);
        const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }
    try testing.expectEqual(
        core.storage.Placement.cold,
        h.engine.find("legacy").?.space.placement,
    );

    // And `on_disk: false` is Cached, NOT pinned. This is the misreading that
    // would let strawmANN claim it matched Qdrant's in-RAM configuration while
    // running a placement Qdrant does not offer for dense vectors.
    {
        var w = wire.Writer.init(&req_buf);
        try createWithPlacement(&w, "notdisk", 8, null, false);
        const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }
    try testing.expectEqual(
        core.storage.Placement.cached,
        h.engine.find("notdisk").?.space.placement,
    );
}

test "e2e: an unstated placement is pinned, and stays heap-allocated" {
    var tmp = try TmpDir.make("default");
    defer tmp.deinit();

    var h = try Harness.startWithDataDir(testing.allocator, tmp.slice());
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;

    {
        var w = wire.Writer.init(&req_buf);
        try createWithPlacement(&w, "plain", 8, null, null);
        const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }
    const coll = h.engine.find("plain").?;
    try testing.expectEqual(core.storage.Placement.pinned, coll.space.placement);
    try testing.expect(coll.space.region == null);

    // No file was created for it. Qdrant would have made one: its default for
    // dense vectors is Cached. That divergence is the §2 semantic gap, and it
    // is asserted here so it cannot drift silently.
    try testing.expectEqual(@as(?u64, null), tmp.sizeOf("plain.vectors.bin"));
}

test "e2e: a mapped placement without --data-dir is refused, not downgraded" {
    var h = try Harness.start(testing.allocator); // no data_dir
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;

    var w = wire.Writer.init(&req_buf);
    try createWithPlacement(&w, "nowhere", 8, 1, null);
    const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
    // Refused, and the message names the flag that fixes it. Serving this from
    // pinned memory would answer the request and measure something else.
    try testing.expectEqual(grpc.Status.failed_precondition, resp.status);
    try testing.expect(std.mem.indexOf(u8, resp.message, "--data-dir") != null);
    try testing.expect(h.engine.find("nowhere") == null);
}

test "e2e: §1 a graph placement is refused by name rather than ignored" {
    var tmp = try TmpDir.make("graph");
    defer tmp.deinit();

    var h = try Harness.startWithDataDir(testing.allocator, tmp.slice());
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;

    // hnsw_config { memory: Cold } inside VectorParams.
    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "g");
    {
        const vc = try w.beginNested(10, 2);
        const p = try w.beginNested(1, 2);
        try w.writeVarintField(1, 8);
        try w.writeVarintField(2, 3);
        {
            const hc = try w.beginNested(3, 2); // hnsw_config
            try w.writeVarintField(8, 1); // memory = Cold
            try w.endNested(hc);
        }
        try w.endNested(p);
        try w.endNested(vc);
    }
    const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
    try testing.expectEqual(grpc.Status.unimplemented, resp.status);
    try testing.expect(std.mem.indexOf(u8, resp.message, "hnsw_config") != null);
}

test "e2e: an unrecognised memory value is refused, not defaulted" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;

    var w = wire.Writer.init(&req_buf);
    try createWithPlacement(&w, "future", 8, 99, null);
    const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
    // Not silently pinned. A build that has never heard of placement 99 must
    // say so, or it reports a number for a configuration it did not run.
    try testing.expectEqual(grpc.Status.invalid_argument, resp.status);
    try testing.expect(h.engine.find("future") == null);
}

// =========================================================================
// Create-time config, semantics that drifted from Qdrant, and the poll loop
// =========================================================================

/// One `QueryPoints` with the knobs these tests turn. `buildQueryBatch` is
/// the minimal request; this is the same message with `params` and
/// `score_threshold` filled.
const QueryOpts = struct {
    limit: ?u64 = 10,
    /// Written even when null-equivalent: `writeVarintFieldAlways`, so an
    /// explicit 0 goes on the wire.
    limit_always: bool = false,
    hnsw_ef: ?u64 = null,
    exact: bool = false,
    score_threshold: ?f32 = null,
    /// `QueryPoints.offset`, field 9 (VERIFIED).
    offset: ?u64 = null,
};

fn buildQuery(buf: []u8, name: []const u8, q: []const f32, opts: QueryOpts) ![]const u8 {
    var w = wire.Writer.init(buf);
    try w.writeStringField(1, name);
    const qp = try w.beginNested(2, 3);
    try w.writeStringField(1, name);
    {
        const query = try w.beginNested(3, 3);
        const nearest = try w.beginNested(1, 3);
        const dv = try w.beginNested(2, 3);
        try w.writePackedFloats(1, q);
        try w.endNested(dv);
        try w.endNested(nearest);
        try w.endNested(query);
    }
    if (opts.hnsw_ef != null or opts.exact) {
        const params = try w.beginNested(6, 2);
        if (opts.hnsw_ef) |ef| try w.writeVarintField(1, ef);
        try w.writeBoolField(2, opts.exact);
        try w.endNested(params);
    }
    if (opts.score_threshold) |th| try w.writeFloatFieldAlways(7, th);
    if (opts.limit) |l| {
        if (opts.limit_always) try w.writeVarintFieldAlways(8, l) else try w.writeVarintField(8, l);
    }
    if (opts.offset) |o| try w.writeVarintFieldAlways(9, o);
    try w.endNested(qp);
    return w.written();
}

/// Ids and scores of the first `BatchResult`, in order. Returns the count.
fn readFirstBatch(body: []const u8, ids: []u64, scores: []f32) !usize {
    var r = wire.Reader.init(body);
    var n: usize = 0;
    while (!r.atEnd()) {
        const t = try r.tag();
        if (t.field != 1) {
            try r.skip(t.wire_type);
            continue;
        }
        var batch = try r.nested();
        while (!batch.atEnd()) {
            const bt = try batch.tag();
            if (bt.field != 1) {
                try batch.skip(bt.wire_type);
                continue;
            }
            var sp = try batch.nested();
            var id: u64 = 0;
            var score: f32 = 0;
            while (!sp.atEnd()) {
                const st = try sp.tag();
                switch (st.field) {
                    1 => {
                        var idr = try sp.nested();
                        id = (try msg.PointId.decode(&idr)).num;
                    },
                    3 => score = try sp.float(),
                    else => try sp.skip(st.wire_type),
                }
            }
            if (n < ids.len) {
                ids[n] = id;
                scores[n] = score;
            }
            n += 1;
        }
        return n;
    }
    return n;
}

/// The bits of `CollectionInfo` these tests read back.
const Info = struct {
    status: u64 = 0,
    points: u64 = 0,
    indexed: u64 = 0,
    saw_config: bool = false,
    size: u64 = 0,
    distance: u64 = 0,
    datatype: u64 = 0,
    hnsw_m: u64 = 0,
    hnsw_ef_construct: u64 = 0,
    /// Null when the field was absent: it is `optional uint64`, and a
    /// threshold of 0 must arrive as 0, not as nothing.
    hnsw_full_scan_threshold: ?u64 = null,
    /// 0 none, 1 scalar, 2 product, 3 binary: the `QuantizationConfig` oneof.
    quant_kind: u64 = 0,
    quantile: f32 = 0,
};

fn getInfo(c: *Client, name: []const u8, rb: []u8, ob: []u8) !Info {
    var w = wire.Writer.init(rb);
    try w.writeStringField(1, name);
    const resp = try c.call("/qdrant.Collections/Get", w.written(), ob);
    if (resp.status != .ok) return error.GetFailed;
    var info: Info = .{};
    var r = wire.Reader.init(resp.body);
    _ = try r.tag();
    var ci = try r.nested();
    while (!ci.atEnd()) {
        const t = try ci.tag();
        switch (t.field) {
            1 => info.status = try ci.varint(),
            7 => {
                info.saw_config = true;
                var cfg = try ci.nested();
                while (!cfg.atEnd()) {
                    const ct = try cfg.tag();
                    switch (ct.field) {
                        1 => { // CollectionParams
                            var params = try cfg.nested();
                            while (!params.atEnd()) {
                                const pt = try params.tag();
                                if (pt.field != 5) {
                                    try params.skip(pt.wire_type);
                                    continue;
                                }
                                var vc = try params.nested(); // VectorsConfig
                                while (!vc.atEnd()) {
                                    const vt = try vc.tag();
                                    if (vt.field != 1) {
                                        try vc.skip(vt.wire_type);
                                        continue;
                                    }
                                    var vp = try vc.nested(); // VectorParams
                                    while (!vp.atEnd()) {
                                        const ft = try vp.tag();
                                        switch (ft.field) {
                                            1 => info.size = try vp.varint(),
                                            2 => info.distance = try vp.varint(),
                                            6 => info.datatype = try vp.varint(),
                                            else => try vp.skip(ft.wire_type),
                                        }
                                    }
                                }
                            }
                        },
                        2 => { // HnswConfigDiff
                            var hc = try cfg.nested();
                            while (!hc.atEnd()) {
                                const ht = try hc.tag();
                                switch (ht.field) {
                                    1 => info.hnsw_m = try hc.varint(),
                                    2 => info.hnsw_ef_construct = try hc.varint(),
                                    3 => info.hnsw_full_scan_threshold = try hc.varint(),
                                    else => try hc.skip(ht.wire_type),
                                }
                            }
                        },
                        5 => { // QuantizationConfig
                            var qc = try cfg.nested();
                            while (!qc.atEnd()) {
                                const qt = try qc.tag();
                                info.quant_kind = qt.field;
                                var inner = try qc.nested();
                                while (!inner.atEnd()) {
                                    const it = try inner.tag();
                                    if (qt.field == 1 and it.field == 2) {
                                        info.quantile = try inner.float();
                                    } else try inner.skip(it.wire_type);
                                }
                            }
                        },
                        else => try cfg.skip(ct.wire_type),
                    }
                }
            },
            9 => info.points = try ci.varint(),
            10 => info.indexed = try ci.varint(),
            else => try ci.skip(t.wire_type),
        }
    }
    return info;
}

/// Poll `Collections/Get` the way bfb's `wait_index` does, until Green three
/// times in a row. Returns the number of polls it took.
fn waitGreen(c: *Client, name: []const u8, rb: []u8, ob: []u8) !usize {
    var greens: usize = 0;
    var polls: usize = 0;
    while (greens < 3 and polls < 3000) : (polls += 1) {
        const info = try getInfo(c, name, rb, ob);
        if (info.status == 1) greens += 1 else greens = 0;
    }
    if (greens < 3) return error.NeverGreen;
    return polls;
}

fn uploadRandom(c: *Client, name: []const u8, dim: usize, stored: []f32, rb: []u8, ob: []u8) !void {
    const n = stored.len / dim;
    var batch_ids: [50]u64 = undefined;
    var batch_vecs: [50][]const f32 = undefined;
    var i: usize = 0;
    while (i < n) : (i += 50) {
        const take = @min(50, n - i);
        for (0..take) |j| {
            batch_ids[j] = i + j;
            batch_vecs[j] = stored[(i + j) * dim ..][0..dim];
        }
        const resp = try c.call("/qdrant.Points/Upsert", try buildUpsert(rb, name, true, batch_ids[0..take], batch_vecs[0..take]), ob);
        if (resp.status != .ok) return error.UpsertFailed;
    }
}

test "e2e: top-level hnsw_config and quantization_config take effect, and read back" {
    // bfb puts both at the top level of `CreateCollection` (`from_args.rs`),
    // and always sends `optimizers_config` and `hnsw_config.on_disk = false`
    // with them. Before this was parsed, `--quantization scalar --hnsw-m 32`
    // built an fp32 collection at M=16 and said nothing.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;

    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "top");
        {
            // hnsw_config = 4: m=32, ef_construct=64, on_disk=false.
            const hc = try w.beginNested(4, 2);
            try w.writeVarintField(1, 32);
            try w.writeVarintField(2, 64);
            try w.writeVarintFieldAlways(5, 0); // on_disk = false, as bfb sends
            try w.endNested(hc);
        }
        {
            // optimizers_config = 6, as bfb sends with `--segments 1`.
            const oc = try w.beginNested(6, 2);
            try w.writeVarintField(3, 1); // default_segment_number
            try w.writeVarintField(4, 200_000); // max_segment_size
            try w.endNested(oc);
        }
        {
            const vc = try w.beginNested(10, 2);
            const p = try w.beginNested(1, 2);
            try w.writeVarintField(1, 16); // size
            try w.writeVarintField(2, 3); // Dot
            try w.endNested(p);
            try w.endNested(vc);
        }
        {
            // quantization_config = 14: scalar Int8 at quantile 0.9, always_ram.
            const qc = try w.beginNested(14, 2);
            const sq = try w.beginNested(1, 2);
            try w.writeVarintField(1, 1);
            try w.writeFloatFieldAlways(2, 0.9);
            try w.writeBoolField(3, true);
            try w.endNested(sq);
            try w.endNested(qc);
        }
        const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }

    // The engine's view.
    const coll = h.engine.find("top").?;
    try testing.expectEqual(@as(usize, 32), coll.config.hnsw_m);
    try testing.expectEqual(@as(usize, 64), coll.config.hnsw_ef_construct);
    try testing.expect(coll.quant_mode == .scalar);
    try testing.expectApproxEqAbs(@as(f32, 0.9), coll.quant_quantile, 1e-6);

    // The client's view, via `CollectionInfo.config`.
    const info = try getInfo(&c, "top", &req_buf, &out);
    try testing.expect(info.saw_config);
    try testing.expectEqual(@as(u64, 16), info.size);
    try testing.expectEqual(@as(u64, 3), info.distance);
    try testing.expectEqual(@as(u64, 1), info.datatype); // Float32
    try testing.expectEqual(@as(u64, 32), info.hnsw_m);
    try testing.expectEqual(@as(u64, 64), info.hnsw_ef_construct);
    try testing.expectEqual(@as(u64, 1), info.quant_kind); // scalar
    try testing.expectApproxEqAbs(@as(f32, 0.9), info.quantile, 1e-6);

    // And the collection is *actually* quantized once built: the graph is at
    // M=32 and a store exists.
    var prng = std.Random.DefaultPrng.init(0x7071);
    const rnd = prng.random();
    const stored = try testing.allocator.alloc(f32, 300 * 16);
    defer testing.allocator.free(stored);
    for (stored) |*x| x.* = rnd.floatNorm(f32);
    try uploadRandom(&c, "top", 16, stored, &req_buf, &out);
    _ = try waitGreen(&c, "top", &req_buf, &out);
    try testing.expect(coll.quant.load(.acquire) != null);
    try testing.expect(coll.quant.load(.acquire).?.* == .scalar);
    try testing.expectEqual(@as(usize, 32), coll.graph.?.params.m);
}

test "e2e: a nested VectorParams.hnsw_config overrides the top-level one" {
    // Qdrant's precedence: per-vector over collection.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "prec");
    {
        const hc = try w.beginNested(4, 2);
        try w.writeVarintField(1, 32);
        try w.writeVarintField(2, 64);
        try w.endNested(hc);
    }
    {
        const vc = try w.beginNested(10, 2);
        const p = try w.beginNested(1, 2);
        try w.writeVarintField(1, 8);
        try w.writeVarintField(2, 3);
        {
            const nh = try w.beginNested(3, 2);
            try w.writeVarintField(1, 8); // m only; ef_construct falls through
            try w.endNested(nh);
        }
        try w.endNested(p);
        try w.endNested(vc);
    }
    const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
    try testing.expectEqual(grpc.Status.ok, resp.status);
    const coll = h.engine.find("prec").?;
    try testing.expectEqual(@as(usize, 8), coll.config.hnsw_m);
    try testing.expectEqual(@as(usize, 64), coll.config.hnsw_ef_construct);
}

test "e2e: a top-level hnsw_config asking for an on-disk graph is refused, on_disk=false is not" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    for ([_]struct { on_disk: bool, memory: u64, want: grpc.Status }{
        .{ .on_disk = false, .memory = 0, .want = .ok },
        .{ .on_disk = true, .memory = 0, .want = .unimplemented },
        .{ .on_disk = false, .memory = 1, .want = .unimplemented },
    }, 0..) |case, i| {
        var name_buf: [8]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "od{d}", .{i});
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, name);
        {
            const hc = try w.beginNested(4, 2);
            try w.writeVarintFieldAlways(5, @intFromBool(case.on_disk));
            try w.writeVarintField(8, case.memory);
            try w.endNested(hc);
        }
        {
            const vc = try w.beginNested(10, 2);
            const p = try w.beginNested(1, 2);
            try w.writeVarintField(1, 8);
            try w.writeVarintField(2, 3);
            try w.endNested(p);
            try w.endNested(vc);
        }
        const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
        try testing.expectEqual(case.want, resp.status);
        if (case.want != .ok) try testing.expect(std.mem.indexOf(u8, resp.message, "hnsw_config") != null);
    }
}

test "e2e: binary encodings other than OneBit and a bad quantile are refused by name" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    const Case = struct { oneof: u32, field: u32, value: u64, nested_setting: bool = false, want: grpc.Status, name_in_msg: []const u8 };
    for ([_]Case{
        // binary.encoding = TwoBits
        .{ .oneof = 3, .field = 2, .value = 1, .want = .unimplemented, .name_in_msg = "encoding" },
        // binary.query_encoding { setting = Scalar8Bits }
        .{ .oneof = 3, .field = 3, .value = 3, .nested_setting = true, .want = .unimplemented, .name_in_msg = "query_encoding" },
        // binary.query_encoding { setting = Binary }: fine
        .{ .oneof = 3, .field = 3, .value = 1, .nested_setting = true, .want = .ok, .name_in_msg = "" },
        // binary.always_ram = true: accepted (§6.7)
        .{ .oneof = 3, .field = 1, .value = 1, .want = .ok, .name_in_msg = "" },
    }, 0..) |case, i| {
        var name_buf: [8]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "bq{d}", .{i});
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, name);
        {
            const vc = try w.beginNested(10, 2);
            const p = try w.beginNested(1, 2);
            try w.writeVarintField(1, 8);
            try w.writeVarintField(2, 3);
            try w.endNested(p);
            try w.endNested(vc);
        }
        {
            const qc = try w.beginNested(14, 2);
            const inner = try w.beginNested(case.oneof, 2);
            if (case.nested_setting) {
                const qe = try w.beginNested(case.field, 2);
                try w.writeVarintFieldAlways(4, case.value);
                try w.endNested(qe);
            } else {
                try w.writeVarintFieldAlways(case.field, case.value);
            }
            try w.endNested(inner);
            try w.endNested(qc);
        }
        const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
        try testing.expectEqual(case.want, resp.status);
        if (case.name_in_msg.len > 0) try testing.expect(std.mem.indexOf(u8, resp.message, case.name_in_msg) != null);
    }

    // scalar.quantile = 0.3: outside Qdrant's [0.5, 1.0].
    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "qq");
        {
            const vc = try w.beginNested(10, 2);
            const p = try w.beginNested(1, 2);
            try w.writeVarintField(1, 8);
            try w.writeVarintField(2, 3);
            try w.endNested(p);
            try w.endNested(vc);
        }
        {
            const qc = try w.beginNested(14, 2);
            const sq = try w.beginNested(1, 2);
            try w.writeVarintField(1, 1);
            try w.writeFloatFieldAlways(2, 0.3);
            try w.endNested(sq);
            try w.endNested(qc);
        }
        const resp = try c.call("/qdrant.Collections/Create", w.written(), &out);
        try testing.expectEqual(grpc.Status.invalid_argument, resp.status);
        try testing.expect(std.mem.indexOf(u8, resp.message, "quantile") != null);
    }
}

test "e2e: uint8 cosine scores are Qdrant's cosine_similarity_bytes, not zero" {
    // The query was normalised as for fp32 cosine and then truncated to
    // bytes, so every score was 0. Expected values by hand: dot / sqrt(Σq²·Σr²).
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    // Cosine (1), Uint8 (2).
    const cr = try c.call("/qdrant.Collections/Create", try buildCreateCollectionDt(&req_buf, "u8cos", 4, 1, 2), &out);
    try testing.expectEqual(grpc.Status.ok, cr.status);

    const vecs = [_][4]f32{
        .{ 3, 4, 0, 0 }, // parallel to the query: 1.0
        .{ 4, 3, 0, 0 }, // (24+24)/(10*5) = 0.96
        .{ 0, 0, 5, 12 }, // orthogonal: 0
    };
    var slices: [3][]const f32 = undefined;
    for (&slices, 0..) |*sl, i| sl.* = &vecs[i];
    const ids = [_]u64{ 1, 2, 3 };
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "u8cos", true, &ids, &slices), &out);

    const q = [_]f32{ 6, 8, 0, 0 };
    const resp = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "u8cos", &q, .{ .limit = 3 }), &out);
    try testing.expectEqual(grpc.Status.ok, resp.status);

    var got_ids: [3]u64 = undefined;
    var got_scores: [3]f32 = undefined;
    try testing.expectEqual(@as(usize, 3), try readFirstBatch(resp.body, &got_ids, &got_scores));
    try testing.expectEqual(@as(u64, 1), got_ids[0]);
    try testing.expectApproxEqAbs(@as(f32, 1.0), got_scores[0], 1e-6);
    try testing.expectEqual(@as(u64, 2), got_ids[1]);
    try testing.expectApproxEqAbs(@as(f32, 0.96), got_scores[1], 1e-6);
    try testing.expectEqual(@as(u64, 3), got_ids[2]);
    try testing.expectEqual(@as(f32, 0.0), got_scores[2]);
}

test "e2e: a small write after Green keeps the collection Green, and a query starts the build" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "tail", 8, 2), &out);
    var prng = std.Random.DefaultPrng.init(0x7a17);
    const rnd = prng.random();
    const n = 500;
    const stored = try testing.allocator.alloc(f32, n * 8);
    defer testing.allocator.free(stored);
    for (stored) |*x| x.* = rnd.floatNorm(f32);
    try uploadRandom(&c, "tail", 8, stored, &req_buf, &out);

    // `--skip-wait-index --search`: no Collections/Get at all. The first
    // query must kick the build off; before, nothing did, and the collection
    // brute-forced (and reported Yellow) for the whole run.
    const coll = h.engine.find("tail").?;
    try testing.expectEqual(core.collection.IndexState.absent, coll.index_state.load(.acquire));
    {
        const q = stored[0..8];
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "tail", q, .{ .limit = 5 }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }
    try testing.expect(coll.index_state.load(.acquire) != .absent);
    _ = try waitGreen(&c, "tail", &req_buf, &out);

    // A handful of new points, well under `rebuild_ratio`: still Green, with
    // an honest `indexed_vectors_count`. It used to go Yellow and stay there,
    // since a tail this size never triggers a rebuild.
    {
        var extra: [3][8]f32 = undefined;
        var sl: [3][]const f32 = undefined;
        for (&extra, 0..) |*e, i| {
            for (e) |*x| x.* = rnd.floatNorm(f32);
            sl[i] = e;
        }
        const ids = [_]u64{ n, n + 1, n + 2 };
        _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "tail", true, &ids, &sl), &out);
    }
    var greens: usize = 0;
    for (0..3) |_| {
        const info = try getInfo(&c, "tail", &req_buf, &out);
        if (info.status == 1) greens += 1;
        try testing.expectEqual(@as(u64, n + 3), info.points);
        try testing.expect(info.indexed <= n + 3);
    }
    try testing.expectEqual(@as(usize, 3), greens);

    // Cross the ratio: the poll path must be able to start a rebuild and
    // reach Green with everything indexed.
    {
        const more = @as(usize, @intFromFloat(@as(f64, n) * core.collection.rebuild_ratio)) + 5;
        const buf = try testing.allocator.alloc(f32, more * 8);
        defer testing.allocator.free(buf);
        for (buf) |*x| x.* = rnd.floatNorm(f32);
        var batch_ids: [50]u64 = undefined;
        var batch_vecs: [50][]const f32 = undefined;
        var i: usize = 0;
        while (i < more) : (i += 50) {
            const take = @min(50, more - i);
            for (0..take) |j| {
                batch_ids[j] = n + 3 + i + j;
                batch_vecs[j] = buf[(i + j) * 8 ..][0..8];
            }
            _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "tail", true, batch_ids[0..take], batch_vecs[0..take]), &out);
        }
        _ = try waitGreen(&c, "tail", &req_buf, &out);
        const info = try getInfo(&c, "tail", &req_buf, &out);
        try testing.expectEqual(info.points, info.indexed);
    }
}

test "e2e: score_threshold is strict, as Qdrant's check_threshold is" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    // Dot (3): scores against (1,0,0,0) are 3, 2, 1.
    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "th", 4, 3), &out);
    const vecs = [_][4]f32{ .{ 3, 0, 0, 0 }, .{ 2, 0, 0, 0 }, .{ 1, 0, 0, 0 } };
    var slices: [3][]const f32 = undefined;
    for (&slices, 0..) |*sl, i| sl.* = &vecs[i];
    const ids = [_]u64{ 1, 2, 3 };
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "th", true, &ids, &slices), &out);

    const q = [_]f32{ 1, 0, 0, 0 };
    const resp = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "th", &q, .{ .limit = 10, .score_threshold = 2.0 }), &out);
    try testing.expectEqual(grpc.Status.ok, resp.status);
    var got_ids: [3]u64 = undefined;
    var got_scores: [3]f32 = undefined;
    // `> 2.0`: only the score of 3 passes; the point scoring exactly 2 is out.
    try testing.expectEqual(@as(usize, 1), try readFirstBatch(resp.body, &got_ids, &got_scores));
    try testing.expectEqual(@as(u64, 1), got_ids[0]);

    // Euclid (2): distances from the origin are 3, 2, 1; `< 2.0` keeps one.
    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "the", 4, 2), &out);
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "the", true, &ids, &slices), &out);
    const zero = [_]f32{ 0, 0, 0, 0 };
    const resp2 = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "the", &zero, .{ .limit = 10, .score_threshold = 2.0 }), &out);
    try testing.expectEqual(grpc.Status.ok, resp2.status);
    try testing.expectEqual(@as(usize, 1), try readFirstBatch(resp2.body, &got_ids, &got_scores));
    try testing.expectEqual(@as(u64, 3), got_ids[0]);
}

test "e2e: limit 0 is INVALID_ARGUMENT, absent is 10" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "lim", 4, 3), &out);
    var vecs: [12][4]f32 = undefined;
    var slices: [12][]const f32 = undefined;
    var ids: [12]u64 = undefined;
    for (&vecs, 0..) |*v, i| {
        v.* = .{ @floatFromInt(i + 1), 0, 0, 0 };
        slices[i] = v;
        ids[i] = i;
    }
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "lim", true, &ids, &slices), &out);

    const q = [_]f32{ 1, 0, 0, 0 };
    var got_ids: [12]u64 = undefined;
    var got_scores: [12]f32 = undefined;
    {
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "lim", &q, .{ .limit = 0, .limit_always = true }), &out);
        try testing.expectEqual(grpc.Status.invalid_argument, resp.status);
        try testing.expect(std.mem.indexOf(u8, resp.message, "limit") != null);
    }
    {
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "lim", &q, .{ .limit = null }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expectEqual(@as(usize, 10), try readFirstBatch(resp.body, &got_ids, &got_scores));
    }
}

test "e2e: a point's payload is stored with it and comes back with with_payload, and a bare upsert clears it" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "pl", 4, 3), &out);

    // One point with `payload { "k": Value{ string_value: "v" }, "n": 7 }`:
    // map entries at field 3, key = 1, value = 2; `Value.string_value` = 4
    // and `Value.integer_value` = 3 (VERIFIED, json_with_int.proto). Two
    // entries, because the decoder used to keep only the last.
    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "pl");
    try w.writeBoolField(2, true);
    {
        const pt = try w.beginNested(3, 3);
        {
            const idn = try w.beginNested(1, 2);
            try w.writeVarintFieldAlways(1, 1);
            try w.endNested(idn);
        }
        try writePayloadEntry(&w, "k", "v");
        {
            const entry = try w.beginNested(3, 2);
            try w.writeStringField(1, "n");
            const val = try w.beginNested(2, 2);
            try w.writeVarintFieldAlways(3, 7);
            try w.endNested(val);
            try w.endNested(entry);
        }
        {
            const vs = try w.beginNested(4, 3);
            const vec = try w.beginNested(1, 3);
            const dv = try w.beginNested(101, 3);
            try w.writePackedFloats(1, &[_]f32{ 1, 2, 3, 4 });
            try w.endNested(dv);
            try w.endNested(vec);
            try w.endNested(vs);
        }
        try w.endNested(pt);
    }
    const resp = try c.call("/qdrant.Points/Upsert", w.written(), &out);
    try testing.expectEqual(grpc.Status.ok, resp.status);
    try testing.expectEqual(@as(usize, 1), h.engine.find("pl").?.count());

    // Both entries come back, bytes as sent, under `ScoredPoint.payload`.
    var pts: [4]PointPayload = undefined;
    const q = [_]f32{ 1, 2, 3, 4 };
    const qr = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "pl", &q, .{ .with_payload = true }), &out);
    try testing.expectEqual(grpc.Status.ok, qr.status);
    const n = try readFirstBatchPayloads(qr.body, &pts);
    try testing.expectEqual(@as(usize, 1), n);
    try testing.expectEqual(@as(u64, 1), pts[0].id);
    try testing.expectEqualStrings("v", pts[0].keyword("k").?);
    try testing.expectEqual(@as(?i64, 7), pts[0].integer("n"));
    // Without `with_payload`, none of it is sent.
    const qr2 = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "pl", &q, .{}), &out);
    try testing.expectEqual(@as(usize, 1), try readFirstBatchPayloads(qr2.body, &pts));
    try testing.expect(pts[0].keyword("k") == null);

    // The same point upserted without a payload has none: an upsert replaces
    // the payload whole, as Qdrant's does.
    const ids = [_]u64{1};
    const v = [_]f32{ 1, 2, 3, 4 };
    const sl = [_][]const f32{&v};
    const ok_resp = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "pl", true, &ids, &sl), &out);
    try testing.expectEqual(grpc.Status.ok, ok_resp.status);
    const qr3 = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "pl", &q, .{ .with_payload = true }), &out);
    try testing.expectEqual(@as(usize, 1), try readFirstBatchPayloads(qr3.body, &pts));
    try testing.expect(pts[0].keyword("k") == null);
}

/// A `{ key: Value{ string_value } }` map entry at field 3, inside a point
/// (or a `SetPayloadPoints`, which uses the same field number).
fn writePayloadEntry(w: *wire.Writer, key: []const u8, value: []const u8) !void {
    const entry = try w.beginNested(3, 2);
    try w.writeStringField(1, key);
    const val = try w.beginNested(2, 2);
    try w.writeStringField(4, value);
    try w.endNested(val);
    try w.endNested(entry);
}

fn writeIntegerPayloadEntry(w: *wire.Writer, key: []const u8, value: i64) !void {
    const entry = try w.beginNested(3, 2);
    try w.writeStringField(1, key);
    const val = try w.beginNested(2, 2);
    try w.writeVarintFieldAlways(3, @bitCast(value));
    try w.endNested(val);
    try w.endNested(entry);
}

/// `buildUpsert` with a keyword payload `a` and an integer payload `n` per
/// point — bfb's `-k` shape plus the integer `tag` docs/workloads.md asks
/// W12 to grow into.
fn buildUpsertWithPayload(buf: []u8, name: []const u8, ids: []const u64, vecs: []const []const f32, kws: []const []const u8, ints: []const i64) ![]const u8 {
    var w = wire.Writer.init(buf);
    try w.writeStringField(1, name);
    try w.writeBoolField(2, true);
    for (ids, vecs, kws, ints) |id, v, kw, n| {
        const pt = try w.beginNested(3, 3);
        {
            const idn = try w.beginNested(1, 2);
            try w.writeVarintFieldAlways(1, id);
            try w.endNested(idn);
        }
        try writePayloadEntry(&w, "a", kw);
        try writeIntegerPayloadEntry(&w, "n", n);
        {
            const vs = try w.beginNested(4, 3);
            const vec = try w.beginNested(1, 3);
            const dv = try w.beginNested(101, 3);
            try w.writePackedFloats(1, v);
            try w.endNested(dv);
            try w.endNested(vec);
            try w.endNested(vs);
        }
        try w.endNested(pt);
    }
    return w.written();
}

const FilterSpec = struct {
    /// `must` on `key` matching `keyword`, or on `key` matching `integer`.
    key: []const u8 = "a",
    keyword: ?[]const u8 = null,
    integer: ?i64 = null,
    /// A second `must_not` keyword condition on the same key.
    must_not_keyword: ?[]const u8 = null,
    /// Two `should` keyword conditions on `key`.
    should: ?[2][]const u8 = null,
};

const FilteredQueryOpts = struct {
    limit: u64 = 10,
    hnsw_ef: ?u64 = null,
    exact: bool = false,
    with_payload: bool = false,
    filter: ?FilterSpec = null,
};

fn writeKeywordCondition(w: *wire.Writer, clause: u32, key: []const u8, keyword: []const u8) !void {
    const cond = try w.beginNested(clause, 2);
    const fc = try w.beginNested(1, 2);
    try w.writeStringField(1, key);
    const m = try w.beginNested(2, 2);
    try w.writeStringField(1, keyword);
    try w.endNested(m);
    try w.endNested(fc);
    try w.endNested(cond);
}

/// A one-query `QueryBatch` in bfb's W12 shape: `Filter.must = 2` holding a
/// `Condition.field = 1` with `FieldCondition { key = 1; match = 2 }` and
/// `Match.keyword = 1` (or `Match.integer = 2`); `with_payload = 11`.
fn buildFilteredQuery(buf: []u8, name: []const u8, q: []const f32, opts: FilteredQueryOpts) ![]const u8 {
    var w = wire.Writer.init(buf);
    try w.writeStringField(1, name);
    const qp = try w.beginNested(2, 3);
    try w.writeStringField(1, name);
    {
        const query = try w.beginNested(3, 3);
        const nearest = try w.beginNested(1, 3);
        const dv = try w.beginNested(2, 3);
        try w.writePackedFloats(1, q);
        try w.endNested(dv);
        try w.endNested(nearest);
        try w.endNested(query);
    }
    if (opts.filter) |f| {
        const fl = try w.beginNested(5, 2);
        if (f.keyword) |kw| try writeKeywordCondition(&w, 2, f.key, kw);
        if (f.integer) |i| {
            const cond = try w.beginNested(2, 2);
            const fc = try w.beginNested(1, 2);
            try w.writeStringField(1, f.key);
            const m = try w.beginNested(2, 2);
            try w.writeVarintFieldAlways(2, @bitCast(i));
            try w.endNested(m);
            try w.endNested(fc);
            try w.endNested(cond);
        }
        if (f.must_not_keyword) |kw| try writeKeywordCondition(&w, 3, f.key, kw);
        if (f.should) |pair| {
            try writeKeywordCondition(&w, 1, f.key, pair[0]);
            try writeKeywordCondition(&w, 1, f.key, pair[1]);
        }
        try w.endNested(fl);
    }
    if (opts.hnsw_ef != null or opts.exact) {
        const params = try w.beginNested(6, 2);
        if (opts.hnsw_ef) |ef| try w.writeVarintField(1, ef);
        try w.writeBoolField(2, opts.exact);
        try w.endNested(params);
    }
    try w.writeVarintFieldAlways(8, opts.limit);
    if (opts.with_payload) {
        const wp = try w.beginNested(11, 2);
        try w.writeBoolField(1, true);
        try w.endNested(wp);
    }
    try w.endNested(qp);
    return w.written();
}

/// One point of a response with its payload entries kept as bytes.
const PointPayload = struct {
    id: u64 = 0,
    score: f32 = 0,
    entries: [8][]const u8 = undefined,
    n: usize = 0,

    fn keyword(self: *const PointPayload, key: []const u8) ?[]const u8 {
        for (self.entries[0..self.n]) |e| {
            const kv = core.payload.splitEntry(e) orelse continue;
            if (!std.mem.eql(u8, kv.key, key)) continue;
            var r = wire.Reader.init(kv.value);
            while (!r.atEnd()) {
                const t = r.tag() catch return null;
                if (t.field == 4) return r.bytes() catch null;
                r.skip(t.wire_type) catch return null;
            }
        }
        return null;
    }

    fn integer(self: *const PointPayload, key: []const u8) ?i64 {
        for (self.entries[0..self.n]) |e| {
            const kv = core.payload.splitEntry(e) orelse continue;
            if (!std.mem.eql(u8, kv.key, key)) continue;
            var r = wire.Reader.init(kv.value);
            while (!r.atEnd()) {
                const t = r.tag() catch return null;
                if (t.field == 3) return @bitCast(r.varint() catch return null);
                r.skip(t.wire_type) catch return null;
            }
        }
        return null;
    }
};

/// The first `BatchResult`'s points with their payloads. Returns the count.
fn readFirstBatchPayloads(body: []const u8, pts: []PointPayload) !usize {
    var r = wire.Reader.init(body);
    var n: usize = 0;
    while (!r.atEnd()) {
        const t = try r.tag();
        if (t.field != 1) {
            try r.skip(t.wire_type);
            continue;
        }
        var batch = try r.nested();
        while (!batch.atEnd()) {
            const bt = try batch.tag();
            if (bt.field != 1) {
                try batch.skip(bt.wire_type);
                continue;
            }
            var sp = try batch.nested();
            var p = PointPayload{};
            while (!sp.atEnd()) {
                const st = try sp.tag();
                switch (st.field) {
                    1 => {
                        var idr = try sp.nested();
                        p.id = (try msg.PointId.decode(&idr)).num;
                    },
                    2 => {
                        const e = try sp.bytes();
                        if (p.n < p.entries.len) {
                            p.entries[p.n] = e;
                            p.n += 1;
                        }
                    },
                    3 => p.score = try sp.float(),
                    else => try sp.skip(st.wire_type),
                }
            }
            if (n < pts.len) pts[n] = p;
            n += 1;
        }
        return n;
    }
    return n;
}

/// `CreateFieldIndexCollection { collection_name = 1; wait = 2; field_name = 3;
/// field_type = 4 }`, the call bfb makes before uploading with `-k`.
fn buildCreateFieldIndex(buf: []u8, name: []const u8, field: []const u8, field_type: u64) ![]const u8 {
    var w = wire.Writer.init(buf);
    try w.writeStringField(1, name);
    try w.writeBoolField(2, true);
    try w.writeStringField(3, field);
    try w.writeVarintFieldAlways(4, field_type);
    return w.written();
}

/// Names of the `payload_schema` entries (field 8) in a `CollectionInfo`,
/// with their `data_type`.
const SchemaEntry = struct { name: []const u8, data_type: u64, points: u64 };

fn readPayloadSchema(body: []const u8, out: []SchemaEntry) !usize {
    var r = wire.Reader.init(body);
    _ = try r.tag();
    var info = try r.nested();
    var n: usize = 0;
    while (!info.atEnd()) {
        const t = try info.tag();
        if (t.field != 8) {
            try info.skip(t.wire_type);
            continue;
        }
        var entry = try info.nested();
        var e = SchemaEntry{ .name = &.{}, .data_type = 0, .points = 0 };
        while (!entry.atEnd()) {
            const et = try entry.tag();
            switch (et.field) {
                1 => e.name = try entry.bytes(),
                2 => {
                    var si = try entry.nested();
                    while (!si.atEnd()) {
                        const st = try si.tag();
                        switch (st.field) {
                            1 => e.data_type = try si.varint(),
                            3 => e.points = try si.varint(),
                            else => try si.skip(st.wire_type),
                        }
                    }
                },
                else => try entry.skip(et.wire_type),
            }
        }
        if (n < out.len) out[n] = e;
        n += 1;
    }
    return n;
}

/// The exact filtered top-k over `stored` (d=16, Euclid): the ids, nearest
/// first, of points `admit` accepts.
fn filteredTruth(stored: []const f32, q: []const f32, admit: *const fn (usize) bool, k: usize, out: []u64) usize {
    const n = stored.len / 16;
    var best_ids: [64]u64 = undefined;
    var best_d: [64]f32 = undefined;
    var m: usize = 0;
    for (0..n) |i| {
        if (!admit(i)) continue;
        var d: f32 = 0;
        for (0..16) |j| {
            const x = stored[i * 16 + j] - q[j];
            d += x * x;
        }
        // Insert into the sorted top-k.
        var pos = m;
        while (pos > 0 and best_d[pos - 1] > d) : (pos -= 1) {
            if (pos < k) {
                best_d[pos] = best_d[pos - 1];
                best_ids[pos] = best_ids[pos - 1];
            }
        }
        if (pos < k) {
            best_d[pos] = d;
            best_ids[pos] = i;
            if (m < k) m += 1;
        }
    }
    @memcpy(out[0..m], best_ids[0..m]);
    return m;
}

test "e2e: W12's shape: CreateFieldIndex, keyword payloads, filtered QueryBatch exact on every path" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;

    // `full_scan_threshold = 1` KB, 16 points at d=16 fp32, so a filter
    // admitting more than that traverses the graph under the predicate and
    // one admitting fewer scores its set directly: both of Qdrant's arms.
    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "w12");
        {
            const hc = try w.beginNested(4, 2); // hnsw_config
            try w.writeVarintFieldAlways(3, 1); // full_scan_threshold, KB
            try w.endNested(hc);
        }
        {
            const vc = try w.beginNested(10, 2); // vectors_config (VERIFIED: 10)
            const p = try w.beginNested(1, 2);
            try w.writeVarintFieldAlways(1, 16);
            try w.writeVarintFieldAlways(2, 2);
            try w.endNested(p);
            try w.endNested(vc);
        }
        const cr = try c.call("/qdrant.Collections/Create", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, cr.status);
    }
    // bfb creates the keyword index *before* uploading (`create_field_indices`).
    {
        const resp = try c.call("/qdrant.Points/CreateFieldIndex", try buildCreateFieldIndex(&req_buf, "w12", "a", 0), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }

    const n = 600;
    var prng = std.Random.DefaultPrng.init(0x1212);
    const rnd = prng.random();
    const stored = try testing.allocator.alloc(f32, n * 16);
    defer testing.allocator.free(stored);
    for (stored) |*x| x.* = rnd.floatNorm(f32);
    const Tags = struct {
        fn kw(i: usize) []const u8 {
            return switch (i % 5) {
                0 => "keyword_0",
                1 => "keyword_1",
                2 => "keyword_2",
                3 => "keyword_3",
                else => "keyword_4",
            };
        }
        fn tag(i: usize) i64 {
            return @intCast(i % 7);
        }
        fn isK1(i: usize) bool {
            return i % 5 == 1;
        }
        fn isTag3(i: usize) bool {
            return i % 7 == 3;
        }
        fn isRare(i: usize) bool {
            return i < 10;
        }
        fn isK0or4(i: usize) bool {
            return i % 5 == 0 or i % 5 == 4;
        }
    };
    var batch_ids: [50]u64 = undefined;
    var batch_vecs: [50][]const f32 = undefined;
    var batch_kws: [50][]const u8 = undefined;
    var batch_ints: [50]i64 = undefined;
    var i: usize = 0;
    while (i < n) : (i += 50) {
        for (0..50) |j| {
            batch_ids[j] = i + j;
            batch_vecs[j] = stored[(i + j) * 16 ..][0..16];
            // The first ten carry a rarer keyword instead, so a filter on it
            // admits fewer points than the threshold.
            batch_kws[j] = if (Tags.isRare(i + j)) "rare" else Tags.kw(i + j);
            batch_ints[j] = Tags.tag(i + j);
        }
        const resp = try c.call("/qdrant.Points/Upsert", try buildUpsertWithPayload(&req_buf, "w12", &batch_ids, &batch_vecs, &batch_kws, &batch_ints), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }
    _ = try waitGreen(&c, "w12", &req_buf, &out);

    const q = stored[7 * 16 ..][0..16];
    var truth: [16]u64 = undefined;
    var pts: [16]PointPayload = undefined;

    // An indexed keyword, 120 of 600 admitted. Scored directly: at n=600 the
    // dispatch crossover (`plainFilteredSearch`, sqrt(ef*m0*n)) is past the
    // collection itself, so every filter on a collection this small scores
    // its matching set rather than traversing. Exact either way, which is
    // what this asserts; which arm runs is pinned by that function's own
    // test, and the traversal arm is exercised at the end of this test.
    {
        const tn = filteredTruth(stored, q, &Tags.isK1, 10, &truth);
        try testing.expectEqual(@as(usize, 10), tn);
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "w12", q, .{
            .hnsw_ef = n,
            .with_payload = true,
            .filter = .{ .keyword = "keyword_1" },
        }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expectEqual(@as(usize, 10), try readFirstBatchPayloads(resp.body, &pts));
        for (pts[0..10], truth[0..10]) |p, want| {
            try testing.expectEqual(want, p.id);
            try testing.expectEqualStrings("keyword_1", p.keyword("a").?);
        }
    }
    // "rare" admits 10 points, fewer than `limit`, and the page says so
    // rather than padding.
    {
        const tn = filteredTruth(stored, q, &Tags.isRare, 10, &truth);
        try testing.expectEqual(@as(usize, 10), tn);
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "w12", q, .{
            .limit = 16,
            .with_payload = true,
            .filter = .{ .keyword = "rare" },
        }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expectEqual(@as(usize, 10), try readFirstBatchPayloads(resp.body, &pts));
        for (pts[0..10], truth[0..10]) |p, want| try testing.expectEqual(want, p.id);
    }
    // Unindexed integer field: evaluated against the blobs (no
    // `CreateFieldIndex` for `n` yet), on the graph.
    {
        const tn = filteredTruth(stored, q, &Tags.isTag3, 10, &truth);
        try testing.expectEqual(@as(usize, 10), tn);
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "w12", q, .{
            .hnsw_ef = n,
            .with_payload = true,
            .filter = .{ .key = "n", .integer = 3 },
        }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expectEqual(@as(usize, 10), try readFirstBatchPayloads(resp.body, &pts));
        for (pts[0..10], truth[0..10]) |p, want| {
            try testing.expectEqual(want, p.id);
            try testing.expectEqual(@as(?i64, 3), p.integer("n"));
        }
        // Then indexed after the fact, over the points already stored: the
        // same answer from the posting list.
        const cf = try c.call("/qdrant.Points/CreateFieldIndex", try buildCreateFieldIndex(&req_buf, "w12", "n", 1), &out);
        try testing.expectEqual(grpc.Status.ok, cf.status);
        const again = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "w12", q, .{
            .hnsw_ef = n,
            .filter = .{ .key = "n", .integer = 3 },
        }), &out);
        try testing.expectEqual(@as(usize, 10), try readFirstBatchPayloads(again.body, &pts));
        for (pts[0..10], truth[0..10]) |p, want| try testing.expectEqual(want, p.id);
    }
    // `exact` over a filter: the fp32 scan, filtered, same answer.
    {
        const tk = filteredTruth(stored, q, &Tags.isK1, 10, &truth);
        try testing.expectEqual(@as(usize, 10), tk);
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "w12", q, .{
            .exact = true,
            .filter = .{ .keyword = "keyword_1" },
        }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expectEqual(@as(usize, 10), try readFirstBatchPayloads(resp.body, &pts));
        for (pts[0..10], truth[0..10]) |p, want| try testing.expectEqual(want, p.id);
    }
    // must + must_not on one key: keyword_1 minus keyword_1 is nothing;
    // keyword_1 minus keyword_2 is keyword_1.
    {
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "w12", q, .{
            .hnsw_ef = n,
            .filter = .{ .keyword = "keyword_1", .must_not_keyword = "keyword_1" },
        }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expectEqual(@as(usize, 0), try readFirstBatchPayloads(resp.body, &pts));
        const resp2 = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "w12", q, .{
            .hnsw_ef = n,
            .filter = .{ .keyword = "keyword_1", .must_not_keyword = "keyword_2" },
        }), &out);
        const tk = filteredTruth(stored, q, &Tags.isK1, 10, &truth);
        try testing.expectEqual(@as(usize, 10), try readFirstBatchPayloads(resp2.body, &pts));
        for (pts[0..tk], truth[0..tk]) |p, want| try testing.expectEqual(want, p.id);
    }
    // `should`: either of two keywords.
    {
        const tn = filteredTruth(stored, q, &Tags.isK0or4, 10, &truth);
        try testing.expectEqual(@as(usize, 10), tn);
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "w12", q, .{
            .hnsw_ef = n,
            .with_payload = true,
            .filter = .{ .should = .{ "keyword_0", "keyword_4" } },
        }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expectEqual(@as(usize, 10), try readFirstBatchPayloads(resp.body, &pts));
        for (pts[0..10], truth[0..10]) |p, want| {
            try testing.expectEqual(want, p.id);
            const kw = p.keyword("a").?;
            try testing.expect(std.mem.eql(u8, kw, "keyword_0") or std.mem.eql(u8, kw, "keyword_4"));
        }
    }
    // A value nobody carries: an empty page, not an error.
    {
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "w12", q, .{
            .filter = .{ .keyword = "keyword_9" },
        }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expectEqual(@as(usize, 0), try readFirstBatchPayloads(resp.body, &pts));
    }
    // The traversal arm, which the cases above no longer reach on a
    // collection this small: a `must_not` alone has no indexed `must` for
    // `select` to build a bitset from, so the filter is evaluated against
    // the blobs *during* a graph walk. Permissive (590 of 600 admitted) at a
    // narrow `ef`, so the result heap fills and the walk terminates — the
    // regime the dispatch keeps for the traversal. Approximate by
    // construction at ef=16, so what is asserted is the filter's own
    // property: a full page, and nothing on it that the filter excludes.
    {
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "w12", q, .{
            .hnsw_ef = 16,
            .with_payload = true,
            .filter = .{ .must_not_keyword = "rare" },
        }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expectEqual(@as(usize, 10), try readFirstBatchPayloads(resp.body, &pts));
        for (pts[0..10]) |p| {
            try testing.expect(!std.mem.eql(u8, p.keyword("a").?, "rare"));
            try testing.expect(p.id >= 10); // the first ten are the rare ones
        }
    }

    // The harness reads the index back from `payload_schema`
    // (docs/workloads.md, W12 point 1): both fields, with their types.
    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "w12");
        const resp = try c.call("/qdrant.Collections/Get", w.written(), &out);
        var schema: [4]SchemaEntry = undefined;
        try testing.expectEqual(@as(usize, 2), try readPayloadSchema(resp.body, &schema));
        try testing.expectEqualStrings("a", schema[0].name);
        try testing.expectEqual(@as(u64, 1), schema[0].data_type); // Keyword
        try testing.expectEqual(@as(u64, n), schema[0].points);
        try testing.expectEqualStrings("n", schema[1].name);
        try testing.expectEqual(@as(u64, 2), schema[1].data_type); // Integer
    }
}

test "e2e: Scroll with a filter pages the matching points in id order, with their payloads" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;
    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "sf", 2, 3), &out);

    var ids: [40]u64 = undefined;
    var vecs: [40][]const f32 = undefined;
    var kws: [40][]const u8 = undefined;
    var ints: [40]i64 = undefined;
    const v0 = [_]f32{ 1, 0 };
    for (0..40) |i| {
        ids[i] = 39 - i; // out of order on the wire, in order on the page
        vecs[i] = &v0;
        kws[i] = if ((39 - i) % 2 == 1) "odd" else "even";
        ints[i] = @intCast(39 - i);
    }
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsertWithPayload(&req_buf, "sf", &ids, &vecs, &kws, &ints), &out);

    // `ScrollPoints { collection_name = 1; filter = 2; limit = 4; with_payload = 6 }`.
    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "sf");
    {
        const fl = try w.beginNested(2, 2);
        try writeKeywordCondition(&w, 2, "a", "odd");
        try w.endNested(fl);
    }
    try w.writeVarintField(4, 5);
    {
        const wp = try w.beginNested(6, 2);
        try w.writeBoolField(1, true);
        try w.endNested(wp);
    }
    const p1 = try c.call("/qdrant.Points/Scroll", w.written(), &out);
    try testing.expectEqual(grpc.Status.ok, p1.status);
    const page = try decodeScroll(p1.body);
    try testing.expectEqual(@as(usize, 5), page.n);
    for (page.ids[0..5], [_]u64{ 1, 3, 5, 7, 9 }) |got, want| try testing.expectEqual(want, got);
    // The cursor is the next *admitted* point.
    try testing.expectEqual(@as(?u64, 11), page.next);
    // And the payload rode along: the first point's entries carry `a = odd`.
    var r = wire.Reader.init(p1.body);
    var saw_payload = false;
    while (!r.atEnd()) {
        const t = try r.tag();
        if (t.field != 2) {
            try r.skip(t.wire_type);
            continue;
        }
        var rp = try r.nested();
        while (!rp.atEnd()) {
            const rt = try rp.tag();
            if (rt.field == 2) {
                const kv = core.payload.splitEntry(try rp.bytes()).?;
                if (std.mem.eql(u8, kv.key, "a")) saw_payload = true;
            } else try rp.skip(rt.wire_type);
        }
        break;
    }
    try testing.expect(saw_payload);
}

test "e2e: SetPayload merges keys, and a filter sees the merged value" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;
    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "sp", 2, 3), &out);

    var ids: [4]u64 = .{ 0, 1, 2, 3 };
    var vecs: [4][]const f32 = undefined;
    var kws: [4][]const u8 = .{ "k0", "k0", "k0", "k0" };
    var ints: [4]i64 = .{ 0, 1, 2, 3 };
    const v0 = [_]f32{ 1, 0 };
    for (&vecs) |*v| v.* = &v0;
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsertWithPayload(&req_buf, "sp", &ids, &vecs, &kws, &ints), &out);

    // `SetPayloadPoints { collection_name = 1; wait = 2; payload = 3;
    // points_selector = 5 { PointsIdsList points = 1 { ids = 1 } } }`.
    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "sp");
    try w.writeBoolField(2, true);
    try writePayloadEntry(&w, "a", "zzz");
    {
        const sel = try w.beginNested(5, 2);
        const list = try w.beginNested(1, 2);
        const pid = try w.beginNested(1, 2);
        try w.writeVarintFieldAlways(1, 3);
        try w.endNested(pid);
        try w.endNested(list);
        try w.endNested(sel);
    }
    const resp = try c.call("/qdrant.Points/SetPayload", w.written(), &out);
    try testing.expectEqual(grpc.Status.ok, resp.status);

    const q = [_]f32{ 1, 0 };
    var pts: [4]PointPayload = undefined;
    const fq = try c.call("/qdrant.Points/QueryBatch", try buildFilteredQuery(&req_buf, "sp", &q, .{
        .with_payload = true,
        .filter = .{ .keyword = "zzz" },
    }), &out);
    try testing.expectEqual(grpc.Status.ok, fq.status);
    try testing.expectEqual(@as(usize, 1), try readFirstBatchPayloads(fq.body, &pts));
    try testing.expectEqual(@as(u64, 3), pts[0].id);
    // The key not set survived the merge.
    try testing.expectEqual(@as(?i64, 3), pts[0].integer("n"));

    // A point that does not exist: NOT_FOUND, as Qdrant answers.
    var w2 = wire.Writer.init(&req_buf);
    try w2.writeStringField(1, "sp");
    try writePayloadEntry(&w2, "a", "zzz");
    {
        const sel = try w2.beginNested(5, 2);
        const list = try w2.beginNested(1, 2);
        const pid = try w2.beginNested(1, 2);
        try w2.writeVarintFieldAlways(1, 99);
        try w2.endNested(pid);
        try w2.endNested(list);
        try w2.endNested(sel);
    }
    const missing = try c.call("/qdrant.Points/SetPayload", w2.written(), &out);
    try testing.expectEqual(grpc.Status.not_found, missing.status);
}

test "e2e: filter constructs outside the keyword/integer match are refused by name" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;
    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "rf", 2, 3), &out);

    // A `Range` condition (`FieldCondition.range = 3`).
    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "rf");
    const qp = try w.beginNested(2, 3);
    try w.writeStringField(1, "rf");
    {
        const query = try w.beginNested(3, 3);
        const nearest = try w.beginNested(1, 3);
        const dv = try w.beginNested(2, 3);
        try w.writePackedFloats(1, &[_]f32{ 1, 0 });
        try w.endNested(dv);
        try w.endNested(nearest);
        try w.endNested(query);
    }
    {
        const fl = try w.beginNested(5, 2);
        const cond = try w.beginNested(2, 2);
        const fc = try w.beginNested(1, 2);
        try w.writeStringField(1, "n");
        const range = try w.beginNested(3, 2);
        try w.writeDoubleField(2, 1.5); // gt
        try w.endNested(range);
        try w.endNested(fc);
        try w.endNested(cond);
        try w.endNested(fl);
    }
    try w.endNested(qp);
    const resp = try c.call("/qdrant.Points/QueryBatch", w.written(), &out);
    try testing.expectEqual(grpc.Status.unimplemented, resp.status);
    try testing.expectEqualStrings("range conditions", resp.message);

    // A geo field index.
    const geo = try c.call("/qdrant.Points/CreateFieldIndex", try buildCreateFieldIndex(&req_buf, "rf", "loc", 3), &out);
    try testing.expectEqual(grpc.Status.unimplemented, geo.status);
    try testing.expect(std.mem.indexOf(u8, geo.message, "keyword and integer") != null);

    // `UpsertPoints.update_mode = 8` (`update_only`): used to be skipped and
    // answered with an unconditional upsert, which inserts the points the
    // client asked not to insert.
    var wu = wire.Writer.init(&req_buf);
    try wu.writeStringField(1, "rf");
    try wu.writeVarintFieldAlways(8, 1);
    const um = try c.call("/qdrant.Points/Upsert", wu.written(), &out);
    try testing.expectEqual(grpc.Status.unimplemented, um.status);
    try testing.expectEqualStrings("UpsertPoints.update_mode", um.message);

    // `ScrollPoints.with_vectors { include = 2 }`: the same refusal
    // `QueryPoints` gives, not a bare `true` that returned every vector.
    var ws = wire.Writer.init(&req_buf);
    try ws.writeStringField(1, "rf");
    {
        const wv = try ws.beginNested(7, 2);
        const inc = try ws.beginNested(2, 2);
        try ws.writeStringField(1, "v");
        try ws.endNested(inc);
        try ws.endNested(wv);
    }
    const sv = try c.call("/qdrant.Points/Scroll", ws.written(), &out);
    try testing.expectEqual(grpc.Status.unimplemented, sv.status);
    try testing.expectEqualStrings("named vector output selectors", sv.message);
}

test "e2e: Delete of a missing collection is result=false, Create over an existing one is ALREADY_EXISTS" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    const readResult = struct {
        fn f(body: []const u8) !bool {
            var r = wire.Reader.init(body);
            var result = false;
            while (!r.atEnd()) {
                const t = try r.tag();
                if (t.field == 1) result = try r.boolean() else try r.skip(t.wire_type);
            }
            return result;
        }
    }.f;

    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "ghost");
        const resp = try c.call("/qdrant.Collections/Delete", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expect(!try readResult(resp.body));
    }
    {
        const resp = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "dup", 4, 3), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expect(try readResult(resp.body));
    }
    {
        const resp = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "dup", 8, 2), &out);
        try testing.expectEqual(grpc.Status.already_exists, resp.status);
        // And the original is untouched.
        try testing.expectEqual(@as(usize, 4), h.engine.find("dup").?.config.dim);
    }
    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "dup");
        const resp = try c.call("/qdrant.Collections/Delete", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expect(try readResult(resp.body));
    }
}

// =========================================================================
// §2: `time` is measured from arrival, not from dequeue
// =========================================================================

/// A handler that takes a fixed 40 ms and answers OK, so queueing is the only
/// variable between requests.
fn slowDispatch(_: *anyopaque, req: *const server.Request, out: *server.ResponseBuf) server.Completion {
    var ts = linux.timespec{ .sec = 0, .nsec = 40 * std.time.ns_per_ms };
    _ = linux.nanosleep(&ts, null);
    var w = wire.Writer.init(out.available());
    w.writeBoolField(1, true) catch unreachable;
    return .{
        .conn = req.conn,
        .stream_idx = req.stream_idx,
        .stream_id = req.stream_id,
        .status = .ok,
        .body = out.commit(w.pos),
    };
}

fn timedCall(port: u16, out_time: *f64, failed: *std.atomic.Value(bool)) void {
    var c = Client.connect(port) catch {
        failed.store(true, .release);
        return;
    };
    defer c.close();
    var out: [1024]u8 = undefined;
    const resp = c.call("/x/y", &.{}, &out) catch {
        failed.store(true, .release);
        return;
    };
    if (resp.status != .ok) {
        failed.store(true, .release);
        return;
    }
    // `time` is field 2, appended by the worker.
    var r = wire.Reader.init(resp.body);
    var t: f64 = -1;
    while (!r.atEnd()) {
        const tag = r.tag() catch break;
        if (tag.field == 2) t = r.double() catch break else r.skip(tag.wire_type) catch break;
    }
    if (t < 0) failed.store(true, .release);
    out_time.* = t;
}

test "e2e: reported time includes the time spent queued for a worker" {
    // Two workers, four simultaneous requests, each 40 ms of handler time.
    // Two run at once; the other two wait a full request behind them, so
    // their `time` must be about twice the first pair's. Timing from dequeue
    // (the old behaviour) reports ~40 ms for all four, hiding a saturated
    // pool from the one number bfb takes as the server-side latency.
    var dummy: u8 = 0;
    const srv = try server.Server.init(testing.allocator, .{
        .addr = server.Address.init(server.Address.loopback, 0),
        .io_threads = 1,
        .worker_threads = 2,
        .streams_per_conn = 2,
        .connections_per_io = 4,
        .request_buffer = 4096,
        .response_buffer = 4096,
    }, slowDispatch, @ptrCast(&dummy));
    defer srv.deinit();
    const port = try srv.bind();
    try srv.start();
    defer srv.stop();

    var times = [_]f64{ 0, 0, 0, 0 };
    var failed = std.atomic.Value(bool).init(false);
    var threads: [4]std.Thread = undefined;
    for (&threads, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, timedCall, .{ port, &times[i], &failed });
    for (threads) |t| t.join();
    try testing.expect(!failed.load(.acquire));

    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    // Everyone waited at least the handler's own 40 ms.
    try testing.expect(times[0] >= 0.039);
    // And the slowest waited for a whole earlier request as well. 1.6x rather
    // than 2x leaves room for scheduling jitter without letting the
    // dequeue-timed answer (ratio ~1.0) through.
    try testing.expect(times[3] >= times[0] * 1.6);
}

// =========================================================================
// Transport: frame sizes, flow control, cancellation, backpressure
// =========================================================================

/// The number of `ScoredPoint`s in each `BatchResult` of a QueryBatch
/// response, in order. Returns how many batches there were.
fn countBatchResults(body: []const u8, counts: []usize) !usize {
    var r = wire.Reader.init(body);
    var n: usize = 0;
    while (!r.atEnd()) {
        const t = try r.tag();
        if (t.field != 1) {
            try r.skip(t.wire_type);
            continue;
        }
        var batch = try r.nested();
        var k: usize = 0;
        while (!batch.atEnd()) {
            const bt = try batch.tag();
            if (bt.field == 1) k += 1;
            try batch.skip(bt.wire_type);
        }
        if (n < counts.len) counts[n] = k;
        n += 1;
    }
    return n;
}

test "e2e: a QueryBatch response over 16 KiB is split into frames the client's max frame size allows" {
    // bfb's default search shape, 16 queries x limit 100, encodes to well
    // over hyper's 16 384-byte SETTINGS_MAX_FRAME_SIZE. Written as one DATA
    // frame it drew FRAME_SIZE_ERROR and a GOAWAY from the client, which
    // took every in-flight call on the connection with it. The test client
    // enforces the same limit (`Client.max_frame_size`), so this failed with
    // `error.FrameSizeError` before the body was split.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();

    const dim = 8;
    const n = 256;
    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;
    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "wide", dim, 3), &out);

    const stored = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(stored);
    var prng = std.Random.DefaultPrng.init(0xbead);
    for (stored) |*x| x.* = prng.random().float(f32);
    try uploadRandom(&c, "wide", dim, stored, &req_buf, &out);

    var qs: [16][]const f32 = undefined;
    for (&qs, 0..) |*q, i| q.* = stored[i * dim ..][0..dim];
    const resp = try c.call("/qdrant.Points/QueryBatch", try buildQueryBatch(&req_buf, "wide", &qs, 100), &out);
    try testing.expectEqual(grpc.Status.ok, resp.status);
    // The shape: several DATA frames, each within the limit (checked by the
    // client as it reads), then trailers.
    try testing.expect(resp.body.len > 16 * 1024);
    try testing.expect(resp.data_frames >= 2);
    try testing.expect(resp.saw_end_stream);

    var counts: [16]usize = undefined;
    try testing.expectEqual(@as(usize, 16), try countBatchResults(resp.body, &counts));
    for (counts) |k| try testing.expectEqual(@as(usize, 100), k);
}

/// A handler that answers with a body of `big_body_len` patterned bytes after
/// a short delay, so several are in flight at once and the responses are far
/// larger than a frame, a small write buffer, or a default flow-control
/// window.
const big_body_len: usize = 200 * 1024;
fn bigDispatch(_: *anyopaque, req: *const server.Request, out: *server.ResponseBuf) server.Completion {
    var ts = linux.timespec{ .sec = 0, .nsec = 10 * std.time.ns_per_ms };
    _ = linux.nanosleep(&ts, null);
    const dst = out.available()[0..big_body_len];
    for (dst, 0..) |*b, i| b.* = @truncate(i ^ req.stream_id);
    return .{
        .conn = req.conn,
        .stream_idx = req.stream_idx,
        .stream_id = req.stream_id,
        .status = .ok,
        .body = out.commit(big_body_len),
        // The body is not a protobuf message; keep the worker from
        // appending a `time` field to it.
        .wants_time = false,
    };
}

fn expectBigBody(resp: Client.Response, sid: u31) !void {
    try testing.expectEqual(grpc.Status.ok, resp.status);
    try testing.expectEqual(big_body_len, resp.body.len);
    for (resp.body, 0..) |b, i| {
        if (b != @as(u8, @truncate(i ^ sid))) return error.BodyCorrupt;
    }
}

/// A server around `bigDispatch` with a deliberately small write buffer.
fn startBigServer(write_buffer: usize, workers: usize) !*server.Server {
    var dummy: u8 = 0;
    const srv = try server.Server.init(testing.allocator, .{
        .addr = server.Address.init(server.Address.loopback, 0),
        .io_threads = 1,
        .worker_threads = workers,
        .streams_per_conn = 8,
        .connections_per_io = 2,
        .request_buffer = 4096,
        .response_buffer = big_body_len + 64,
        .write_buffer = write_buffer,
        // Small, so a client that stops reading backs the write buffer up
        // within a few frames instead of after megabytes of kernel buffer.
        .sndbuf = 16 * 1024,
    }, bigDispatch, @ptrCast(&dummy));
    errdefer srv.deinit();
    _ = try srv.bind();
    try srv.start();
    return srv;
}

fn stopServer(srv: *server.Server) void {
    srv.stop();
    srv.deinit();
}

test "e2e: a body larger than the peer's windows is parked and finishes as WINDOW_UPDATEs arrive" {
    // A client at the RFC defaults, 65 535-byte windows, asking for 200 KiB:
    // the server may only send a window's worth, then must wait. It used to
    // send everything, which a conforming client answers with
    // FLOW_CONTROL_ERROR; the test client checks the same thing.
    const srv = try startBigServer(1 << 20, 2);
    defer stopServer(srv);
    const port = try server.boundPort(srv.io[0].listen_fd);

    var c = try Client.connectWith(port, .{ .stream_window = 65535, .conn_window = 65535 });
    defer c.close();
    const out = try testing.allocator.alloc(u8, big_body_len + 4096);
    defer testing.allocator.free(out);

    // Two in flight, so the connection window is contended as well.
    const a = try c.send("/x/y", &.{});
    const b = try c.send("/x/y", &.{});
    try expectBigBody(try c.readResponse(a, out), a);
    try expectBigBody(try c.readResponse(b, out), b);
    // Every DATA frame was released, so the window is back where it began.
    try testing.expectEqual(@as(i64, 65535), c.conn_window);
}

test "e2e: a slow reader with a small write buffer gets every response and no GOAWAY" {
    // Eight 200 KiB responses into a 32 KiB write buffer, to a client that
    // has stopped reading. Before parking, a completion that did not fit
    // became INTERNAL "response buffer overflow" (or was silently dropped),
    // and a PING arriving while the buffer was full turned into a
    // PROTOCOL_ERROR GOAWAY because its ACK could not be written.
    const srv = try startBigServer(32 * 1024, 2);
    defer stopServer(srv);
    const port = try server.boundPort(srv.io[0].listen_fd);

    var c = try Client.connectWith(port, .{ .rcvbuf = 4096 });
    defer c.close();

    var sids: [8]u31 = undefined;
    for (&sids, 0..) |*sid, i| {
        sid.* = try c.send("/x/y", &.{});
        // Control frames while the server's output is backed up.
        if (i % 2 == 0) try c.ping();
    }
    // Do not read: let the socket, then the write buffer, fill up.
    var ts = linux.timespec{ .sec = 0, .nsec = 300 * std.time.ns_per_ms };
    _ = linux.nanosleep(&ts, null);
    // Now that the buffer is nearly full, a burst of PINGs whose ACKs
    // (17 bytes each) exceed the whole 32 KiB buffer, so the reader has to
    // pause and resume, whatever headroom the parked bodies left.
    for (0..2500) |_| try c.ping();
    _ = linux.nanosleep(&ts, null);

    const out = try testing.allocator.alloc(u8, big_body_len + 4096);
    defer testing.allocator.free(out);
    for (sids) |sid| try expectBigBody(try c.readResponse(sid, out), sid);
    try testing.expectEqual(@as(usize, 0), c.retainedFrames());

    // The connection is still good, and every PING was answered, none was
    // dropped and none turned into a GOAWAY.
    const again = try c.send("/x/y", &.{});
    try expectBigBody(try c.readResponse(again, out), again);
    try testing.expectEqual(@as(usize, 2504), c.ping_acks);
    // The reader did pause on the full buffer at least once, so the above
    // exercised the pause-and-resume path rather than a buffer that never
    // filled.
    try testing.expect(srv.totalStats().pauses > 0);
}

/// A handler that takes 20 ms and echoes the stream id, so a response can be
/// matched to its request.
fn echoDispatch(_: *anyopaque, req: *const server.Request, out: *server.ResponseBuf) server.Completion {
    var ts = linux.timespec{ .sec = 0, .nsec = 20 * std.time.ns_per_ms };
    _ = linux.nanosleep(&ts, null);
    var w = wire.Writer.init(out.available());
    w.writeVarintFieldAlways(1, req.stream_id) catch unreachable;
    return .{
        .conn = req.conn,
        .stream_idx = req.stream_idx,
        .stream_id = req.stream_id,
        .status = .ok,
        .body = out.commit(w.pos),
    };
}

fn expectEcho(resp: Client.Response, sid: u31) !void {
    try testing.expectEqual(grpc.Status.ok, resp.status);
    var r = wire.Reader.init(resp.body);
    const t = try r.tag();
    try testing.expectEqual(@as(u32, 1), t.field);
    try testing.expectEqual(@as(u64, sid), try r.varint());
}

test "e2e: streams cancelled under load are dropped, and every later request is answered" {
    // Four slots, one worker, and a client that cancels two of every four
    // requests while they are queued or running. A cancelled slot used to be
    // freed on the spot; the next HEADERS reused it under the worker, and
    // when that worker's completion arrived its `closeStream` reset the new
    // stream, whose response then never went out, while the cancelled
    // stream's response *did*. Here that shows up as a timeout on a later
    // read, or as frames retained for a cancelled stream id.
    var dummy: u8 = 0;
    const srv = try server.Server.init(testing.allocator, .{
        .addr = server.Address.init(server.Address.loopback, 0),
        .io_threads = 1,
        .worker_threads = 1,
        .streams_per_conn = 4,
        .connections_per_io = 2,
        .request_buffer = 4096,
        .response_buffer = 4096,
    }, echoDispatch, @ptrCast(&dummy));
    defer srv.deinit();
    const port = try srv.bind();
    try srv.start();
    defer srv.stop();

    var c = try Client.connect(port);
    defer c.close();
    var out: [4096]u8 = undefined;

    for (0..8) |round| {
        // Fill the pool: the first runs, the rest queue behind it.
        var sids: [4]u31 = undefined;
        for (&sids) |*sid| sid.* = try c.send("/x/y", &.{});
        // Cancel two: on even rounds two that are queued, on odd rounds the
        // one the worker is running and the one behind it. The RSTs land
        // within microseconds, while the worker is still on the first, so
        // every cancelled stream is dispatched-and-owned when reset. The
        // last stream is always kept: the single worker completes in order,
        // so by the time its response is read every cancelled completion
        // has drained and freed its slot for the next round.
        const drop = if (round % 2 == 0) [_]usize{ 1, 2 } else [_]usize{ 0, 1 };
        for (drop) |d| try c.reset(sids[d]);
        for (sids, 0..) |sid, i| {
            if (i == drop[0] or i == drop[1]) continue;
            try expectEcho(try c.readResponse(sid, &out), sid);
        }
    }
    // Every kept request was answered exactly once above. A cancelled one may
    // legally have been answered if its response beat the RST (counted in
    // `late_cancelled`, timing-dependent, so not asserted either way); what
    // must hold is that nothing is left over, nothing was stranded, and the
    // connection still serves.
    try testing.expectEqual(@as(usize, 0), c.retainedFrames());
    try testing.expectEqual(@as(u64, 32), srv.totalStats().requests);
    const after = try c.send("/x/y", &.{});
    try expectEcho(try c.readResponse(after, &out), after);
    try testing.expectEqual(@as(usize, 0), c.retainedFrames());
}

test "e2e: a stream reset while its body is parked frees the slot and the connection stays usable" {
    // With 65 535-byte windows the 200 KiB body parks after the first
    // window's worth. A reset then must drop the parked remainder and free
    // the slot; the parked state used to be dropped but the slot kept
    // `.closed` forever, since no completion was ever going to free it.
    // Eight slots, so nine further calls fail with REFUSED_STREAM if even one
    // leaked.
    const srv = try startBigServer(1 << 20, 1);
    defer stopServer(srv);
    const port = try server.boundPort(srv.io[0].listen_fd);

    var c = try Client.connectWith(port, .{ .stream_window = 65535, .conn_window = 65535 });
    defer c.close();
    const out = try testing.allocator.alloc(u8, big_body_len + 4096);
    defer testing.allocator.free(out);

    for (0..9) |_| {
        const sid = try c.send("/x/y", &.{});
        // Wait until the first window's worth is here, unread: the body is
        // parked on the server at exactly this point, so the reset lands on
        // a parked stream every time rather than whenever a sleep says.
        try c.awaitData(sid, 65535);
        try c.reset(sid);
        try c.dropResetFrames();
    }
    // Every slot must be free again: eight in flight at once, then a ninth,
    // each answered in full. A leaked slot shows as REFUSED_STREAM.
    var sids: [8]u31 = undefined;
    for (&sids) |*sid| sid.* = try c.send("/x/y", &.{});
    for (sids) |sid| try expectBigBody(try c.readResponse(sid, out), sid);
    const last = try c.send("/x/y", &.{});
    try expectBigBody(try c.readResponse(last, out), last);
    // The parked bodies did go out up to the window, were dropped as late
    // frames for a reset stream, and their bytes released.
    try testing.expect(c.late_cancelled > 0);
    try testing.expectEqual(@as(i64, 65535), c.conn_window);
}

test "e2e: stopping the server sends GOAWAY to open connections" {
    var dummy: u8 = 0;
    const srv = try server.Server.init(testing.allocator, .{
        .addr = server.Address.init(server.Address.loopback, 0),
        .io_threads = 1,
        .worker_threads = 1,
        .streams_per_conn = 2,
        .connections_per_io = 2,
        .request_buffer = 4096,
        .response_buffer = 4096,
    }, echoDispatch, @ptrCast(&dummy));
    defer srv.deinit();
    const port = try srv.bind();
    try srv.start();

    var c = try Client.connect(port);
    defer c.close();
    var out: [4096]u8 = undefined;
    const sid = try c.send("/x/y", &.{});
    try expectEcho(try c.readResponse(sid, &out), sid);

    srv.stop();
    // The next read yields the GOAWAY, then EOF: `readResponse` reports it
    // as an error rather than a timeout, and the frame names the last
    // stream the server processed.
    try testing.expectError(error.Goaway, c.readResponse(sid + 2, &out));
    var pos: usize = 0;
    var saw = false;
    while (pos + h2.frame_header_len <= c.len) {
        const fh = h2.FrameHeader.parse(c.buf[pos..][0..h2.frame_header_len]);
        if (fh.frame_type == .goaway) {
            const last = std.mem.readInt(u32, c.buf[pos + h2.frame_header_len ..][0..4], .big);
            try testing.expectEqual(@as(u32, sid), last);
            saw = true;
        }
        pos += h2.frame_header_len + fh.length;
    }
    try testing.expect(saw);
}

test "e2e: requests arriving during stop are answered or named below the GOAWAY, never stranded" {
    // One worker at 20 ms per request and a queue built up ahead of `stop`,
    // with the client still sending while the workers drain and join. `stop`
    // used to close the submit queue and join the workers with the I/O
    // threads still dispatching into it: those requests were counted in the
    // GOAWAY's last stream id and never answered.
    var dummy: u8 = 0;
    const srv = try server.Server.init(testing.allocator, .{
        .addr = server.Address.init(server.Address.loopback, 0),
        .io_threads = 1,
        .worker_threads = 1,
        .streams_per_conn = 32,
        .connections_per_io = 2,
        .request_buffer = 4096,
        .response_buffer = 4096,
    }, echoDispatch, @ptrCast(&dummy));
    defer srv.deinit();
    const port = try srv.bind();
    try srv.start();

    var c = try Client.connect(port);
    defer c.close();
    var out: [4096]u8 = undefined;

    // Settle the connection (SETTINGS exchanged) before the server goes away.
    const warm = try c.send("/x/y", &.{});
    try expectEcho(try c.readResponse(warm, &out), warm);

    var sids: [32]u31 = undefined;
    var sent: usize = 0;
    // Enough queued that the join takes ~100 ms.
    while (sent < 5) : (sent += 1) sids[sent] = try c.send("/x/y", &.{});
    const stopper = try std.Thread.spawn(.{}, server.Server.stop, .{srv});
    // Keep sending while it stops, until the socket refuses or the slots
    // run out. Sends after the fd is closed fail; that is the end of it.
    while (sent < sids.len) : (sent += 1) {
        sids[sent] = c.send("/x/y", &.{}) catch break;
        var ts = linux.timespec{ .sec = 0, .nsec = 5 * std.time.ns_per_ms };
        _ = linux.nanosleep(&ts, null);
    }
    stopper.join();
    c.peer_may_be_gone = true;

    // Every request is either answered (OK, or UNAVAILABLE once the workers
    // were gone), or lies above the GOAWAY's last stream id. None below it
    // may go unanswered.
    var answered: usize = 0;
    for (sids[0..sent]) |sid| {
        const resp = c.readResponse(sid, &out) catch |e| switch (e) {
            error.Goaway => {
                const last = c.goawayLastStream() orelse return error.GoawayWithoutFrame;
                try testing.expect(last < sid);
                break;
            },
            else => return e,
        };
        switch (resp.status) {
            .ok => try expectEcho(resp, sid),
            .unavailable => {},
            else => return error.UnexpectedStatus,
        }
        answered += 1;
    }
    // The five queued ahead of `stop` were all served.
    try testing.expect(answered >= 5);
}

// =========================================================================
// Semantics pinned after the conformance review
// =========================================================================

test "e2e: Scroll limit 0 is INVALID_ARGUMENT, absent is 10" {
    // Qdrant validates `ScrollPoints.limit` as `range(min = 1)` and defaults
    // it to 10. An explicit 0 used to be an empty page *with* a cursor: a
    // pager following `next_page_offset` never terminates.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "sl", 2, 3), &out);
    var ids: [12]u64 = undefined;
    var vecs: [12][]const f32 = undefined;
    const v0 = [_]f32{ 1, 0 };
    for (&ids, 0..) |*id, i| {
        id.* = i;
        vecs[i] = &v0;
    }
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "sl", true, &ids, &vecs), &out);

    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "sl");
        try w.writeVarintFieldAlways(4, 0);
        const resp = try c.call("/qdrant.Points/Scroll", w.written(), &out);
        try testing.expectEqual(grpc.Status.invalid_argument, resp.status);
        try testing.expect(std.mem.indexOf(u8, resp.message, "limit") != null);
    }
    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "sl");
        const resp = try c.call("/qdrant.Points/Scroll", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        const page = try decodeScroll(resp.body);
        try testing.expectEqual(@as(usize, 10), page.n);
        try testing.expectEqual(@as(u64, 10), page.next.?);
    }
}

test "e2e: an exact query does not start a background build, an approximate one does" {
    // W9 (`--search-exact`) is the bandwidth measurement, and a build it did
    // not ask for would run on `build_threads` cores underneath it.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "ex2", 8, 3), &out);
    var prng = std.Random.DefaultPrng.init(0xe7);
    const rnd = prng.random();
    const stored = try testing.allocator.alloc(f32, 100 * 8);
    defer testing.allocator.free(stored);
    for (stored) |*x| x.* = rnd.floatNorm(f32);
    try uploadRandom(&c, "ex2", 8, stored, &req_buf, &out);
    const coll = h.engine.find("ex2").?;
    try testing.expectEqual(core.collection.IndexState.absent, coll.index_state.load(.acquire));

    const q = stored[0..8];
    for (0..5) |_| {
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "ex2", q, .{ .limit = 5, .exact = true }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }
    try testing.expectEqual(core.collection.IndexState.absent, coll.index_state.load(.acquire));

    const resp = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "ex2", q, .{ .limit = 5 }), &out);
    try testing.expectEqual(grpc.Status.ok, resp.status);
    try testing.expect(coll.index_state.load(.acquire) != .absent);
}

test "e2e: below full_scan_threshold, an approximate query does not start a build either" {
    // The trigger keys on the computed `exact`: a collection Qdrant would
    // plain-scan is one whose graph would never be consulted, so building it
    // on the query path is the W9 confound under a different name.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    // Default threshold (10 000 KB): the helper's collection sends 0, so this
    // one is built by hand without an hnsw_config at all.
    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "tiny");
    const vc = try w.beginNested(10, 2);
    const pp = try w.beginNested(1, 2);
    try w.writeVarintField(1, 8);
    try w.writeVarintField(2, 3); // Dot
    try w.endNested(pp);
    try w.endNested(vc);
    const create = try testing.allocator.dupe(u8, w.written());
    defer testing.allocator.free(create);
    _ = try c.call("/qdrant.Collections/Create", create, &out);

    var prng = std.Random.DefaultPrng.init(0x71);
    const rnd = prng.random();
    const stored = try testing.allocator.alloc(f32, 100 * 8);
    defer testing.allocator.free(stored);
    for (stored) |*x| x.* = rnd.floatNorm(f32);
    try uploadRandom(&c, "tiny", 8, stored, &req_buf, &out);
    const coll = h.engine.find("tiny").?;
    try testing.expectEqual(core.collection.IndexState.absent, coll.index_state.load(.acquire));

    // No Collections/Get here (that path polls a build on purpose): only
    // approximate queries, which are answered by the plain scan.
    for (0..5) |_| {
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "tiny", stored[0..8], .{ .limit = 5, .hnsw_ef = 16 }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
    }
    try testing.expectEqual(core.collection.IndexState.absent, coll.index_state.load(.acquire));
}

test "e2e: two concurrent Creates of one name: one result=true, one ALREADY_EXISTS, config from the winner" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();

    const Racer = struct {
        fn run(port: u16, dim: u64, status: *grpc.Status, start: *std.atomic.Value(bool)) void {
            var c = Client.connect(port) catch {
                status.* = .internal;
                return;
            };
            defer c.close();
            var req_buf: [4096]u8 = undefined;
            var out: [4096]u8 = undefined;
            const body = buildCreateCollection(&req_buf, "twice", dim, 3) catch {
                status.* = .internal;
                return;
            };
            while (!start.load(.acquire)) std.atomic.spinLoopHint();
            const resp = c.call("/qdrant.Collections/Create", body, &out) catch {
                status.* = .internal;
                return;
            };
            status.* = resp.status;
        }
    };
    var statuses: [2]grpc.Status = .{ .unknown, .unknown };
    var start = std.atomic.Value(bool).init(false);
    const t0 = try std.Thread.spawn(.{}, Racer.run, .{ h.port, 4, &statuses[0], &start });
    const t1 = try std.Thread.spawn(.{}, Racer.run, .{ h.port, 8, &statuses[1], &start });
    start.store(true, .release);
    t0.join();
    t1.join();

    var oks: usize = 0;
    var exists: usize = 0;
    var winner_dim: usize = 0;
    for (statuses, [_]usize{ 4, 8 }) |st, dim| {
        switch (st) {
            .ok => {
                oks += 1;
                winner_dim = dim;
            },
            .already_exists => exists += 1,
            else => return error.UnexpectedStatus,
        }
    }
    try testing.expectEqual(@as(usize, 1), oks);
    try testing.expectEqual(@as(usize, 1), exists);
    try testing.expectEqual(winner_dim, h.engine.find("twice").?.config.dim);
}

test "e2e: below full_scan_threshold a query is answered exactly, as Qdrant 1.19 answers it" {
    // 1000 points at d=128 fp32 is 512 KB, under Qdrant's default threshold
    // of 10 000 KB, so Qdrant answers every unfiltered query with a plain
    // scan whatever `hnsw_ef` says. strawmann used to accept the field and
    // traverse anyway, so a small-collection comparison was exact against
    // approximate.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 18]u8 = undefined;
    var out: [1 << 18]u8 = undefined;

    const dim = 128;
    const n = 1000;
    var prng = std.Random.DefaultPrng.init(0xf5c4);
    const rnd = prng.random();
    const stored = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(stored);
    for (stored) |*x| x.* = rnd.floatNorm(f32);

    // Two collections over the same data: Qdrant's default threshold, and
    // threshold 0. Both get a deliberately poor graph (M=4, ef_construct=1)
    // so that "the graph was consulted" is visible as imperfect recall at
    // ef=16 rather than hidden by a graph good enough to be exact anyway.
    const Create = struct {
        fn build(buf: []u8, name: []const u8, threshold: ?u64) ![]const u8 {
            var w = wire.Writer.init(buf);
            try w.writeStringField(1, name);
            {
                const hc = try w.beginNested(4, 2);
                try w.writeVarintField(1, 4); // m
                try w.writeVarintField(2, 1); // ef_construct
                if (threshold) |t| try w.writeVarintFieldAlways(3, t);
                try w.endNested(hc);
            }
            const vc = try w.beginNested(10, 2);
            const p = try w.beginNested(1, 2);
            try w.writeVarintField(1, dim);
            try w.writeVarintField(2, 3); // Dot
            try w.endNested(p);
            try w.endNested(vc);
            return w.written();
        }
    };
    _ = try c.call("/qdrant.Collections/Create", try Create.build(&req_buf, "small", null), &out);
    _ = try c.call("/qdrant.Collections/Create", try Create.build(&req_buf, "graph", 0), &out);
    try uploadRandom(&c, "small", dim, stored, &req_buf, &out);
    try uploadRandom(&c, "graph", dim, stored, &req_buf, &out);
    _ = try waitGreen(&c, "small", &req_buf, &out);
    _ = try waitGreen(&c, "graph", &req_buf, &out);

    // The threshold reads back in `CollectionInfo.config.hnsw_config`.
    try testing.expectEqual(@as(?u64, 10_000), (try getInfo(&c, "small", &req_buf, &out)).hnsw_full_scan_threshold);
    try testing.expectEqual(@as(?u64, 0), (try getInfo(&c, "graph", &req_buf, &out)).hnsw_full_scan_threshold);

    var truth_ids: [10]u64 = undefined;
    var got_ids: [10]u64 = undefined;
    var scores: [10]f32 = undefined;
    var small_mismatch: usize = 0;
    var graph_mismatch: usize = 0;
    for (0..20) |_| {
        var q: [dim]f32 = undefined;
        for (&q) |*x| x.* = rnd.floatNorm(f32);
        const exact = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "small", &q, .{ .limit = 10, .exact = true }), &out);
        try testing.expectEqual(@as(usize, 10), try readFirstBatch(exact.body, &truth_ids, &scores));

        const small = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "small", &q, .{ .limit = 10, .hnsw_ef = 16 }), &out);
        try testing.expectEqual(@as(usize, 10), try readFirstBatch(small.body, &got_ids, &scores));
        if (!std.mem.eql(u64, &truth_ids, &got_ids)) small_mismatch += 1;

        const graph = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "graph", &q, .{ .limit = 10, .hnsw_ef = 16 }), &out);
        var got_n: [10]u64 = undefined;
        const gn = try readFirstBatch(graph.body, &got_n, &scores);
        if (gn != 10 or !std.mem.eql(u64, &truth_ids, &got_n)) graph_mismatch += 1;
    }
    // Under the threshold: exactly the exact answer, every time, at ef=16.
    try testing.expectEqual(@as(usize, 0), small_mismatch);
    // Threshold 0: the graph answered, and this graph is not exact.
    try testing.expect(graph_mismatch > 0);
}

test "e2e: QueryBatch offset pages past the first results" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "off", 2, 3), &out);
    // Point i scores i against [1, 0], so the ranking is 5, 4, 3, 2, 1.
    const ids = [_]u64{ 1, 2, 3, 4, 5 };
    const v = [_][2]f32{ .{ 1, 0 }, .{ 2, 0 }, .{ 3, 0 }, .{ 4, 0 }, .{ 5, 0 } };
    var vecs: [5][]const f32 = undefined;
    for (&vecs, 0..) |*p, i| p.* = &v[i];
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "off", true, &ids, &vecs), &out);

    var got: [8]u64 = undefined;
    var scores: [8]f32 = undefined;
    const q = [_]f32{ 1, 0 };
    {
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "off", &q, .{ .limit = 2, .offset = 2 }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expectEqual(@as(usize, 2), try readFirstBatch(resp.body, &got, &scores));
        try testing.expectEqualSlices(u64, &.{ 3, 2 }, got[0..2]);
    }
    // An offset past the end is an empty page, not an error.
    {
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "off", &q, .{ .limit = 2, .offset = 10 }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expectEqual(@as(usize, 0), try readFirstBatch(resp.body, &got, &scores));
    }
}

test "e2e: QueryBatch limit + offset beyond the result heap is INVALID_ARGUMENT, not a silent clamp" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "lim", 2, 3), &out);
    const ids = [_]u64{ 1, 2 };
    const v0 = [_]f32{ 1, 0 };
    var vecs: [2][]const f32 = .{ &v0, &v0 };
    _ = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "lim", true, &ids, &vecs), &out);

    const m = api.Workspace.max_limit;
    const q = [_]f32{ 1, 0 };
    const cases = [_]struct { limit: u64, offset: ?u64, status: grpc.Status }{
        .{ .limit = 5000, .offset = null, .status = .invalid_argument },
        .{ .limit = m, .offset = 1, .status = .invalid_argument },
        .{ .limit = 1, .offset = std.math.maxInt(u64), .status = .invalid_argument },
        .{ .limit = m, .offset = null, .status = .ok },
        .{ .limit = m - 10, .offset = 10, .status = .ok },
    };
    for (cases) |cs| {
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "lim", &q, .{ .limit = cs.limit, .offset = cs.offset }), &out);
        try testing.expectEqual(cs.status, resp.status);
        // Loud, and naming the bound, the way an oversized body is refused.
        if (cs.status != .ok) try testing.expect(std.mem.indexOf(u8, resp.message, "4096") != null);
    }
}

test "e2e: a NaN or infinite component is INVALID_ARGUMENT on upsert and on query" {
    // A NaN score compares false both ways in the result heaps, so one such
    // vector would poison every search that touched it. Qdrant refuses them.
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "nan", 2, 3), &out);

    const bad = [_][2]f32{
        .{ 1, std.math.nan(f32) },
        .{ std.math.inf(f32), 0 },
        .{ 0, -std.math.inf(f32) },
    };
    for (bad) |b| {
        const ids = [_]u64{7};
        var vecs: [1][]const f32 = .{&b};
        const up = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "nan", true, &ids, &vecs), &out);
        try testing.expectEqual(grpc.Status.invalid_argument, up.status);
        try testing.expect(std.mem.indexOf(u8, up.message, "NaN") != null);

        const qr = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "nan", &b, .{}), &out);
        try testing.expectEqual(grpc.Status.invalid_argument, qr.status);
        try testing.expect(std.mem.indexOf(u8, qr.message, "NaN") != null);
    }
    // Nothing was written by the refused upserts, and a finite one still works.
    try testing.expectEqual(@as(usize, 0), h.engine.find("nan").?.count());
    const good = [_]f32{ 1, 0 };
    const ids = [_]u64{7};
    var vecs: [1][]const f32 = .{&good};
    try testing.expectEqual(grpc.Status.ok, (try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "nan", true, &ids, &vecs), &out)).status);
    try testing.expectEqual(grpc.Status.ok, (try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "nan", &good, .{}), &out)).status);
}

test "e2e: an empty collection name is INVALID_ARGUMENT at create" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [4096]u8 = undefined;
    var out: [4096]u8 = undefined;

    const resp = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "", 2, 3), &out);
    try testing.expectEqual(grpc.Status.invalid_argument, resp.status);
    try testing.expect(std.mem.indexOf(u8, resp.message, "collection_name") != null);
    try testing.expect(h.engine.find("") == null);
}

test "e2e: a vector of the wrong dimension is INVALID_ARGUMENT on upsert and on query" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "dim", 4, 3), &out);
    for ([_][]const f32{ &[_]f32{ 1, 0, 0 }, &[_]f32{ 1, 0, 0, 0, 0 } }) |v| {
        const ids = [_]u64{1};
        var vecs: [1][]const f32 = .{v};
        const up = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "dim", true, &ids, &vecs), &out);
        try testing.expectEqual(grpc.Status.invalid_argument, up.status);
        try testing.expect(std.mem.indexOf(u8, up.message, "dimension") != null);

        const qr = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "dim", v, .{}), &out);
        try testing.expectEqual(grpc.Status.invalid_argument, qr.status);
        try testing.expect(std.mem.indexOf(u8, qr.message, "dimension") != null);
    }
    try testing.expectEqual(@as(usize, 0), h.engine.find("dim").?.count());
}

test "e2e: Upsert, QueryBatch and Scroll on a missing collection are NOT_FOUND" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    const v = [_]f32{ 1, 0 };
    const ids = [_]u64{1};
    var vecs: [1][]const f32 = .{&v};
    const up = try c.call("/qdrant.Points/Upsert", try buildUpsert(&req_buf, "ghost", true, &ids, &vecs), &out);
    try testing.expectEqual(grpc.Status.not_found, up.status);

    const qr = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "ghost", &v, .{}), &out);
    try testing.expectEqual(grpc.Status.not_found, qr.status);

    var w = wire.Writer.init(&req_buf);
    try w.writeStringField(1, "ghost");
    const sc = try c.call("/qdrant.Points/Scroll", w.written(), &out);
    try testing.expectEqual(grpc.Status.not_found, sc.status);
}

/// `buildUpsert` for string (UUID) ids: `PointId.uuid = 2`.
fn buildUpsertUuid(buf: []u8, name: []const u8, ids: []const []const u8, vecs: []const []const f32) ![]const u8 {
    var w = wire.Writer.init(buf);
    try w.writeStringField(1, name);
    try w.writeBoolField(2, true);
    for (ids, vecs) |id, v| {
        const pt = try w.beginNested(3, 3);
        {
            const idn = try w.beginNested(1, 2);
            try w.writeStringField(2, id);
            try w.endNested(idn);
        }
        {
            const vs = try w.beginNested(4, 3);
            const vec = try w.beginNested(1, 3);
            const dv = try w.beginNested(101, 3);
            try w.writePackedFloats(1, v);
            try w.endNested(dv);
            try w.endNested(vec);
            try w.endNested(vs);
        }
        try w.endNested(pt);
    }
    return w.written();
}

/// The UUID text of every `PointId` in `body` at `path`: the ids of the first
/// `BatchResult` for a QueryBatch response, or of the page for a Scroll
/// response (both nest `PointId` at field 1 of a repeated field). Copies each
/// out, since the response buffer is reused.
fn readUuidIds(body: []const u8, list_field: u32, out_ids: [][36]u8) !usize {
    var r = wire.Reader.init(body);
    var n: usize = 0;
    while (!r.atEnd()) {
        const t = try r.tag();
        if (t.field != list_field) {
            try r.skip(t.wire_type);
            continue;
        }
        var list = try r.nested();
        // QueryBatch: BatchResult { repeated ScoredPoint result = 1 }; Scroll:
        // the repeated RetrievedPoint is the list itself, one per tag.
        if (list_field == 1) {
            while (!list.atEnd()) {
                const bt = try list.tag();
                if (bt.field != 1) {
                    try list.skip(bt.wire_type);
                    continue;
                }
                var sp = try list.nested();
                n += try readOneUuid(&sp, out_ids[n..]);
            }
        } else {
            n += try readOneUuid(&list, out_ids[n..]);
        }
    }
    return n;
}

/// Scroll's `next_page_offset` (field 1, a bare `PointId`), copied out.
fn scrollNextUuid(body: []const u8, out_id: *[36]u8) !bool {
    var r = wire.Reader.init(body);
    while (!r.atEnd()) {
        const t = try r.tag();
        if (t.field != 1) {
            try r.skip(t.wire_type);
            continue;
        }
        var idr = try r.nested();
        const s = switch (try msg.PointId.decode(&idr)) {
            .uuid => |u| u,
            .num => return error.ExpectedUuid,
        };
        if (s.len != 36) return error.NotCanonical;
        @memcpy(out_id, s);
        return true;
    }
    return false;
}

fn readOneUuid(sp: *wire.Reader, out_ids: [][36]u8) !usize {
    while (!sp.atEnd()) {
        const st = try sp.tag();
        if (st.field == 1) {
            var idr = try sp.nested();
            const pid = try msg.PointId.decode(&idr);
            const s = switch (pid) {
                .uuid => |u| u,
                .num => return error.ExpectedUuid,
            };
            if (s.len != 36) return error.NotCanonical;
            @memcpy(&out_ids[0], s);
            return 1;
        } else try sp.skip(st.wire_type);
    }
    return 0;
}

test "e2e: UUID ids round-trip through upsert, query and scroll in canonical form" {
    var h = try Harness.start(testing.allocator);
    defer h.stop();
    var c = try Client.connect(h.port);
    defer c.close();
    var req_buf: [1 << 16]u8 = undefined;
    var out: [1 << 16]u8 = undefined;

    _ = try c.call("/qdrant.Collections/Create", try buildCreateCollection(&req_buf, "uu", 2, 3), &out);

    // Three spellings Qdrant accepts, deliberately not in sorted order, and
    // one canonical id per point so the answers below can be checked by eye:
    // point k scores k against [1, 0].
    const spelled = [_][]const u8{
        "{00000000-0000-4000-8000-000000000003}",
        "urn:uuid:00000000-0000-4000-8000-000000000001",
        "00000000000040008000000000000002",
    };
    const canon = [_][]const u8{
        "00000000-0000-4000-8000-000000000003",
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
    };
    const v = [_][2]f32{ .{ 3, 0 }, .{ 1, 0 }, .{ 2, 0 } };
    var vecs: [3][]const f32 = undefined;
    for (&vecs, 0..) |*p, i| p.* = &v[i];
    const up = try c.call("/qdrant.Points/Upsert", try buildUpsertUuid(&req_buf, "uu", &spelled, &vecs), &out);
    try testing.expectEqual(grpc.Status.ok, up.status);
    try testing.expectEqual(@as(usize, 3), h.engine.find("uu").?.count());

    // Query: ranked 3, 2, 1, every id in the canonical hyphenated lowercase form.
    var got: [4][36]u8 = undefined;
    {
        const q = [_]f32{ 1, 0 };
        const resp = try c.call("/qdrant.Points/QueryBatch", try buildQuery(&req_buf, "uu", &q, .{ .limit = 3 }), &out);
        try testing.expectEqual(grpc.Status.ok, resp.status);
        try testing.expectEqual(@as(usize, 3), try readUuidIds(resp.body, 1, &got));
        try testing.expectEqualStrings(canon[0], &got[0]);
        try testing.expectEqualStrings(canon[2], &got[1]);
        try testing.expectEqualStrings(canon[1], &got[2]);
    }

    // Scroll: id order, first page of two, cursor to the third, then the rest.
    {
        var w = wire.Writer.init(&req_buf);
        try w.writeStringField(1, "uu");
        try w.writeVarintField(4, 2);
        const p1 = try c.call("/qdrant.Points/Scroll", w.written(), &out);
        try testing.expectEqual(grpc.Status.ok, p1.status);
        try testing.expectEqual(@as(usize, 2), try readUuidIds(p1.body, 2, &got));
        try testing.expectEqualStrings(canon[1], &got[0]);
        try testing.expectEqualStrings(canon[2], &got[1]);
        var next: [36]u8 = undefined;
        try testing.expect(try scrollNextUuid(p1.body, &next));
        try testing.expectEqualStrings(canon[0], &next);

        // Resume from the cursor, spelled the simple way, to show the offset
        // goes through the same lenient parse.
        var w2 = wire.Writer.init(&req_buf);
        try w2.writeStringField(1, "uu");
        {
            const off = try w2.beginNested(3, 2);
            try w2.writeStringField(2, "00000000000040008000000000000003");
            try w2.endNested(off);
        }
        try w2.writeVarintField(4, 2);
        const p2 = try c.call("/qdrant.Points/Scroll", w2.written(), &out);
        try testing.expectEqual(grpc.Status.ok, p2.status);
        try testing.expectEqual(@as(usize, 1), try readUuidIds(p2.body, 2, &got));
        try testing.expectEqualStrings(canon[0], &got[0]);
        try testing.expect(!try scrollNextUuid(p2.body, &next));
    }

    // Upserting under a different spelling of the same id overwrites, it does
    // not add: the forms are one id.
    {
        const again = [_][]const u8{"00000000-0000-4000-8000-000000000003"};
        const nv = [_]f32{ 9, 0 };
        var nvecs: [1][]const f32 = .{&nv};
        const r2 = try c.call("/qdrant.Points/Upsert", try buildUpsertUuid(&req_buf, "uu", &again, &nvecs), &out);
        try testing.expectEqual(grpc.Status.ok, r2.status);
        try testing.expectEqual(@as(usize, 3), h.engine.find("uu").?.count());
    }

    // And a malformed one is still refused.
    {
        const badid = [_][]const u8{"not-a-uuid"};
        const nv = [_]f32{ 1, 0 };
        var nvecs: [1][]const f32 = .{&nv};
        const r3 = try c.call("/qdrant.Points/Upsert", try buildUpsertUuid(&req_buf, "uu", &badid, &nvecs), &out);
        try testing.expectEqual(grpc.Status.invalid_argument, r3.status);
    }
}

test "e2e: every deferred RPC answers UNIMPLEMENTED, names itself, and cites §2" {
    // `Rpc.disposition` groups the routed methods into `implemented`,
    // `deferred`, `legacy_alias` and `unlisted`, and the comment above the
    // dispatch says why the enum is closed: adding a member should be a
    // compile error "rather than a working endpoint that answers UNIMPLEMENTED
    // forever". Nothing checked the other half of that — that everything in
    // `deferred` still *does* answer UNIMPLEMENTED, with the message §12 asks
    // for. `Points/Delete` had a test and the other five did not, so five
    // shipped refusals were unexercised.
    //
    // `CreateFieldIndex` and `SetPayload` were on this list until 2026-09-03,
    // and the first had a bill attached: bfb panics on the refusal rather
    // than degrading, which is why every run passed `--skip-field-indices`
    // and why Qdrant's W12 measured the cost of *this* engine's missing
    // feature (findings 43). Both answer now (`payload.zig`), so they are
    // exercised by the payload tests instead.
    var h = try Harness.start(testing.allocator);
    defer h.stop();

    var c = try Client.connect(h.port);
    defer c.close();

    var out: [4096]u8 = undefined;

    const deferred = [_][]const u8{
        "/qdrant.Collections/List",
        "/qdrant.Collections/Update",
        "/qdrant.Points/Delete",
        "/qdrant.Points/Get",
    };

    for (deferred) |path| {
        const resp = try c.call(path, &.{}, &out);
        try testing.expectEqual(grpc.Status.unimplemented, resp.status);
        // §12: "with a message naming the RPC" — and naming *which* one, since
        // a client reaching several of these cannot tell them apart otherwise.
        try testing.expect(std.mem.indexOf(u8, resp.message, path[1..]) != null);
        // Deferred to a later phase, not a §1 non-goal. Sending a client to
        // "non-goals" for a scheduled RPC is a wrong answer that reads like a
        // right one.
        try testing.expect(std.mem.indexOf(u8, resp.message, "§2") != null);
        try testing.expect(std.mem.indexOf(u8, resp.message, "§1") == null);
        // Trailers-only: a refusal carries no DATA frame.
        try testing.expectEqual(@as(usize, 0), resp.data_frames);
        try testing.expectEqual(@as(usize, 1), resp.headers_frames);
    }
}
