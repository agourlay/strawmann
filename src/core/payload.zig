//! §3: "**Payload**: opaque, append-only blob per point, phase 2." And M7's
//! keyword index, which is what turns W12 from a row this engine declines into
//! a filtered-search row.
//!
//! ## What is stored
//!
//! The map entries of `PointStruct.payload` (`map<string, Value>`, field 3),
//! **as they arrived on the wire**, framed one after another as
//! `[varint len][entry bytes]`. An entry is a protobuf map entry — a nested
//! message with `key = 1` (string) and `value = 2` (`qdrant.Value`) — and the
//! response messages carry the same map under a different field number
//! (`ScoredPoint.payload = 2`, `RetrievedPoint.payload = 2`), so `with_payload`
//! is a re-tag and a copy, never a decode and re-encode. Opaque, as §3 says:
//! nothing here knows what a `Value` *means* beyond the two kinds a filter can
//! ask about.
//!
//! ## Where it lives
//!
//! An append-only chunk arena. Chunks are allocated as they fill and never
//! moved, so a reader holding a slice into one is safe for as long as the
//! store exists; a point's slot is a single `u64` — chunk-relative offset and
//! length packed — published with release ordering *after* the bytes are
//! written, so a reader that can see the slot can see the blob. Overwriting a
//! point's payload appends a new blob and repoints the slot; the old bytes are
//! garbage until the store is freed, which is §3's "no compaction" applied to
//! payloads. The write side runs under `Collection.write_lock`; the read side
//! takes no lock (§6.3).
//!
//! ## The index
//!
//! Per indexed field, a posting list per value: `keyword` maps string to
//! offsets, `integer` maps `i64` to offsets. Postings are **hints, never
//! truth**: an overwrite leaves the old value's posting in place rather than
//! searching it out (a linear scan of a posting list on every overwrite is the
//! cost W11 would pay for an index it does not use), so every offset a posting
//! yields is re-checked against the point's actual blob before it is admitted,
//! unless no write has yet made any posting stale (`Store.exact_postings`).
//! The index can only over-approximate, never miss, and an unindexed filter is
//! answered by evaluating the blob directly, so the answer does not depend on
//! whether `CreateFieldIndex` was ever called, only its cost does. That is
//! Qdrant's contract too, and decisions.md records what the unindexed cost is:
//! a scan of every point per query.
//!
//! Only the postings take a lock — a `RwLock`, read for the length of one
//! `select` and written for one insert — because the maps behind them are
//! rehashed as they grow, and a reader on a map mid-rehash is a use-after-free.

const std = @import("std");
const wire = @import("../proto/wire.zig");
const futex_lock = @import("../lock.zig");
const text = @import("../text/text.zig");

// `qdrant.Value { oneof kind { NullValue null_value = 1; double double_value = 2;
//   int64 integer_value = 3; string string_value = 4; bool bool_value = 5;
//   Struct struct_value = 6; ListValue list_value = 7; } }` (VERIFIED,
// json_with_int.proto in qdrant-client 1.19.0). `ListValue { repeated Value
// values = 1; }`.
pub const value_null: u32 = 1;
pub const value_double: u32 = 2;
pub const value_integer: u32 = 3;
pub const value_string: u32 = 4;
pub const value_bool: u32 = 5;
/// `Struct { map<string, Value> fields = 1; }`, which a path's key steps
/// descend into.
pub const value_struct: u32 = 6;
pub const value_list: u32 = 7;

/// A map entry, on the wire: `key = 1`, `value = 2`.
const entry_key: u32 = 1;
const entry_value: u32 = 2;

// =========================================================================
// Filters
// =========================================================================

/// One `Match`, the part of `FieldCondition` this engine evaluates.
///
/// `Match { oneof match_value { string keyword = 1; int64 integer = 2;
/// bool boolean = 3; string text = 4; RepeatedStrings keywords = 5;
/// RepeatedIntegers integers = 6; ... } }` (VERIFIED, points.proto). `text`,
/// `phrase`, `prefix` and the `except_*` forms are refused by name at decode.
pub const Match = union(enum) {
    keyword: []const u8,
    integer: i64,
    boolean: bool,
    /// The bytes of a `RepeatedStrings { repeated string strings = 1; }`:
    /// any one of them matching is a match (`match_any`).
    keywords: []const u8,
    /// The bytes of a `RepeatedIntegers { repeated int64 integers = 1; }`.
    integers: []const u8,
    /// `FieldCondition.range`, which is not a `Match` on the wire but
    /// evaluates like one: per value, any list element matching.
    range: Range,
    /// `FieldCondition.datetime_range`: per value, any list element that is
    /// a string Qdrant parses as a datetime and falls in the range.
    datetime: DatetimeRange,
    /// `FieldCondition.values_count`. This and the two below look at a
    /// value's shape, not at its elements, and decide a missing field too.
    values_count: ValuesCount,
    /// `FieldCondition.is_empty`: any value the path reaches is (or is not)
    /// empty, as `check_field_condition` (query_checker.rs) ORs
    /// `FieldCondition::check` over them.
    is_empty: bool,
    /// `FieldCondition.is_null`, and `Condition.is_null` as `true`: the two
    /// agree, since `check_is_null` (lib/segment/src/common/utils.rs) is an
    /// any-of too.
    is_null: bool,
    /// `Condition.is_empty`: every value the path reaches is empty, none
    /// included (`check_is_empty`, lib/segment/src/common/utils.rs, an
    /// all-of). It differs from `is_empty = true` only where a path reaches
    /// several values, `[1, null]` through `a[]` say.
    all_empty,
};

/// `Range { optional double lt = 1; gt = 2; gte = 3; lte = 4; }` (VERIFIED,
/// qdrant_common.proto), normalized to one lower and one upper bound.
///
/// Qdrant compares in `OrderedFloat` (`Range<OrderedFloat<f64>>::check_range`,
/// lib/segment/src/types.rs), a total order with every NaN equal and above
/// every number, and `-0.0 == 0.0`. In that order the bounds a request gives
/// fold exactly into the strictest lower and the strictest upper bound, and
/// the absent ones are the order's ends: `>= -inf` and `<= NaN` hold for
/// every number. So a range with no bounds matches every numeric value, as
/// Qdrant's does, and a NaN bound behaves as Qdrant's does (`lt: NaN` admits
/// every number, `gt: NaN` none). A NaN *value* never matches: Qdrant cannot
/// hold one (`proto_to_json` refuses a NaN `double_value` at upsert,
/// lib/api/src/conversions/json.rs), and the float index does not store it.
pub const Range = struct {
    lo: f64 = -std.math.inf(f64),
    lo_strict: bool = false,
    hi: f64 = std.math.nan(f64),
    hi_strict: bool = false,

    pub fn fromBounds(lt: ?f64, gt: ?f64, gte: ?f64, lte: ?f64) Range {
        var r = Range{};
        if (gt) |x| r.tightenLo(x, true);
        if (gte) |x| r.tightenLo(x, false);
        if (lt) |x| r.tightenHi(x, true);
        if (lte) |x| r.tightenHi(x, false);
        return r;
    }

    fn tightenLo(self: *Range, x: f64, strict: bool) void {
        switch (orderedCmp(x, self.lo)) {
            .gt => {
                self.lo = x;
                self.lo_strict = strict;
            },
            .eq => self.lo_strict = self.lo_strict or strict,
            .lt => {},
        }
    }

    fn tightenHi(self: *Range, x: f64, strict: bool) void {
        switch (orderedCmp(x, self.hi)) {
            .lt => {
                self.hi = x;
                self.hi_strict = strict;
            },
            .eq => self.hi_strict = self.hi_strict or strict,
            .gt => {},
        }
    }

    /// `v` clears the lower bound. Monotone in `v`, which the float index's
    /// binary search relies on.
    fn aboveLo(self: Range, v: f64) bool {
        return switch (orderedCmp(v, self.lo)) {
            .gt => true,
            .eq => !self.lo_strict,
            .lt => false,
        };
    }

    fn belowHi(self: Range, v: f64) bool {
        return switch (orderedCmp(v, self.hi)) {
            .lt => true,
            .eq => !self.hi_strict,
            .gt => false,
        };
    }

    pub fn contains(self: Range, v: f64) bool {
        if (std.math.isNan(v)) return false;
        return self.aboveLo(v) and self.belowHi(v);
    }
};

/// `OrderedFloat`'s order: NaN equal to NaN and above everything else.
fn orderedCmp(a: f64, b: f64) std.math.Order {
    if (std.math.isNan(a)) return if (std.math.isNan(b)) .eq else .gt;
    if (std.math.isNan(b)) return .lt;
    return std.math.order(a, b);
}

/// `ValuesCount { optional uint64 lt = 1; gt = 2; gte = 3; lte = 4; }`
/// (VERIFIED, qdrant_common.proto), folded into `[lo, hi)`. `u65` so that
/// `gt = 2^64 - 1` and `lte = 2^64 - 1` fold without overflow; the count it
/// is compared with is a `usize`, as Qdrant's (`ValuesCount::check_count`,
/// lib/segment/src/types.rs, every bound ANDed).
pub const ValuesCount = struct {
    lo: u65 = 0,
    hi: u65 = std.math.maxInt(u65),

    pub fn fromBounds(lt: ?u64, gt: ?u64, gte: ?u64, lte: ?u64) ValuesCount {
        var v = ValuesCount{};
        if (gt) |x| v.lo = @max(v.lo, @as(u65, x) + 1);
        if (gte) |x| v.lo = @max(v.lo, x);
        if (lt) |x| v.hi = @min(v.hi, x);
        if (lte) |x| v.hi = @min(v.hi, @as(u65, x) + 1);
        return v;
    }

    pub fn contains(self: ValuesCount, n: usize) bool {
        return n >= self.lo and n < self.hi;
    }
};

/// A UTC instant at Qdrant's precision. Qdrant truncates every datetime it
/// compares to whole microseconds, a payload string and a filter bound alike
/// (`From<DateTime<Utc>> for DateTimeWrapper`, `trunc_subsecs(6)`,
/// lib/segment/src/types.rs). `micros` reaches past 999_999 only for a leap
/// second, which chrono keeps as second 59 with the excess in the fraction,
/// so the pair compares in chrono's order where a single count would not.
pub const Instant = struct {
    secs: i64,
    micros: u32,

    pub const min: Instant = .{ .secs = std.math.minInt(i64), .micros = 0 };
    pub const max: Instant = .{ .secs = std.math.maxInt(i64), .micros = std.math.maxInt(u32) };

    pub fn order(a: Instant, b: Instant) std.math.Order {
        const o = std.math.order(a.secs, b.secs);
        return if (o != .eq) o else std.math.order(a.micros, b.micros);
    }
};

/// `DatetimeRange { optional google.protobuf.Timestamp lt = 1; gt = 2;
/// gte = 3; lte = 4; }` (VERIFIED, qdrant_common.proto), folded as `Range`
/// is. Instants are totally ordered, so the strictest bound on each side
/// is exactly Qdrant's four ANDed comparisons (`Range::check_range`); the
/// absent ends are sentinels no parsed instant reaches.
pub const DatetimeRange = struct {
    lo: Instant = Instant.min,
    lo_strict: bool = false,
    hi: Instant = Instant.max,
    hi_strict: bool = false,

    pub fn fromBounds(lt: ?Instant, gt: ?Instant, gte: ?Instant, lte: ?Instant) DatetimeRange {
        var r = DatetimeRange{};
        if (gt) |x| r.tightenLo(x, true);
        if (gte) |x| r.tightenLo(x, false);
        if (lt) |x| r.tightenHi(x, true);
        if (lte) |x| r.tightenHi(x, false);
        return r;
    }

    fn tightenLo(self: *DatetimeRange, x: Instant, strict: bool) void {
        switch (x.order(self.lo)) {
            .gt => {
                self.lo = x;
                self.lo_strict = strict;
            },
            .eq => self.lo_strict = self.lo_strict or strict,
            .lt => {},
        }
    }

    fn tightenHi(self: *DatetimeRange, x: Instant, strict: bool) void {
        switch (x.order(self.hi)) {
            .lt => {
                self.hi = x;
                self.hi_strict = strict;
            },
            .eq => self.hi_strict = self.hi_strict or strict,
            .gt => {},
        }
    }

    pub fn contains(self: DatetimeRange, v: Instant) bool {
        const above = switch (v.order(self.lo)) {
            .gt => true,
            .eq => !self.lo_strict,
            .lt => false,
        };
        const below = switch (v.order(self.hi)) {
            .lt => true,
            .eq => !self.hi_strict,
            .gt => false,
        };
        return above and below;
    }
};

/// Days from 1970-01-01 to a proleptic Gregorian date (Hinnant's
/// `days_from_civil`), the calendar chrono uses.
pub fn daysFromCivil(year: i64, month: u32, day: u32) i64 {
    const y = if (month <= 2) year - 1 else year;
    const era = @divFloor(y, 400);
    const yoe = y - era * 400;
    const m: i64 = month;
    const doy = @divFloor(153 * (if (m > 2) m - 3 else m + 9) + 2, 5) + day - 1;
    const doe = yoe * 365 + @divFloor(yoe, 4) - @divFloor(yoe, 100) + doy;
    return era * 146097 + doe - 719468;
}

fn daysInMonth(year: i64, month: u32) u32 {
    return switch (month) {
        2 => if (@mod(year, 4) == 0 and (@mod(year, 100) != 0 or @mod(year, 400) == 0)) 29 else 28,
        4, 6, 9, 11 => 30,
        else => 31,
    };
}

/// The seconds a `google.protobuf.Timestamp` bound may hold: chrono's
/// `DateTime::from_timestamp` refuses a date outside `NaiveDate`'s years,
/// `(i32::MIN >> 13) + 1` to `(i32::MAX >> 13) - 1` (chrono 0.4.45,
/// naive/date/mod.rs `MIN_YEAR`/`MAX_YEAR`), and Qdrant answers that
/// `invalid_argument` (`try_date_time_from_proto`, lib/api/src/grpc/conversions.rs).
pub const timestamp_min_secs: i64 = daysFromCivil(-262143, 1, 1) * 86_400;
pub const timestamp_max_secs: i64 = daysFromCivil(262142, 12, 31) * 86_400 + 86_399;

/// A `Timestamp { int64 seconds = 1; int32 nanos = 2; }` bound as Qdrant
/// reads it, or null where Qdrant refuses it. A negative `nanos` is read as
/// 0 (`nanos.try_into().unwrap_or(0)`); `nanos` of a second or more is a leap
/// second, valid only on second 59 of a minute and below two seconds
/// (chrono `NaiveTime::from_num_seconds_from_midnight_opt`).
pub fn instantFromTimestamp(seconds: i64, nanos: i32) ?Instant {
    if (seconds < timestamp_min_secs or seconds > timestamp_max_secs) return null;
    const n: u32 = if (nanos < 0) 0 else @intCast(nanos);
    if (n >= 2_000_000_000) return null;
    if (n >= 1_000_000_000 and @mod(seconds, 60) != 59) return null;
    return .{ .secs = seconds, .micros = n / 1000 };
}

/// A payload string as Qdrant's `DateTimePayloadType::from_str` reads it
/// (lib/segment/src/types.rs), restricted to the forms below, every one of
/// which Qdrant accepts with the same instant:
///
///   `YYYY-MM-DD`                                  midnight UTC
///   `YYYY-MM-DD[T ]HH:MM`                         UTC
///   `YYYY-MM-DD[T ]HH:MM:SS[.f]`                  UTC
///   `YYYY-MM-DD[Tt ]HH:MM:SS[.f](Z|z|+HH:MM|+HHMM)`
///   `YYYY-MM-DD[T ]HH:MM:SS[.f]+HH`
///
/// with `-` for `+` too, any number of fraction digits (beyond six they are
/// truncated away, as Qdrant's microseconds are), second 60 a leap second,
/// and an offset below 24 hours. The first three are its `NaiveDateTime`
/// formats; the rest are `parse_from_rfc3339`, chrono's relaxed `FromStr`
/// and `%Y-%m-%dT%H:%M:%S%.f%#z`. What chrono also takes and this does not
/// (whitespace around the parts, a `UTC` suffix, U+2212 as the minus, a year
/// that is not four digits) is null, so such a value never matches a
/// datetime range here where Qdrant's would.
pub fn parseDatetime(s: []const u8) ?Instant {
    if (s.len < 10 or s[4] != '-' or s[7] != '-') return null;
    const year = digitsAt(s, 0, 4) orelse return null;
    const month = digitsAt(s, 5, 2) orelse return null;
    const day = digitsAt(s, 8, 2) orelse return null;
    if (month < 1 or month > 12 or day < 1 or day > daysInMonth(year, month)) return null;
    var secs: i64 = daysFromCivil(year, month, day) * 86_400;
    if (s.len == 10) return .{ .secs = secs, .micros = 0 };

    const sep = s[10];
    if (sep != 'T' and sep != 't' and sep != ' ') return null;
    if (s.len < 16 or s[13] != ':') return null;
    const hour = digitsAt(s, 11, 2) orelse return null;
    const minute = digitsAt(s, 14, 2) orelse return null;
    if (hour > 23 or minute > 59) return null;
    secs += @as(i64, hour) * 3600 + @as(i64, minute) * 60;
    // `HH:MM` is only a naive format, and a naive format has a literal `T`.
    if (s.len == 16) return if (sep == 't') null else .{ .secs = secs, .micros = 0 };

    if (s.len < 19 or s[16] != ':') return null;
    var second = digitsAt(s, 17, 2) orelse return null;
    if (second > 60) return null;
    var micros: u32 = 0;
    if (second == 60) {
        second = 59;
        micros = 1_000_000;
    }
    secs += second;
    var pos: usize = 19;
    if (pos < s.len and s[pos] == '.') {
        pos += 1;
        const start = pos;
        var frac: u32 = 0;
        while (pos < s.len and std.ascii.isDigit(s[pos])) : (pos += 1) {
            if (pos - start < 6) frac = frac * 10 + (s[pos] - '0');
        }
        if (pos == start) return null;
        var scale = pos - start;
        while (scale < 6) : (scale += 1) frac *= 10;
        micros += frac;
    }
    if (pos == s.len) return if (sep == 't') null else .{ .secs = secs, .micros = micros };

    const rest = s[pos..];
    if (rest.len == 1 and (rest[0] == 'Z' or rest[0] == 'z')) return .{ .secs = secs, .micros = micros };
    if (rest[0] != '+' and rest[0] != '-') return null;
    const oh = digitsAt(rest, 1, 2) orelse return null;
    const om: u32 = switch (rest.len) {
        3 => if (sep == 't') return null else 0,
        5 => digitsAt(rest, 3, 2) orelse return null,
        6 => if (rest[3] != ':') return null else digitsAt(rest, 4, 2) orelse return null,
        else => return null,
    };
    if (oh > 23 or om > 59) return null;
    const offset: i64 = @as(i64, oh) * 3600 + @as(i64, om) * 60;
    return .{ .secs = if (rest[0] == '+') secs - offset else secs + offset, .micros = micros };
}

fn digitsAt(s: []const u8, at: usize, n: usize) ?u32 {
    if (at + n > s.len) return null;
    var v: u32 = 0;
    for (s[at..][0..n]) |c| {
        if (!std.ascii.isDigit(c)) return null;
        v = v * 10 + (c - '0');
    }
    return v;
}

// =========================================================================
// JSON paths
// =========================================================================

/// One step of a JSON path after its first key: Qdrant's `JsonPathItem`
/// (lib/segment/src/json_path/mod.rs:26), `.key`, `[3]` and `[]`. A key
/// borrows the path's text.
pub const PathStep = union(enum) {
    key: []const u8,
    index: usize,
    wildcard,

    fn eql(a: PathStep, b: PathStep) bool {
        return switch (a) {
            .key => |k| b == .key and std.mem.eql(u8, k, b.key),
            .index => |i| b == .index and i == b.index,
            .wildcard => b == .wildcard,
        };
    }
};

/// Steps per filter, over all its conditions, and per indexed field. A
/// bound so that a decoded filter stays a fixed-size value (§6.3); Qdrant
/// has none, and a path past it is refused by name.
pub const max_path_steps = 32;

pub const PathError = error{ InvalidPath, PathTooLong };

pub const ParsedPath = struct {
    /// The top-level key, unquoted.
    first: []const u8,
    /// Steps written to the caller's buffer.
    steps: usize,
};

/// Parse `raw` as Qdrant's `json_path` does (lib/segment/src/json_path/
/// parse.rs:28): a first key, then any run of `.key`, `[digits]` and `[]`,
/// the whole string consumed. A key is either raw, one or more alphanumerics,
/// `_` or `-` (`raw_str`), or quoted, `"` then anything but `"` and `\` then
/// `"` (`quoted_str`, which has no escapes and admits an empty key). An index
/// is a `usize`, leading zeros allowed. So `""` and `a."b.c"[007][]` parse;
/// ``, `a.`, `[0]`, `a[x]`, `a b`, `a..b` and `"a\"b"` do not.
///
/// Where this differs: Rust's `is_alphanumeric` is Unicode's, and without
/// its tables every non-ASCII byte is taken as part of a raw key, so a
/// non-ASCII symbol (`a→b`) parses here where Qdrant refuses it.
pub fn parsePath(raw: []const u8, out: []PathStep) PathError!ParsedPath {
    var pos: usize = 0;
    const first = pathKey(raw, &pos) orelse return error.InvalidPath;
    var n: usize = 0;
    while (pos < raw.len) {
        const step: PathStep = switch (raw[pos]) {
            '.' => blk: {
                pos += 1;
                break :blk .{ .key = pathKey(raw, &pos) orelse return error.InvalidPath };
            },
            '[' => blk: {
                pos += 1;
                const start = pos;
                while (pos < raw.len and std.ascii.isDigit(raw[pos])) pos += 1;
                if (pos == raw.len or raw[pos] != ']') return error.InvalidPath;
                const digits = raw[start..pos];
                pos += 1;
                if (digits.len == 0) break :blk .wildcard;
                // `str::parse::<usize>` refuses an overflow, and so the path.
                break :blk .{ .index = std.fmt.parseInt(usize, digits, 10) catch return error.InvalidPath };
            },
            else => return error.InvalidPath,
        };
        if (n == out.len) return error.PathTooLong;
        out[n] = step;
        n += 1;
    }
    return .{ .first = first, .steps = n };
}

fn pathKey(raw: []const u8, pos: *usize) ?[]const u8 {
    const start = pos.*;
    if (start < raw.len and raw[start] == '"') {
        var i = start + 1;
        while (i < raw.len and raw[i] != '"' and raw[i] != '\\') i += 1;
        if (i == raw.len or raw[i] != '"') return null;
        pos.* = i + 1;
        return raw[start + 1 .. i];
    }
    var i = start;
    while (i < raw.len and (std.ascii.isAlphanumeric(raw[i]) or raw[i] == '_' or raw[i] == '-' or raw[i] >= 0x80)) i += 1;
    if (i == start) return null;
    pos.* = i;
    return raw[start..i];
}

/// Call `visitor.visit(value)` on every `Value` the path `first` then
/// `rest` reaches in a framed blob, as Qdrant's `value_get`
/// (lib/segment/src/json_path/mod.rs:320) collects them: a key step reads a
/// struct's field, an index step one list element and `[]` every element;
/// a step that meets any other kind of value reaches nothing, so `a.b`
/// through a list reaches nothing without `a[].b`. A list at the end of the
/// path is one value, which the conditions then descend into. Stops when
/// `visit` returns true, and returns whether it did.
///
/// No allocation: the recursion is one frame per step.
fn visitPath(blob: []const u8, first: []const u8, rest: []const PathStep, visitor: anytype) bool {
    var it = BlobIterator.init(blob);
    while (it.next()) |entry| {
        const kv = splitEntry(entry) orelse continue;
        if (!std.mem.eql(u8, kv.key, first)) continue;
        if (visitSteps(kv.value, rest, visitor)) return true;
    }
    return false;
}

fn visitSteps(value: []const u8, rest: []const PathStep, visitor: anytype) bool {
    if (rest.len == 0) return visitor.visit(value);
    const k = valueKind(value) orelse return false;
    var r = wire.Reader.init(k.body);
    switch (rest[0]) {
        .key => |key| {
            if (k.kind != value_struct) return false;
            // `Struct.fields = 1`, map entries.
            while (!r.atEnd()) {
                const t = r.tag() catch return false;
                if (t.field != 1 or t.wire_type != .length_delimited) {
                    r.skip(t.wire_type) catch return false;
                    continue;
                }
                const kv = splitEntry(r.bytes() catch return false) orelse continue;
                if (std.mem.eql(u8, kv.key, key) and visitSteps(kv.value, rest[1..], visitor)) return true;
            }
        },
        .index, .wildcard => {
            if (k.kind != value_list) return false;
            var i: usize = 0;
            // `ListValue.values = 1`.
            while (!r.atEnd()) {
                const t = r.tag() catch return false;
                if (t.field != 1 or t.wire_type != .length_delimited) {
                    r.skip(t.wire_type) catch return false;
                    continue;
                }
                const elem = r.bytes() catch return false;
                switch (rest[0]) {
                    .index => |want| if (i == want) return visitSteps(elem, rest[1..], visitor),
                    else => if (visitSteps(elem, rest[1..], visitor)) return true,
                }
                i += 1;
            }
        },
    }
    return false;
}

/// A condition's steps after its first key, as a span of its filter's.
pub const StepSpan = struct { start: u8 = 0, len: u8 = 0 };

pub const Condition = struct {
    /// The top-level key: the path's first key, unquoted, once the filter
    /// holds it (`Filter.addPath`).
    key: []const u8,
    /// The rest of the path, in `Filter.steps`. Empty for a top-level key,
    /// the common case and the fast one.
    rest: StepSpan = .{},
    match: Match,
    /// A `FieldCondition` carrying a `match` and a `range` or a
    /// `datetime_range` too: Qdrant admits a value any of them admits
    /// (`ValueChecker for FieldCondition`, condition_checker.rs, ORs them).
    /// Only a `range` or a `datetime_range`. Rare enough that no index
    /// answers it; `cheapestMust` passes over it and the blobs decide.
    or_match: ?Match = null,
};

/// Conditions per clause. bfb's W12 sends one; the bound exists so a
/// decoded filter is a fixed-size value on the handler's stack (§6.3: no
/// allocation on the query path) rather than a list.
pub const max_conditions = 16;

/// A decoded `Filter { should = 1; must = 2; must_not = 3; }`, borrowed from
/// the request body it was decoded from.
///
/// Semantics are Qdrant's: every `must` holds, no `must_not` holds, and when
/// `should` is non-empty at least one of them holds.
pub const Filter = struct {
    must: [max_conditions]Condition = undefined,
    must_len: usize = 0,
    must_not: [max_conditions]Condition = undefined,
    must_not_len: usize = 0,
    should: [max_conditions]Condition = undefined,
    should_len: usize = 0,
    /// Every condition's path steps, each condition's a `StepSpan` of them.
    /// Indexed rather than sliced so the filter stays a value it is safe to
    /// copy.
    steps: [max_path_steps]PathStep = undefined,
    steps_len: usize = 0,

    pub const Clause = enum { must, must_not, should };

    pub fn add(self: *Filter, clause: Clause, c: Condition) error{TooManyConditions}!void {
        const arr, const len = switch (clause) {
            .must => .{ &self.must, &self.must_len },
            .must_not => .{ &self.must_not, &self.must_not_len },
            .should => .{ &self.should, &self.should_len },
        };
        if (len.* >= max_conditions) return error.TooManyConditions;
        arr[len.*] = c;
        len.* += 1;
    }

    /// `add`, with `c.key` read as Qdrant reads a condition key: a JSON
    /// path (`json_path_from_proto`, lib/api/src/conversions/json.rs:44),
    /// parsed once here so that no point re-parses it. The steps borrow the
    /// key's bytes.
    pub fn addPath(self: *Filter, clause: Clause, c: Condition) (error{TooManyConditions} || PathError)!void {
        const p = try parsePath(c.key, self.steps[self.steps_len..]);
        var keyed = c;
        keyed.key = p.first;
        keyed.rest = .{ .start = @intCast(self.steps_len), .len = @intCast(p.steps) };
        try self.add(clause, keyed);
        self.steps_len += p.steps;
    }

    /// `c`'s steps after its first key.
    pub fn restOf(self: *const Filter, c: Condition) []const PathStep {
        return self.steps[c.rest.start..][0..c.rest.len];
    }

    pub fn isEmpty(self: *const Filter) bool {
        return self.must_len == 0 and self.must_not_len == 0 and self.should_len == 0;
    }

    pub fn musts(self: *const Filter) []const Condition {
        return self.must[0..self.must_len];
    }
    pub fn mustNots(self: *const Filter) []const Condition {
        return self.must_not[0..self.must_not_len];
    }
    pub fn shoulds(self: *const Filter) []const Condition {
        return self.should[0..self.should_len];
    }

    /// Whether a framed blob satisfies the filter.
    pub fn matchesBlob(self: *const Filter, blob: []const u8) bool {
        for (self.musts()) |c| if (!conditionMatches(blob, c, self.restOf(c))) return false;
        for (self.mustNots()) |c| if (conditionMatches(blob, c, self.restOf(c))) return false;
        if (self.should_len > 0) {
            for (self.shoulds()) |c| if (conditionMatches(blob, c, self.restOf(c))) return true;
            return false;
        }
        return true;
    }
};

/// Whether any value the condition's path reaches in the blob matches
/// `c.match`, as `check_field_condition` (query_checker.rs:203) ORs
/// `FieldCondition::check` over `get_value`'s values.
///
/// A path that reaches no value matches only what `missingMatches` says,
/// which for a value match is nothing: that is what makes `must_not` on an
/// absent key true and `must` on one false, as in Qdrant.
fn conditionMatches(blob: []const u8, c: Condition, rest: []const PathStep) bool {
    if (c.match == .all_empty) {
        // Every value empty: no value is not.
        var v = ConditionVisitor{ .match = .{ .is_empty = false } };
        return !visitPath(blob, c.key, rest, &v);
    }
    var v = ConditionVisitor{ .match = c.match, .or_match = c.or_match };
    if (visitPath(blob, c.key, rest, &v)) return true;
    return !v.present and missingMatches(c.match);
}

const ConditionVisitor = struct {
    match: Match,
    or_match: ?Match = null,
    present: bool = false,

    fn visit(self: *ConditionVisitor, value: []const u8) bool {
        self.present = true;
        if (valueMatches(value, self.match)) return true;
        if (self.or_match) |m| if (valueMatches(value, m)) return true;
        return false;
    }
};

/// A missing field, as Qdrant's `FieldCondition::check_empty`
/// (condition_checker.rs) answers it: zero values, empty, and not null.
fn missingMatches(m: Match) bool {
    return switch (m) {
        .values_count => |vc| vc.contains(0),
        .is_empty => |want| want,
        .is_null => |want| !want,
        else => false,
    };
}

/// Whether one `Value` matches, descending one level into a `ListValue` so a
/// multi-valued keyword field (`["a", "b"]`) matches either. The shape
/// conditions look at the whole value instead: Qdrant's
/// `FieldCondition::check` tests them on the value before (and instead of)
/// the per-element `_check`.
fn valueMatches(value: []const u8, m: Match) bool {
    switch (m) {
        .values_count, .is_empty, .is_null => {
            const s = valueShape(value) orelse return false;
            return switch (m) {
                .values_count => |vc| vc.contains(s.count),
                .is_empty => |want| s.empty == want,
                .is_null => |want| s.has_null == want,
                else => unreachable,
            };
        },
        else => return valueMatchesAt(value, m, true),
    }
}

/// What the shape conditions read from a `Value`, as Qdrant's
/// `check_count_from` (types.rs), `check_is_empty` and `check_is_null`
/// (condition_checker.rs) read the JSON it becomes: null counts 0 values
/// and is empty and null; a list counts its elements, is empty when it has
/// none and null when one of them is; anything else (a struct included)
/// counts 1 and is neither. So `[null]` is null but not empty.
const Shape = struct { count: usize, empty: bool, has_null: bool };

/// Null on malformed bytes, which then match no shape condition, as a
/// malformed value matches no value condition in `valueMatchesAt`.
fn valueShape(value: []const u8) ?Shape {
    const k = valueKind(value) orelse return null;
    switch (k.kind) {
        0, value_null => return .{ .count = 0, .empty = true, .has_null = true },
        value_list => {
            var n: usize = 0;
            var has_null = false;
            var r = wire.Reader.init(k.body);
            while (!r.atEnd()) {
                const t = r.tag() catch return null;
                if (t.field != 1 or t.wire_type != .length_delimited) {
                    r.skip(t.wire_type) catch return null;
                    continue;
                }
                const elem = r.bytes() catch return null;
                n += 1;
                if (!has_null) {
                    const ek = valueKind(elem) orelse return null;
                    has_null = ek.kind == 0 or ek.kind == value_null;
                }
            }
            return .{ .count = n, .empty = n == 0, .has_null = has_null };
        },
        else => return .{ .count = 1, .empty = false, .has_null = false },
    }
}

/// A `Value`'s `kind`: the last oneof member on the wire, protobuf's rule
/// for a oneof set more than once, with the bytes of a length-delimited
/// one. Kind 0 is none set, which Qdrant reads as JSON null (`proto_to_json`,
/// lib/api/src/conversions/json.rs: no `kind` is `Value::default()`).
fn valueKind(value: []const u8) ?struct { kind: u32, body: []const u8 } {
    var r = wire.Reader.init(value);
    var kind: u32 = 0;
    var body: []const u8 = &.{};
    while (!r.atEnd()) {
        const t = r.tag() catch return null;
        if (t.field < value_null or t.field > value_list) {
            r.skip(t.wire_type) catch return null;
            continue;
        }
        kind = t.field;
        if (t.wire_type == .length_delimited) {
            body = r.bytes() catch return null;
        } else {
            body = &.{};
            r.skip(t.wire_type) catch return null;
        }
    }
    return .{ .kind = kind, .body = body };
}

/// One walk over a `Value`'s scalar arms, with the list descent as a flag
/// rather than a second copy of the arms: a list of lists is not a keyword
/// field, so the descent stops after one level.
fn valueMatchesAt(value: []const u8, m: Match, descend: bool) bool {
    var r = wire.Reader.init(value);
    while (!r.atEnd()) {
        const t = r.tag() catch return false;
        switch (t.field) {
            value_string => {
                const s = r.bytes() catch return false;
                switch (m) {
                    .keyword => |k| if (std.mem.eql(u8, s, k)) return true,
                    .keywords => |list| if (repeatedStringsContain(list, s)) return true,
                    // `Range<DateTimePayloadType>::check_match`: a string
                    // that does not parse is no match.
                    .datetime => |dr| if (parseDatetime(s)) |v| if (dr.contains(v)) return true,
                    else => {},
                }
            },
            value_integer => {
                const v: i64 = @bitCast(r.varint() catch return false);
                switch (m) {
                    .integer => |i| if (v == i) return true,
                    .integers => |list| if (repeatedIntegersContain(list, v)) return true,
                    // `as_f64`, as Qdrant compares an integer to a range.
                    .range => |rg| if (rg.contains(@floatFromInt(v))) return true,
                    else => {},
                }
            },
            value_double => {
                if (t.wire_type != .fixed64) {
                    r.skip(t.wire_type) catch return false;
                    continue;
                }
                const v = r.double() catch return false;
                switch (m) {
                    .range => |rg| if (rg.contains(v)) return true,
                    else => {},
                }
            },
            value_bool => {
                const b = r.boolean() catch return false;
                switch (m) {
                    .boolean => |want| if (b == want) return true,
                    else => {},
                }
            },
            value_list => {
                var list = r.nested() catch return false;
                while (!list.atEnd()) {
                    const lt = list.tag() catch return false;
                    if (lt.field != 1) {
                        list.skip(lt.wire_type) catch return false;
                        continue;
                    }
                    const inner = list.bytes() catch return false;
                    if (descend and valueMatchesAt(inner, m, false)) return true;
                }
            },
            else => r.skip(t.wire_type) catch return false,
        }
    }
    return false;
}

fn repeatedStringsContain(list: []const u8, s: []const u8) bool {
    var r = wire.Reader.init(list);
    while (!r.atEnd()) {
        const t = r.tag() catch return false;
        if (t.field == 1 and t.wire_type == .length_delimited) {
            const x = r.bytes() catch return false;
            if (std.mem.eql(u8, x, s)) return true;
        } else r.skip(t.wire_type) catch return false;
    }
    return false;
}

/// `RepeatedIntegers { repeated int64 integers = 1; }`.
///
/// The field is checked *before* the payload is consumed, as
/// `repeatedStringsContain` and `PostingLists` do. Reading the nested blob
/// first and testing the field inside it meant a length-delimited field under
/// some other number was parsed as a run of varints, so malformed bytes in a
/// field this has no interest in abandoned the whole condition (`catch return
/// false`) instead of being skipped. Four walks over this message shape exist
/// in this file and two of them did it the other way round; they now agree.
fn repeatedIntegersContain(list: []const u8, v: i64) bool {
    var r = wire.Reader.init(list);
    while (!r.atEnd()) {
        const t = r.tag() catch return false;
        if (t.field != 1) {
            r.skip(t.wire_type) catch return false;
            continue;
        }
        switch (t.wire_type) {
            .varint => {
                const x: i64 = @bitCast(r.varint() catch return false);
                if (x == v) return true;
            },
            // `repeated int64` is packed in proto3.
            .length_delimited => {
                var packed_r = r.nested() catch return false;
                while (!packed_r.atEnd()) {
                    const x: i64 = @bitCast(packed_r.varint() catch return false);
                    if (x == v) return true;
                }
            },
            else => r.skip(t.wire_type) catch return false,
        }
    }
    return false;
}

// =========================================================================
// Blob framing
// =========================================================================

/// Walks the framed entries of one point's blob.
pub const BlobIterator = struct {
    buf: []const u8,
    pos: usize = 0,

    pub fn init(blob: []const u8) BlobIterator {
        return .{ .buf = blob };
    }

    pub fn next(self: *BlobIterator) ?[]const u8 {
        if (self.pos >= self.buf.len) return null;
        var r = wire.Reader.init(self.buf[self.pos..]);
        const len = r.varint() catch return null;
        const start = self.pos + r.pos;
        const end = start + @as(usize, @intCast(len));
        if (end > self.buf.len) return null;
        self.pos = end;
        return self.buf[start..end];
    }
};

/// Walks the map entries of a `map<string, Value>` field in a request body:
/// every `field`-numbered length-delimited value in `bytes`, in order. The
/// bytes may hold other fields between entries; they are skipped.
pub const WireEntries = struct {
    r: wire.Reader,
    field: u32,

    pub fn init(bytes: []const u8, field: u32) WireEntries {
        return .{ .r = wire.Reader.init(bytes), .field = field };
    }

    pub fn next(self: *WireEntries) wire.Error!?[]const u8 {
        while (!self.r.atEnd()) {
            const t = try self.r.tag();
            if (t.field == self.field and t.wire_type == .length_delimited) return try self.r.bytes();
            try self.r.skip(t.wire_type);
        }
        return null;
    }
};

const Entry = struct { key: []const u8, value: []const u8 };

/// The key and the `Value` bytes of one map entry, or null when it has no key.
pub fn splitEntry(entry: []const u8) ?Entry {
    var r = wire.Reader.init(entry);
    var key: ?[]const u8 = null;
    var value: []const u8 = &.{};
    while (!r.atEnd()) {
        const t = r.tag() catch return null;
        switch (t.field) {
            entry_key => key = r.bytes() catch return null,
            entry_value => value = r.bytes() catch return null,
            else => r.skip(t.wire_type) catch return null,
        }
    }
    return .{ .key = key orelse return null, .value = value };
}

fn varintLen(v: u64) usize {
    var n: usize = 1;
    var x = v >> 7;
    while (x != 0) : (x >>= 7) n += 1;
    return n;
}

fn writeVarint(buf: []u8, v: u64) usize {
    var x = v;
    var i: usize = 0;
    while (x >= 0x80) : (x >>= 7) {
        buf[i] = @as(u8, @truncate(x & 0x7f)) | 0x80;
        i += 1;
    }
    buf[i] = @truncate(x);
    return i + 1;
}

/// Bytes the framed form of `entry` occupies.
pub fn framedLen(entry: []const u8) usize {
    return varintLen(entry.len) + entry.len;
}

/// Append `entry` in framed form.
pub fn frameInto(out: *std.ArrayList(u8), alloc: std.mem.Allocator, entry: []const u8) !void {
    var lenbuf: [10]u8 = undefined;
    const n = writeVarint(&lenbuf, entry.len);
    try out.appendSlice(alloc, lenbuf[0..n]);
    try out.appendSlice(alloc, entry);
}

// =========================================================================
// The index
// =========================================================================

/// `FieldType` values this engine indexes (`FieldTypeKeyword = 0`,
/// `FieldTypeInteger = 1`, `FieldTypeFloat = 2`, `FieldTypeText = 4`,
/// `FieldTypeDatetime = 6`, VERIFIED points.proto:314). Stored as the
/// `PayloadSchemaType` the collection info reports them as (`Keyword = 1`,
/// `Integer = 2`, `Float = 3`, `Text = 5`, `Datetime = 7`,
/// collections.proto:184), which is a different enum with different numbers.
/// Persisted (payload.schema), so a value never changes meaning.
pub const Kind = enum(u8) {
    keyword = 1,
    integer = 2,
    /// A sorted `(value, offset)` array (`FloatIndex`), answering `range`.
    float = 3,
    /// A text index scored with BM25 (`FieldTypeText = 4`, `Text = 5`), the
    /// only text index this engine builds (decisions.md, 2026-10-07).
    text = 5,
    /// A sorted `(microseconds, offset)` array (`DatetimeIndex`), answering
    /// `datetime_range`.
    datetime = 7,

    pub fn fromFieldType(v: u64) ?Kind {
        return switch (v) {
            0 => .keyword,
            1 => .integer,
            2 => .float,
            4 => .text,
            6 => .datetime,
            else => null,
        };
    }

    /// `PayloadSchemaType`.
    pub fn schemaType(self: Kind) u32 {
        return @backingInt(self);
    }
};

/// One value's points: the offsets in write order and, once the value is
/// dense enough that a bitset over the store's capacity is no larger than the
/// list, the same points as that bitset. `select` then ORs words in rather
/// than visiting every point: W15-filtered's half-the-corpus `parity` filter
/// visited 261k points per query on Quora to admit a dozen (2026-10-13).
const Postings = struct {
    list: std.ArrayList(u32) = .empty,
    /// Empty until the value is dense, then `capacity / 64` words.
    dense: []u64 = &.{},

    fn deinit(self: *Postings, alloc: std.mem.Allocator) void {
        self.list.deinit(alloc);
        alloc.free(self.dense);
    }

    fn append(self: *Postings, alloc: std.mem.Allocator, offset: u32, capacity: usize) !void {
        try self.list.append(alloc, offset);
        if (self.dense.len != 0) {
            self.dense[offset / 64] |= @as(u64, 1) << @intCast(offset % 64);
            return;
        }
        // `capacity / 8` bytes against the list's 4 per entry.
        if (self.list.items.len * 32 < capacity) return;
        const dense = try alloc.alloc(u64, (capacity + 63) / 64);
        @memset(dense, 0);
        for (self.list.items) |o| dense[o / 64] |= @as(u64, 1) << @intCast(o % 64);
        self.dense = dense;
    }
};

/// A float field's index: every numeric value (doubles, and integers as
/// `f64`) with its point's offset, sorted by value, plus an unsorted tail of
/// the writes since the last merge. A range is a binary search into the
/// sorted part and a scan of the tail.
///
/// Entries are hints like the postings (the module doc): an overwrite leaves
/// the old value's entry, and `select` verifies what it admits by the same
/// rules. A NaN value is not stored, since no range admits it.
const FloatIndex = SortedIndex(f64, Range);
const FloatEntry = FloatIndex.Item;

/// A datetime field's index: `FloatIndex`'s design over each parsed instant
/// as `i64` microseconds since the epoch, Qdrant's own key for it
/// (`NumericIndex<IntPayloadType, DateTimePayloadType>`, `into_inner_value`
/// is `timestamp_micros`, lib/segment/src/index/field_index/numeric_index/
/// value_indexer.rs:87).
///
/// One wrinkle: a leap second's instant sorts before the next second in
/// chrono's order but its microsecond count does not, so the key clamps the
/// sub-second part to 999_999 (`microsKey`). The key is then monotone, never
/// strictly, in the instant: a range over keys admits a superset, exact
/// unless a leap second is stored or bounds the range (`MicrosRange`), and
/// `select` trusts it only when exact.
const DatetimeIndex = struct {
    sorted: SortedIndex(i64, MicrosRange),
    /// A stored instant fell in a leap second, so two instants may share a
    /// key.
    leap: bool = false,
};

fn microsKey(i: Instant) i64 {
    return i.secs * 1_000_000 + @min(i.micros, 999_999);
}

/// A `DatetimeRange` over `microsKey`s. Every bound a request can give fits
/// in an `i64` of microseconds (chrono's years, `timestamp_min_secs`), and the
/// absent ends map to the type's.
const MicrosRange = struct {
    lo: i64,
    lo_strict: bool,
    hi: i64,
    hi_strict: bool,
    /// Admits exactly the keys of the instants `DatetimeRange` admits, when
    /// no stored instant is a leap second. A leap-second bound is made
    /// inclusive instead, a superset.
    exact: bool,

    fn of(r: DatetimeRange) MicrosRange {
        const lo_leap = r.lo.secs != Instant.min.secs and r.lo.micros >= 1_000_000;
        const hi_leap = r.hi.secs != Instant.max.secs and r.hi.micros >= 1_000_000;
        return .{
            .lo = if (r.lo.secs == Instant.min.secs) std.math.minInt(i64) else microsKey(r.lo),
            .lo_strict = r.lo_strict and !lo_leap,
            .hi = if (r.hi.secs == Instant.max.secs) std.math.maxInt(i64) else microsKey(r.hi),
            .hi_strict = r.hi_strict and !hi_leap,
            .exact = !lo_leap and !hi_leap,
        };
    }

    fn aboveLo(self: MicrosRange, v: i64) bool {
        return v > self.lo or (v == self.lo and !self.lo_strict);
    }

    fn belowHi(self: MicrosRange, v: i64) bool {
        return v < self.hi or (v == self.hi and !self.hi_strict);
    }

    fn contains(self: MicrosRange, v: i64) bool {
        return self.aboveLo(v) and self.belowHi(v);
    }
};

/// `(value, offset)` entries sorted by value, plus an unsorted tail of the
/// writes since the last merge, answering a range `R` (one with `aboveLo`,
/// monotone, `belowHi`, antitone, and `contains`).
fn SortedIndex(comptime V: type, comptime R: type) type {
    return struct {
        const Self = @This();

        sorted: std.ArrayList(Item) = .empty,
        tail: std.ArrayList(Item) = .empty,

        pub const Item = struct {
            v: V,
            off: u32,

            fn lessThan(_: void, a: Item, b: Item) bool {
                return a.v < b.v;
            }
        };

        /// The tail is merged once it reaches this many entries. Every range
        /// query scans the whole tail and every merge moves the whole sorted
        /// part, so the bound trades one against the other: a fixed bound
        /// keeps the per-query scan a constant, the price of the binary
        /// search's neighbourhood rather than of the corpus, and costs one
        /// pass over the sorted part per `tail_max` writes. Proportional to
        /// the sorted part it would make the merges amortized constant but
        /// the scan grow with the corpus, and the scan is on the query path.
        const tail_max = 4096;

        fn deinit(self: *Self, alloc: std.mem.Allocator) void {
            self.sorted.deinit(alloc);
            self.tail.deinit(alloc);
        }

        /// Under the index's exclusive lock, which also covers the merge.
        fn add(self: *Self, alloc: std.mem.Allocator, v: V, offset: u32) !void {
            try self.tail.append(alloc, .{ .v = v, .off = offset });
            if (self.tail.items.len >= tail_max) try self.merge(alloc);
        }

        /// Sort the tail and merge it into the sorted part, back to front and
        /// in place. A failure to grow leaves both parts as they were, which
        /// still answers correctly, with a longer tail.
        fn merge(self: *Self, alloc: std.mem.Allocator) !void {
            const t = self.tail.items;
            try self.sorted.ensureUnusedCapacity(alloc, t.len);
            std.mem.sort(Item, t, {}, Item.lessThan);
            var i = self.sorted.items.len;
            var j = t.len;
            self.sorted.items.len += t.len;
            const out = self.sorted.items;
            var k = out.len;
            while (j > 0) {
                k -= 1;
                if (i > 0 and out[i - 1].v > t[j - 1].v) {
                    i -= 1;
                    out[k] = out[i];
                } else {
                    j -= 1;
                    out[k] = t[j];
                }
            }
            self.tail.clearRetainingCapacity();
        }

        /// The sorted entries `r` admits: a contiguous run, since `aboveLo`
        /// and `belowHi` are monotone in the value.
        fn span(self: *const Self, r: R) []const Item {
            const items = self.sorted.items;
            const lo = std.sort.partitionPoint(Item, items, r, belowLo);
            const hi = std.sort.partitionPoint(Item, items, r, withinHi);
            return items[lo..@max(lo, hi)];
        }

        fn belowLo(r: R, e: Item) bool {
            return !r.aboveLo(e.v);
        }

        fn withinHi(r: R, e: Item) bool {
            return r.belowHi(e.v);
        }

        /// Entries `r` admits, for the planner. Over-counts as `points`
        /// does: a point with several values in range, or an overwritten
        /// one, counts once per entry.
        fn count(self: *const Self, r: R) usize {
            var n = self.span(r).len;
            for (self.tail.items) |e| {
                if (r.contains(e.v)) n += 1;
            }
            return n;
        }

        /// Set the bit of every entry `r` admits, through `admit`.
        fn admitRange(self: *const Self, r: R, store: *const Store, filter: *const Filter, bits: []u64, bound: usize, trusted: bool) void {
            for (self.span(r)) |e| store.admit(filter, bits, bound, trusted, e.off);
            for (self.tail.items) |e| {
                if (r.contains(e.v)) store.admit(filter, bits, bound, trusted, e.off);
            }
        }
    };
}

pub const Field = struct {
    /// The path as `CreateFieldIndex` named it; `first` and `steps` borrow it.
    name: []u8,
    /// The parsed path whose values the index holds: a nested path indexes
    /// what a condition on that path reads, as Qdrant's indexes ingest
    /// `get_value(field)` (lib/segment/src/index/struct_payload_index/
    /// payload_index.rs:168).
    first: []const u8,
    steps: []PathStep,
    index: Index,
    /// Postings written, which over-counts overwritten points the same way
    /// the postings do. Reported as `PayloadSchemaInfo.points`.
    points: usize = 0,
    /// The store's capacity, which a dense posting's bitset spans.
    capacity: usize = 0,

    /// The one index a field holds, by its kind.
    pub const Index = union(Kind) {
        keyword: std.StringHashMapUnmanaged(Postings),
        integer: std.AutoHashMapUnmanaged(i64, Postings),
        float: FloatIndex,
        /// The BM25 index. Unlike the postings it is exact: a rewrite
        /// replaces the point's document and a delete removes it.
        text: *text.TextIndex,
        datetime: DatetimeIndex,
    };

    pub fn kind(self: *const Field) Kind {
        return self.index;
    }

    /// Whether this field indexes the path `first` then `rest`: Qdrant
    /// looks a condition's index up by its whole path
    /// (`field_indexes.get(&field_condition.key)`, query_checker.rs:213), so
    /// an index on `a` answers nothing about `a[]` or `a.b`.
    pub fn hasPath(self: *const Field, first: []const u8, rest: []const PathStep) bool {
        if (!std.mem.eql(u8, self.first, first) or self.steps.len != rest.len) return false;
        for (self.steps, rest) |a, b| if (!a.eql(b)) return false;
        return true;
    }

    /// A field for the path `name`, owning a copy of it, with `index`.
    fn init(alloc: std.mem.Allocator, name: []const u8, index: Index, capacity: usize) !Field {
        const owned = try alloc.dupe(u8, name);
        errdefer alloc.free(owned);
        var buf: [max_path_steps]PathStep = undefined;
        const p = try parsePath(owned, &buf);
        const steps = try alloc.dupe(PathStep, buf[0..p.steps]);
        return .{ .name = owned, .first = p.first, .steps = steps, .index = index, .capacity = capacity };
    }

    fn deinit(self: *Field, alloc: std.mem.Allocator) void {
        switch (self.index) {
            .keyword => |*m| {
                var ki = m.iterator();
                while (ki.next()) |e| {
                    alloc.free(e.key_ptr.*);
                    e.value_ptr.deinit(alloc);
                }
                m.deinit(alloc);
            },
            .integer => |*m| {
                var ii = m.iterator();
                while (ii.next()) |e| e.value_ptr.deinit(alloc);
                m.deinit(alloc);
            },
            .float => |*fi| fi.deinit(alloc),
            .datetime => |*di| di.sorted.deinit(alloc),
            .text => |ti| {
                ti.deinit();
                alloc.destroy(ti);
            },
        }
        alloc.free(self.steps);
        alloc.free(self.name);
    }

    /// One value's strings (a string, or a list's strings), appended to the
    /// text index's document.
    fn collectStrings(alloc: std.mem.Allocator, values: *std.ArrayList([]const u8), value: []const u8) IndexError!void {
        var r = wire.Reader.init(value);
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                value_string => try values.append(alloc, try r.bytes()),
                value_list => {
                    var list = try r.nested();
                    while (!list.atEnd()) {
                        const lt = try list.tag();
                        if (lt.field != 1) {
                            try list.skip(lt.wire_type);
                            continue;
                        }
                        var inner = wire.Reader.init(try list.bytes());
                        while (!inner.atEnd()) {
                            const it = try inner.tag();
                            if (it.field == value_string) {
                                try values.append(alloc, try inner.bytes());
                            } else try inner.skip(it.wire_type);
                        }
                    }
                },
                else => try r.skip(t.wire_type),
            }
        }
    }

    const IndexError = wire.Error || std.mem.Allocator.Error;

    /// Index one value the field's path reaches.
    fn add(self: *Field, alloc: std.mem.Allocator, offset: u32, value: []const u8) IndexError!void {
        var r = wire.Reader.init(value);
        while (!r.atEnd()) {
            const t = try r.tag();
            switch (t.field) {
                value_string => try self.addString(alloc, offset, try r.bytes()),
                value_integer => {
                    const v: i64 = @bitCast(try r.varint());
                    try self.addNumber(alloc, offset, v);
                },
                value_double => try self.addDouble(alloc, offset, &r, t.wire_type),
                value_list => {
                    var list = try r.nested();
                    while (!list.atEnd()) {
                        const lt = try list.tag();
                        if (lt.field != 1) {
                            try list.skip(lt.wire_type);
                            continue;
                        }
                        // One level, like `valueMatches`.
                        var inner = wire.Reader.init(try list.bytes());
                        while (!inner.atEnd()) {
                            const it = try inner.tag();
                            switch (it.field) {
                                value_string => try self.addString(alloc, offset, try inner.bytes()),
                                value_integer => {
                                    const v: i64 = @bitCast(try inner.varint());
                                    try self.addNumber(alloc, offset, v);
                                },
                                value_double => try self.addDouble(alloc, offset, &inner, it.wire_type),
                                else => try inner.skip(it.wire_type),
                            }
                        }
                    }
                },
                else => try r.skip(t.wire_type),
            }
        }
    }

    /// Every value this field's path reaches in a framed blob. A text
    /// field's document is replaced, or removed when the path reaches none.
    fn addBlob(self: *Field, alloc: std.mem.Allocator, offset: u32, blob: []const u8) !void {
        var strings: std.ArrayList([]const u8) = .empty;
        defer strings.deinit(alloc);
        var v = IndexVisitor{ .f = self, .alloc = alloc, .offset = offset, .strings = if (self.index == .text) &strings else null };
        _ = visitPath(blob, self.first, self.steps, &v);
        if (v.err) |e| return e;
        if (self.index != .text) return;
        if (v.present) return self.index.text.set(offset, strings.items);
        self.index.text.remove(offset);
    }

    const IndexVisitor = struct {
        f: *Field,
        alloc: std.mem.Allocator,
        offset: u32,
        /// A text field's document, gathered over every value.
        strings: ?*std.ArrayList([]const u8),
        present: bool = false,
        err: ?IndexError = null,

        fn visit(self: *IndexVisitor, value: []const u8) bool {
            self.present = true;
            const res = if (self.strings) |l|
                collectStrings(self.alloc, l, value)
            else
                self.f.add(self.alloc, self.offset, value);
            res catch |e| {
                self.err = e;
                return true;
            };
            return false;
        }
    };

    /// A string: a keyword's posting, or a datetime field's instant when it
    /// parses as `datetime_range` parses it (Qdrant's datetime index takes
    /// what `DateTimePayloadType::from_str` takes, value_indexer.rs:75).
    fn addString(self: *Field, alloc: std.mem.Allocator, offset: u32, s: []const u8) IndexError!void {
        switch (self.index) {
            .keyword => try self.addKeyword(alloc, offset, s),
            .datetime => |*di| {
                const at = parseDatetime(s) orelse return;
                try di.sorted.add(alloc, microsKey(at), offset);
                if (at.micros >= 1_000_000) di.leap = true;
                self.points += 1;
            },
            else => {},
        }
    }

    fn addKeyword(self: *Field, alloc: std.mem.Allocator, offset: u32, s: []const u8) !void {
        const gop = try self.index.keyword.getOrPut(alloc, s);
        if (!gop.found_existing) {
            // The map borrows its key: own it, and repoint the entry at the
            // copy before anything can look it up.
            const owned = try alloc.dupe(u8, s);
            gop.key_ptr.* = owned;
            gop.value_ptr.* = .{};
        }
        try gop.value_ptr.append(alloc, offset, self.capacity);
        self.points += 1;
    }

    /// An integer value: a posting on an integer field, an `f64` entry on a
    /// float field (Qdrant's float index takes integers too, and a range
    /// compares them as `f64`).
    fn addNumber(self: *Field, alloc: std.mem.Allocator, offset: u32, v: i64) !void {
        switch (self.index) {
            .integer => try self.addInteger(alloc, offset, v),
            .float => try self.addFloat(alloc, offset, @floatFromInt(v)),
            else => {},
        }
    }

    /// A `double_value`, which only a float field indexes. Read the way
    /// `valueMatchesAt` reads it, so the index holds what a range admits.
    fn addDouble(self: *Field, alloc: std.mem.Allocator, offset: u32, r: *wire.Reader, wt: wire.WireType) !void {
        if (wt != .fixed64) return r.skip(wt);
        const v = try r.double();
        if (self.index == .float) try self.addFloat(alloc, offset, v);
    }

    fn addFloat(self: *Field, alloc: std.mem.Allocator, offset: u32, v: f64) IndexError!void {
        if (std.math.isNan(v)) return;
        try self.index.float.add(alloc, v, offset);
        self.points += 1;
    }

    fn addInteger(self: *Field, alloc: std.mem.Allocator, offset: u32, v: i64) !void {
        const gop = try self.index.integer.getOrPut(alloc, v);
        if (!gop.found_existing) gop.value_ptr.* = .{};
        try gop.value_ptr.append(alloc, offset, self.capacity);
        self.points += 1;
    }
};

// =========================================================================
// The store
// =========================================================================

/// A slot packs the blob's arena offset and length: 40 bits of offset
/// (1 TiB of arena) and 24 of length (16 MiB per point). Zero is "no
/// payload", so an empty map and an absent one are the same thing, which is
/// what Qdrant returns for both.
const len_bits = 24;
const len_mask: u64 = (1 << len_bits) - 1;
pub const max_blob = len_mask;
/// 16 MiB chunks: two of them hold a million bfb keyword payloads.
pub const chunk_size: usize = 1 << 24;
/// Never moved, so a reader's slice stays valid; bounded so the pointer table
/// is a fixed array the reader indexes without a lock.
pub const max_chunks = 4096;

/// Readers-writer lock over the postings. `std.Io.RwLock` needs an `Io`
/// instance this server does not carry, so this is the project's futex mutex
/// plus a reader count: a reader takes the mutex for one increment, a writer
/// holds it and waits for the count to drain. Readers are one `select` long
/// and writers one posting append, so neither side waits for much.
const RwLock = struct {
    mutex: futex_lock.Mutex = .{},
    readers: std.atomic.Value(u32) = .init(0),

    fn lockShared(self: *RwLock) void {
        self.mutex.lock();
        _ = self.readers.fetchAdd(1, .acquire);
        self.mutex.unlock();
    }

    fn unlockShared(self: *RwLock) void {
        _ = self.readers.fetchSub(1, .release);
    }

    fn lock(self: *RwLock) void {
        self.mutex.lock();
        while (self.readers.load(.acquire) != 0) std.atomic.spinLoopHint();
    }

    fn unlock(self: *RwLock) void {
        self.mutex.unlock();
    }
};

pub const Selection = struct {
    /// Points the filter admits, verified against their blobs.
    count: usize,
};

pub const Store = struct {
    slots: []std.atomic.Value(u64),
    chunks: []std.atomic.Value(?[*]u8),
    chunk_count: usize = 0,
    /// Bytes used in the last chunk.
    used: usize = 0,
    /// Points whose slot is non-zero.
    with_payload: usize = 0,
    fields: std.ArrayList(Field) = .empty,
    index_lock: RwLock = .{},
    /// Whether every posting still names a point whose published blob holds
    /// that value. True until a payload is rewritten (overwrite, merge, clear)
    /// or an index insert fails part-way, since either leaves a posting the
    /// blob no longer backs. While it holds, `select` can admit a
    /// single-condition filter's postings without re-reading the blobs.
    exact_postings: std.atomic.Value(bool) = .init(true),
    /// Bit `offset` set while `slots[offset]` is non-zero, written after the
    /// slot with release ordering, so a reader that sees the bit sees the
    /// blob. It is `has` a word at a time, which a dense posting needs.
    published: []std.atomic.Value(u64),

    pub fn init(alloc: std.mem.Allocator, capacity: usize) !Store {
        const slots = try alloc.alloc(std.atomic.Value(u64), capacity);
        errdefer alloc.free(slots);
        for (slots) |*s| s.* = .init(0);
        const published = try alloc.alloc(std.atomic.Value(u64), (capacity + 63) / 64);
        errdefer alloc.free(published);
        for (published) |*w| w.* = .init(0);
        const chunks = try alloc.alloc(std.atomic.Value(?[*]u8), max_chunks);
        for (chunks) |*c| c.* = .init(null);
        return .{ .slots = slots, .chunks = chunks, .published = published };
    }

    pub fn deinit(self: *Store, alloc: std.mem.Allocator) void {
        for (self.chunks[0..self.chunk_count]) |*c| {
            if (c.load(.monotonic)) |p| {
                const whole: []u8 = p[0..chunk_size];
                alloc.free(whole);
            }
        }
        alloc.free(self.chunks);
        alloc.free(self.slots);
        alloc.free(self.published);
        for (self.fields.items) |*f| f.deinit(alloc);
        self.fields.deinit(alloc);
    }

    /// The framed blob of `offset`, empty when it has none. Lock-free.
    pub fn get(self: *const Store, offset: u32) []const u8 {
        const slot = self.slots[offset].load(.acquire);
        if (slot == 0) return &.{};
        const len: usize = @intCast(slot & len_mask);
        const off: usize = @intCast(slot >> len_bits);
        const chunk = self.chunks[off / chunk_size].load(.acquire) orelse return &.{};
        return chunk[off % chunk_size ..][0..len];
    }

    pub fn has(self: *const Store, offset: u32) bool {
        return self.slots[offset].load(.acquire) != 0;
    }

    /// Whether `offset`'s payload satisfies `filter`. Lock-free.
    pub fn matches(self: *const Store, offset: u32, filter: *const Filter) bool {
        return filter.matchesBlob(self.get(offset));
    }

    /// Replace `offset`'s payload with the map entries in `entries` (wire
    /// bytes holding repeated `field`-numbered entries). Under the
    /// collection's write lock. Empty entries clear the payload, which is what
    /// an upsert without one means.
    pub fn set(self: *Store, alloc: std.mem.Allocator, offset: u32, entries: []const u8, map_field: u32) !void {
        self.noteRewrite(offset);
        var total: usize = 0;
        var it = WireEntries.init(entries, map_field);
        while (try it.next()) |e| total += framedLen(e);
        if (total == 0) {
            self.forgetPoint(offset);
            self.publish(offset, 0);
            return;
        }
        if (total > max_blob) return error.PayloadTooLarge;
        const dst = try self.reserve(alloc, total);
        var pos: usize = 0;
        it = WireEntries.init(entries, map_field);
        while (try it.next()) |e| {
            pos += writeVarint(dst.bytes[pos..], e.len);
            @memcpy(dst.bytes[pos..][0..e.len], e);
            pos += e.len;
        }
        std.debug.assert(pos == total);
        // Index first, publish second: a slot published before its posting
        // left a point the bitset path could not find if the insert then
        // failed, and nothing repaired it.
        try self.indexBlob(alloc, offset, dst.bytes);
        self.publish(offset, (@as(u64, dst.offset) << len_bits) | @as(u64, total));
    }

    /// `SetPayload`: merge the entries in `entries` over the existing payload,
    /// replacing values under the same key and keeping the rest.
    pub fn merge(self: *Store, alloc: std.mem.Allocator, offset: u32, entries: []const u8, map_field: u32) !void {
        var framed: std.ArrayList(u8) = .empty;
        defer framed.deinit(alloc);
        // Existing entries whose key is not being set.
        var old = BlobIterator.init(self.get(offset));
        while (old.next()) |entry| {
            const kv = splitEntry(entry) orelse continue;
            var replaced = false;
            var it = WireEntries.init(entries, map_field);
            while (try it.next()) |e| {
                const nkv = splitEntry(e) orelse continue;
                if (std.mem.eql(u8, nkv.key, kv.key)) {
                    replaced = true;
                    break;
                }
            }
            if (!replaced) try frameInto(&framed, alloc, entry);
        }
        var it = WireEntries.init(entries, map_field);
        while (try it.next()) |e| try frameInto(&framed, alloc, e);
        try self.setFramed(alloc, offset, framed.items);
    }

    /// Replace `offset`'s payload with an already framed blob.
    pub fn setFramed(self: *Store, alloc: std.mem.Allocator, offset: u32, framed: []const u8) !void {
        self.noteRewrite(offset);
        if (framed.len == 0) {
            self.forgetPoint(offset);
            self.publish(offset, 0);
            return;
        }
        if (framed.len > max_blob) return error.PayloadTooLarge;
        const dst = try self.reserve(alloc, framed.len);
        @memcpy(dst.bytes, framed);
        try self.indexBlob(alloc, offset, dst.bytes);
        self.publish(offset, (@as(u64, dst.offset) << len_bits) | @as(u64, framed.len));
    }

    const Reserved = struct { bytes: []u8, offset: u64 };

    fn reserve(self: *Store, alloc: std.mem.Allocator, n: usize) !Reserved {
        std.debug.assert(n <= chunk_size);
        if (self.chunk_count == 0 or self.used + n > chunk_size) {
            if (self.chunk_count >= max_chunks) return error.PayloadArenaFull;
            const chunk = try alloc.alloc(u8, chunk_size);
            self.chunks[self.chunk_count].store(chunk.ptr, .release);
            self.chunk_count += 1;
            self.used = 0;
        }
        const chunk = self.chunks[self.chunk_count - 1].load(.monotonic).?;
        const start = self.used;
        self.used += n;
        return .{
            .bytes = chunk[start..][0..n],
            .offset = @as(u64, (self.chunk_count - 1)) * chunk_size + start,
        };
    }

    /// A write to `offset` that replaces a published blob leaves the old
    /// value's postings behind (the module doc), so the postings stop being
    /// exact. Called before the new postings go in: a `select` between the
    /// index insert and the publish would otherwise trust a posting for a
    /// value the published blob does not yet hold.
    fn noteRewrite(self: *Store, offset: u32) void {
        if (self.slots[offset].load(.monotonic) != 0) self.exact_postings.store(false, .release);
    }

    fn publish(self: *Store, offset: u32, slot: u64) void {
        const had = self.slots[offset].load(.monotonic) != 0;
        self.slots[offset].store(slot, .release);
        const bit = @as(u64, 1) << @intCast(offset % 64);
        if (slot != 0) {
            _ = self.published[offset / 64].fetchOr(bit, .release);
        } else {
            _ = self.published[offset / 64].fetchAnd(~bit, .release);
        }
        if (had and slot == 0) self.with_payload -= 1;
        if (!had and slot != 0) self.with_payload += 1;
    }

    /// Add a blob's values to every indexed field. Under the exclusive lock
    /// for the whole blob: `fields.items` is read here and appended to by
    /// `createIndex`, and the postings behind it are rehashed as they grow.
    fn indexBlob(self: *Store, alloc: std.mem.Allocator, offset: u32, blob: []const u8) !void {
        if (self.fields.items.len == 0) return;
        self.index_lock.lock();
        defer self.index_lock.unlock();
        // A failure part-way leaves postings for a blob that is never
        // published, and a later write to the offset will not remove them.
        errdefer self.exact_postings.store(false, .release);
        for (self.fields.items) |*f| try f.addBlob(alloc, offset, blob);
    }

    /// Drop `offset`'s documents from every text index: its payload was
    /// cleared, or the point deleted. The keyword and integer postings are
    /// hints and keep theirs (the module doc); BM25's statistics cannot.
    pub fn forgetPoint(self: *Store, offset: u32) void {
        if (self.fields.items.len == 0) return;
        self.index_lock.lock();
        defer self.index_lock.unlock();
        for (self.fields.items) |*f| if (f.index == .text) f.index.text.remove(offset);
    }

    /// `CreateFieldIndex` for a BM25 text field: tokenized with `opts`, over
    /// the `count` points stored so far that `deleted` does not hold, and
    /// every point written from now on.
    pub fn createTextIndex(
        self: *Store,
        alloc: std.mem.Allocator,
        name: []const u8,
        opts: text.tokenizer.Options,
        count: usize,
        deleted: *const std.bit_set.Dynamic,
    ) !void {
        if (try self.fieldPath(name)) |f| {
            if (f.index != .text) return error.IndexKindMismatch;
            if (!std.meta.eql(f.index.text.opts, opts)) return error.IndexKindMismatch;
            return;
        }
        const ti = try alloc.create(text.TextIndex);
        ti.* = text.TextIndex.init(alloc, opts);
        var f = Field.init(alloc, name, .{ .text = ti }, self.slots.len) catch |e| {
            ti.deinit();
            alloc.destroy(ti);
            return e;
        };
        errdefer f.deinit(alloc);
        // Populated before it is published, as `createIndex` does.
        var off: u32 = 0;
        while (off < count) : (off += 1) {
            if (deleted.isSet(off)) continue;
            const blob = self.get(off);
            if (blob.len == 0) continue;
            try f.addBlob(alloc, off, blob);
        }
        self.index_lock.lock();
        defer self.index_lock.unlock();
        try self.fields.append(alloc, f);
    }

    /// Hold the shared lock while reading `fields.items` from outside:
    /// `createIndex` appends under the exclusive half.
    pub fn lockFields(self: *const Store) void {
        const rw: *RwLock = @constCast(&self.index_lock);
        rw.lockShared();
    }

    pub fn unlockFields(self: *const Store) void {
        const rw: *RwLock = @constCast(&self.index_lock);
        rw.unlockShared();
    }

    /// The field indexing the path `name`, or null, including when `name`
    /// is no path.
    pub fn field(self: *const Store, name: []const u8) ?*const Field {
        return self.fieldPath(name) catch null;
    }

    /// The field indexing the path `name`, however it is spelled: `"a"` and
    /// `a` are one path.
    fn fieldPath(self: *const Store, name: []const u8) PathError!?*const Field {
        var buf: [max_path_steps]PathStep = undefined;
        const p = try parsePath(name, &buf);
        return self.fieldAt(p.first, buf[0..p.steps]);
    }

    fn fieldAt(self: *const Store, first: []const u8, rest: []const PathStep) ?*const Field {
        for (self.fields.items) |*f| if (f.hasPath(first, rest)) return f;
        return null;
    }

    /// `CreateFieldIndex`: index `name` as `kind` over the `count` points
    /// stored so far, and every point written from now on. Idempotent for the
    /// same kind; a different kind on an existing name is refused.
    pub fn createIndex(self: *Store, alloc: std.mem.Allocator, name: []const u8, kind: Kind, count: usize) !void {
        if (try self.fieldPath(name)) |f| {
            if (f.kind() != kind) return error.IndexKindMismatch;
            return;
        }
        const index: Field.Index = switch (kind) {
            .keyword => .{ .keyword = .empty },
            .integer => .{ .integer = .empty },
            .float => .{ .float = .{} },
            .datetime => .{ .datetime = .{ .sorted = .{} } },
            // `createTextIndex` builds a text field, with its options.
            .text => unreachable,
        };
        var f = try Field.init(alloc, name, index, self.slots.len);
        errdefer f.deinit(alloc);
        // Populate first, publish second. The field is private until it is
        // appended, so the backfill needs no lock; appending it under the
        // exclusive half is what makes it visible to `select`, which walks
        // `fields.items`. Published before its postings existed, a concurrent
        // filtered query found the new field, saw a partial posting list,
        // chose it as cheapest and answered a short page from it.
        var off: u32 = 0;
        while (off < count) : (off += 1) {
            const blob = self.get(off);
            if (blob.len == 0) continue;
            try f.addBlob(alloc, off, blob);
        }
        self.index_lock.lock();
        defer self.index_lock.unlock();
        try self.fields.append(alloc, f);
    }

    /// Whether some `must` condition can be answered from a posting list.
    pub fn indexedMust(self: *const Store, filter: *const Filter) bool {
        const rw: *RwLock = @constCast(&self.index_lock);
        rw.lockShared();
        defer rw.unlockShared();
        return self.cheapestMust(filter) != null;
    }

    const Pick = struct { f: *const Field, c: Condition };

    /// The indexed `must` condition with the shortest posting list, if any.
    /// Under the shared lock.
    fn cheapestMust(self: *const Store, filter: *const Filter) ?Pick {
        var best: ?Pick = null;
        var best_len: usize = std.math.maxInt(usize);
        for (filter.musts()) |c| {
            if (c.or_match != null) continue;
            const f = self.fieldAt(c.key, filter.restOf(c)) orelse continue;
            const n = postingLen(f, c.match) orelse continue;
            if (n < best_len) {
                best_len = n;
                best = .{ .f = f, .c = c };
            }
        }
        return best;
    }

    fn postingLen(f: *const Field, m: Match) ?usize {
        switch (m) {
            .keyword => |k| {
                if (f.index != .keyword) return null;
                return if (f.index.keyword.get(k)) |p| p.list.items.len else 0;
            },
            .integer => |i| {
                if (f.index != .integer) return null;
                return if (f.index.integer.get(i)) |p| p.list.items.len else 0;
            },
            .keywords => |list| {
                if (f.index != .keyword) return null;
                var n: usize = 0;
                var r = wire.Reader.init(list);
                while (!r.atEnd()) {
                    const t = r.tag() catch return null;
                    if (t.field == 1 and t.wire_type == .length_delimited) {
                        const s = r.bytes() catch return null;
                        if (f.index.keyword.get(s)) |p| n += p.list.items.len;
                    } else r.skip(t.wire_type) catch return null;
                }
                return n;
            },
            .integers => |list| {
                if (f.index != .integer) return null;
                var n: usize = 0;
                var r = wire.Reader.init(list);
                while (!r.atEnd()) {
                    const t = r.tag() catch return null;
                    // Field first, then the payload. See `repeatedIntegersContain`.
                    if (t.field != 1) {
                        r.skip(t.wire_type) catch return null;
                        continue;
                    }
                    switch (t.wire_type) {
                        .varint => {
                            const v: i64 = @bitCast(r.varint() catch return null);
                            if (f.index.integer.get(v)) |p| n += p.list.items.len;
                        },
                        .length_delimited => {
                            var pr = r.nested() catch return null;
                            while (!pr.atEnd()) {
                                const v: i64 = @bitCast(pr.varint() catch return null);
                                if (f.index.integer.get(v)) |p| n += p.list.items.len;
                            }
                        },
                        else => r.skip(t.wire_type) catch return null,
                    }
                }
                return n;
            },
            // Only a float field answers a range. An integer field's
            // postings hold no `double_value`, which a range admits, so
            // reading a range from them could miss points (the module doc:
            // the index may over-approximate, never miss); the blobs decide.
            .range => |r| {
                if (f.index != .float) return null;
                return f.index.float.count(r);
            },
            .datetime => |r| {
                if (f.index != .datetime) return null;
                return f.index.datetime.sorted.count(MicrosRange.of(r));
            },
            // No index holds what these ask: the blobs decide, as for an
            // unindexed field.
            .boolean, .values_count, .is_empty, .is_null, .all_empty => return null,
        }
    }

    /// Build the set of points `filter` admits, from the index.
    ///
    /// `bits` is the caller's scratch, one bit per offset up to `bound`
    /// (`Collection.id_space.count()` at the moment of the query -- the
    /// published offset count, *not* `Collection.count()`, which subtracts
    /// tombstones and would exclude the last live offsets: a posting may name
    /// an offset whose row is written but not yet published, and nothing may
    /// score that). Every offset a posting yields is verified against the
    /// point's blob and the whole filter, so the result is exact whatever the
    /// postings have accumulated, except where `exact_postings` makes the
    /// check redundant: a single-condition filter over postings no rewrite
    /// has made stale. Returns null when no `must` condition is
    /// indexed, in which case the caller evaluates blobs directly.
    pub fn select(self: *const Store, filter: *const Filter, bits: []u64, bound: usize) ?Selection {
        // The lock is the one thing a *reader* mutates, like `SearchGuard`.
        const rw: *RwLock = @constCast(&self.index_lock);
        rw.lockShared();
        defer rw.unlockShared();
        const pick = self.cheapestMust(filter) orelse return null;
        // When the picked condition is the whole filter and no posting is
        // stale, a posting *is* the answer: re-reading each blob to confirm it
        // cost as much as scoring the vectors on W12-sel1 at d=1536 (32% of
        // the engine's samples in `splitEntry`, `mem.eql` and
        // `valueMatchesAt`, against 47% in the distance kernel). `has` still
        // runs, so a point whose postings went in before its blob was
        // published is not admitted early.
        const trusted = filter.must_len == 1 and filter.must_not_len == 0 and
            filter.should_len == 0 and self.exact_postings.load(.acquire);
        const words = (bound + 63) / 64;
        std.debug.assert(words <= bits.len);
        @memset(bits[0..words], 0);
        switch (pick.f.index) {
            .float => |*fi| fi.admitRange(pick.c.match.range, self, filter, bits, bound, trusted),
            .datetime => |*di| {
                // Keys stand for instants exactly only without a leap second
                // (`DatetimeIndex`); otherwise every entry is verified.
                const r = MicrosRange.of(pick.c.match.datetime);
                di.sorted.admitRange(r, self, filter, bits, bound, trusted and r.exact and !di.leap);
            },
            else => {},
        }
        var lists = PostingLists.init(pick.f, pick.c.match);
        while (lists.next()) |p| {
            // A trusted dense posting is its own answer, masked to the
            // published points as `has` masks a list's.
            if (trusted and p.dense.len != 0) {
                for (bits[0..words], p.dense[0..words], self.published[0..words]) |*w, d, *pw| {
                    w.* |= d & pw.load(.acquire);
                }
                continue;
            }
            for (p.list.items) |off| self.admit(filter, bits, bound, trusted, off);
        }
        // The words cover `bound` rounded up: nothing past it is admitted.
        if (bound % 64 != 0) bits[words - 1] &= (@as(u64, 1) << @intCast(bound % 64)) - 1;
        var count: usize = 0;
        for (bits[0..words]) |w| count += @popCount(w);
        return .{ .count = count };
    }

    /// Set `off`'s bit if the filter admits it: a trusted index entry needs
    /// only a published blob, any other is verified against the blob.
    inline fn admit(self: *const Store, filter: *const Filter, bits: []u64, bound: usize, trusted: bool, off: u32) void {
        if (off >= bound) return;
        const w = off / 64;
        const m = @as(u64, 1) << @intCast(off % 64);
        if (bits[w] & m != 0) return;
        if (trusted) {
            if (!self.has(off)) return;
        } else if (!self.matches(off, filter)) return;
        bits[w] |= m;
    }

    /// The posting lists one `Match` reads: one for a scalar, several for a
    /// `keywords`/`integers` list.
    const PostingLists = struct {
        f: *const Field,
        m: Match,
        r: wire.Reader,
        packed_r: ?wire.Reader = null,
        done: bool = false,

        /// `postingLen` picked `f` for `m`, so its index is of `m`'s kind.
        fn init(f: *const Field, m: Match) PostingLists {
            const raw: []const u8 = switch (m) {
                .keywords, .integers => |l| l,
                else => &.{},
            };
            return .{ .f = f, .m = m, .r = wire.Reader.init(raw) };
        }

        fn next(self: *PostingLists) ?*const Postings {
            switch (self.m) {
                .keyword => |k| {
                    if (self.done) return null;
                    self.done = true;
                    return if (self.f.index.keyword.getPtr(k)) |p| p else null;
                },
                .integer => |i| {
                    if (self.done) return null;
                    self.done = true;
                    return if (self.f.index.integer.getPtr(i)) |p| p else null;
                },
                // A float or datetime field's entries are not postings;
                // `select` reads them before this runs.
                .boolean, .range, .datetime, .values_count, .is_empty, .is_null, .all_empty => return null,
                .keywords => {
                    while (!self.r.atEnd()) {
                        const t = self.r.tag() catch return null;
                        if (t.field == 1 and t.wire_type == .length_delimited) {
                            const s = self.r.bytes() catch return null;
                            if (self.f.index.keyword.getPtr(s)) |p| return p;
                        } else self.r.skip(t.wire_type) catch return null;
                    }
                    return null;
                },
                .integers => {
                    while (true) {
                        if (self.packed_r) |*pr| {
                            while (!pr.atEnd()) {
                                const v: i64 = @bitCast(pr.varint() catch return null);
                                if (self.f.index.integer.getPtr(v)) |p| return p;
                            }
                            self.packed_r = null;
                        }
                        if (self.r.atEnd()) return null;
                        const t = self.r.tag() catch return null;
                        switch (t.wire_type) {
                            .varint => {
                                const v: i64 = @bitCast(self.r.varint() catch return null);
                                if (t.field == 1) if (self.f.index.integer.getPtr(v)) |p| return p;
                            },
                            .length_delimited => {
                                if (t.field == 1) {
                                    self.packed_r = self.r.nested() catch return null;
                                } else self.r.skip(t.wire_type) catch return null;
                            },
                            else => self.r.skip(t.wire_type) catch return null,
                        }
                    }
                },
            }
        }
    };
};

// =========================================================================
// Predicates, in the shape `hnsw.Index.Filter` takes
// =========================================================================

/// A selection bitset: admits the offsets whose bit is set, below `bound`.
///
/// `bound` is the published offset count (`id_space.count()`) the selection
/// was built against. A point published after that (W11 writes during
/// searches) has no bit, and the word it would land in may hold a stale one
/// from an earlier query, so anything past the bound is refused rather than
/// read.
pub const BitsCtx = struct {
    bits: []const u64,
    bound: usize,

    pub fn pred(ctx: *const anyopaque, node: u32) bool {
        const self: *const BitsCtx = @ptrCast(@alignCast(ctx));
        return node < self.bound and testBit(self.bits, node);
    }
};

pub fn testBit(bits: []const u64, node: u32) bool {
    return bits[node / 64] & (@as(u64, 1) << @intCast(node % 64)) != 0;
}

/// Direct evaluation against the blob, for a filter no posting list answers.
pub const EvalCtx = struct {
    store: *const Store,
    filter: *const Filter,

    pub fn pred(ctx: *const anyopaque, node: u32) bool {
        const self: *const EvalCtx = @ptrCast(@alignCast(ctx));
        return self.store.matches(node, self.filter);
    }
};

// =========================================================================
// Tests
// =========================================================================

const testing = std.testing;

/// `{ key: Value }` as a map entry, `Value` being one of the scalar kinds.
fn entryBytes(buf: []u8, key: []const u8, comptime value_field: u32, value: anytype) []const u8 {
    var w = wire.Writer.init(buf);
    w.writeStringField(entry_key, key) catch unreachable;
    const val = w.beginNested(entry_value, 2) catch unreachable;
    if (comptime value_field == value_string) {
        w.writeStringField(value_string, value) catch unreachable;
    } else if (comptime value_field == value_integer) {
        w.writeVarintFieldAlways(value_integer, @bitCast(@as(i64, value))) catch unreachable;
    } else if (comptime value_field == value_bool) {
        w.writeBoolField(value_bool, value) catch unreachable;
    } else if (comptime value_field == value_double) {
        // Always written: `writeDoubleField` drops a 0.0, which is a value.
        w.tag(value_double, .fixed64) catch unreachable;
        w.fixed64(@bitCast(@as(f64, value))) catch unreachable;
    } else {
        @compileError("unsupported value kind");
    }
    w.endNested(val) catch unreachable;
    return w.written();
}

/// A payload map with the entries in `entries`, each tagged as field 3.
fn mapBytes(buf: []u8, entries: []const []const u8) []const u8 {
    var w = wire.Writer.init(buf);
    for (entries) |e| {
        w.tag(3, .length_delimited) catch unreachable;
        w.varint(e.len) catch unreachable;
        w.raw(e) catch unreachable;
    }
    return w.written();
}

fn keywordFilter(key: []const u8, value: []const u8) Filter {
    var f = Filter{};
    f.add(.must, .{ .key = key, .match = .{ .keyword = value } }) catch unreachable;
    return f;
}

test "a payload round-trips as the bytes it arrived as, and a bare upsert clears it" {
    var store = try Store.init(testing.allocator, 8);
    defer store.deinit(testing.allocator);
    var e1: [64]u8 = undefined;
    var e2: [64]u8 = undefined;
    var m: [256]u8 = undefined;
    const a = entryBytes(&e1, "a", value_string, "keyword_7");
    const b = entryBytes(&e2, "n", value_integer, -3);
    try store.set(testing.allocator, 2, mapBytes(&m, &.{ a, b }), 3);
    try testing.expect(store.has(2));
    try testing.expect(!store.has(1));
    try testing.expectEqual(@as(usize, 1), store.with_payload);

    var it = BlobIterator.init(store.get(2));
    try testing.expectEqualSlices(u8, a, it.next().?);
    try testing.expectEqualSlices(u8, b, it.next().?);
    try testing.expect(it.next() == null);

    // Overwrite with nothing: gone, as a Qdrant upsert without a payload does.
    try store.set(testing.allocator, 2, &.{}, 3);
    try testing.expect(!store.has(2));
    try testing.expectEqual(@as(usize, 0), store.with_payload);
    try testing.expectEqual(@as(usize, 0), store.get(2).len);
}

test "must, must_not and should evaluate against the blob with Qdrant's semantics" {
    var store = try Store.init(testing.allocator, 8);
    defer store.deinit(testing.allocator);
    var e1: [64]u8 = undefined;
    var e2: [64]u8 = undefined;
    var e3: [64]u8 = undefined;
    var m: [256]u8 = undefined;
    const a = entryBytes(&e1, "a", value_string, "red");
    const n = entryBytes(&e2, "n", value_integer, 42);
    const b = entryBytes(&e3, "b", value_bool, true);
    try store.set(testing.allocator, 0, mapBytes(&m, &.{ a, n, b }), 3);

    try testing.expect(store.matches(0, &keywordFilter("a", "red")));
    try testing.expect(!store.matches(0, &keywordFilter("a", "blue")));
    // A key the point does not carry matches nothing.
    try testing.expect(!store.matches(0, &keywordFilter("zzz", "red")));
    // Types do not coerce: the integer 42 is not the keyword "42".
    try testing.expect(!store.matches(0, &keywordFilter("n", "42")));

    var f = Filter{};
    try f.add(.must, .{ .key = "n", .match = .{ .integer = 42 } });
    try f.add(.must, .{ .key = "b", .match = .{ .boolean = true } });
    try testing.expect(store.matches(0, &f));
    try f.add(.must_not, .{ .key = "a", .match = .{ .keyword = "red" } });
    try testing.expect(!store.matches(0, &f));

    // `should`: at least one when any is given.
    var s = Filter{};
    try s.add(.should, .{ .key = "a", .match = .{ .keyword = "blue" } });
    try testing.expect(!store.matches(0, &s));
    try s.add(.should, .{ .key = "a", .match = .{ .keyword = "red" } });
    try testing.expect(store.matches(0, &s));

    // `must_not` alone on an absent key holds; an empty filter admits all.
    var mn = Filter{};
    try mn.add(.must_not, .{ .key = "zzz", .match = .{ .keyword = "x" } });
    try testing.expect(store.matches(0, &mn));
    try testing.expect(store.matches(1, &Filter{}));
    try testing.expect(!store.matches(1, &keywordFilter("a", "red")));
}

test "match_any over a RepeatedStrings and a list-valued field" {
    var store = try Store.init(testing.allocator, 8);
    defer store.deinit(testing.allocator);
    // `{ tags: ["x", "y"] }`: a ListValue of two string Values.
    var ebuf: [128]u8 = undefined;
    var w = wire.Writer.init(&ebuf);
    try w.writeStringField(entry_key, "tags");
    const val = try w.beginNested(entry_value, 2);
    const list = try w.beginNested(value_list, 2);
    for ([_][]const u8{ "x", "y" }) |s| {
        const v = try w.beginNested(1, 2);
        try w.writeStringField(value_string, s);
        try w.endNested(v);
    }
    try w.endNested(list);
    try w.endNested(val);
    var m: [256]u8 = undefined;
    try store.set(testing.allocator, 0, mapBytes(&m, &.{w.written()}), 3);

    try testing.expect(store.matches(0, &keywordFilter("tags", "y")));
    try testing.expect(!store.matches(0, &keywordFilter("tags", "z")));

    // `RepeatedStrings { strings = ["q", "x"] }`.
    var rs: [32]u8 = undefined;
    var rw = wire.Writer.init(&rs);
    try rw.writeStringField(1, "q");
    try rw.writeStringField(1, "x");
    var f = Filter{};
    try f.add(.must, .{ .key = "tags", .match = .{ .keywords = rw.written() } });
    try testing.expect(store.matches(0, &f));

    // And the index sees both list elements.
    try store.createIndex(testing.allocator, "tags", .keyword, 1);
    var bits: [1]u64 = undefined;
    const sel = store.select(&keywordFilter("tags", "x"), &bits, 1).?;
    try testing.expectEqual(@as(usize, 1), sel.count);
    try testing.expect(testBit(&bits, 0));
    try testing.expectEqual(@as(usize, 1), store.select(&f, &bits, 1).?.count);
}

test "the index selects exactly the matching points, whether built before or after the writes" {
    var store = try Store.init(testing.allocator, 128);
    defer store.deinit(testing.allocator);
    var m: [128]u8 = undefined;
    var e: [64]u8 = undefined;
    var nbuf: [16]u8 = undefined;
    // 64 points, keyword `a` = "k<i % 4>", indexed after the first half.
    for (0..64) |i| {
        if (i == 32) try store.createIndex(testing.allocator, "a", .keyword, 32);
        const s = try std.mem.print(&nbuf, "k{d}", .{i % 4});
        try store.set(testing.allocator, @intCast(i), mapBytes(&m, &.{entryBytes(&e, "a", value_string, s)}), 3);
    }
    var bits: [2]u64 = undefined;
    const sel = store.select(&keywordFilter("a", "k1"), &bits, 64).?;
    try testing.expectEqual(@as(usize, 16), sel.count);
    for (0..64) |i| try testing.expectEqual(i % 4 == 1, testBit(&bits, @intCast(i)));

    // A value nobody has: an empty selection, not null.
    try testing.expectEqual(@as(usize, 0), store.select(&keywordFilter("a", "k9"), &bits, 64).?.count);
    // An unindexed key: null, so the caller evaluates blobs.
    try testing.expect(store.select(&keywordFilter("zzz", "k1"), &bits, 64) == null);
    // A bound below the postings: offsets past it are not admitted.
    try testing.expectEqual(@as(usize, 4), store.select(&keywordFilter("a", "k1"), &bits, 16).?.count);

    // The same kind again is a no-op; another kind on the name is refused.
    try store.createIndex(testing.allocator, "a", .keyword, 64);
    try testing.expectError(error.IndexKindMismatch, store.createIndex(testing.allocator, "a", .integer, 64));
    try testing.expectEqual(@as(usize, 1), store.fields.items.len);
}

test "a text field and a keyword field refuse each other's name, and a text field selects nothing" {
    var store = try Store.init(testing.allocator, 8);
    defer store.deinit(testing.allocator);
    var deleted = try std.bit_set.Dynamic.initEmpty(testing.allocator, 8);
    defer deleted.deinit(testing.allocator);
    var m: [64]u8 = undefined;
    var e: [32]u8 = undefined;
    try store.set(testing.allocator, 0, mapBytes(&m, &.{entryBytes(&e, "t", value_string, "k1")}), 3);
    try store.createTextIndex(testing.allocator, "t", .{}, 1, &deleted);
    try store.createIndex(testing.allocator, "a", .keyword, 1);
    try testing.expectError(error.IndexKindMismatch, store.createIndex(testing.allocator, "t", .keyword, 1));
    try testing.expectError(error.IndexKindMismatch, store.createTextIndex(testing.allocator, "a", .{}, 1, &deleted));
    try testing.expectEqual(Kind.text, store.field("t").?.kind());
    try testing.expectEqual(Kind.keyword, store.field("a").?.kind());
    // A keyword match on the text field has no posting list to read.
    var bits: [1]u64 = undefined;
    try testing.expect(store.select(&keywordFilter("t", "k1"), &bits, 1) == null);
}

test "an overwrite leaves a stale posting that verification drops" {
    var store = try Store.init(testing.allocator, 8);
    defer store.deinit(testing.allocator);
    try store.createIndex(testing.allocator, "a", .keyword, 0);
    var m: [128]u8 = undefined;
    var e: [64]u8 = undefined;
    try store.set(testing.allocator, 0, mapBytes(&m, &.{entryBytes(&e, "a", value_string, "old")}), 3);
    try store.set(testing.allocator, 0, mapBytes(&m, &.{entryBytes(&e, "a", value_string, "new")}), 3);
    var bits: [1]u64 = undefined;
    // The "old" posting still names offset 0; the blob says otherwise.
    try testing.expectEqual(@as(usize, 0), store.select(&keywordFilter("a", "old"), &bits, 1).?.count);
    try testing.expectEqual(@as(usize, 1), store.select(&keywordFilter("a", "new"), &bits, 1).?.count);
    // A point whose payload was cleared drops out of every selection.
    try store.set(testing.allocator, 0, &.{}, 3);
    try testing.expectEqual(@as(usize, 0), store.select(&keywordFilter("a", "new"), &bits, 1).?.count);
}

test "postings stay exact until a payload is rewritten, and select trusts them only then" {
    var store = try Store.init(testing.allocator, 8);
    defer store.deinit(testing.allocator);
    try store.createIndex(testing.allocator, "a", .keyword, 0);
    try store.createIndex(testing.allocator, "b", .keyword, 0);
    var m: [128]u8 = undefined;
    var e1: [64]u8 = undefined;
    var e2: [64]u8 = undefined;
    // Append-only: 0 is (x, y), 1 is (x, z), 2 has no payload.
    try store.set(testing.allocator, 0, mapBytes(&m, &.{
        entryBytes(&e1, "a", value_string, "x"), entryBytes(&e2, "b", value_string, "y"),
    }), 3);
    try store.set(testing.allocator, 1, mapBytes(&m, &.{
        entryBytes(&e1, "a", value_string, "x"), entryBytes(&e2, "b", value_string, "z"),
    }), 3);
    try store.set(testing.allocator, 2, &.{}, 3);
    try testing.expect(store.exact_postings.load(.acquire));
    var bits: [1]u64 = undefined;
    try testing.expectEqual(@as(usize, 2), store.select(&keywordFilter("a", "x"), &bits, 3).?.count);
    try testing.expectEqual(@as(u64, 0b11), bits[0]);
    // Two conditions: the postings answer one of them, so the blobs still
    // decide, and only point 0 satisfies both.
    var both = keywordFilter("a", "x");
    try both.add(.must, .{ .key = "b", .match = .{ .keyword = "y" } });
    try testing.expectEqual(@as(usize, 1), store.select(&both, &bits, 3).?.count);
    try testing.expectEqual(@as(u64, 0b01), bits[0]);
    // A must_not is not answered by a posting either.
    var not_y = keywordFilter("a", "x");
    try not_y.add(.must_not, .{ .key = "b", .match = .{ .keyword = "y" } });
    try testing.expectEqual(@as(usize, 1), store.select(&not_y, &bits, 3).?.count);
    try testing.expectEqual(@as(u64, 0b10), bits[0]);

    // Rewriting 1 leaves its "x" posting stale: from here the blobs decide.
    try store.set(testing.allocator, 1, mapBytes(&m, &.{entryBytes(&e1, "a", value_string, "w")}), 3);
    try testing.expect(!store.exact_postings.load(.acquire));
    try testing.expectEqual(@as(usize, 1), store.select(&keywordFilter("a", "x"), &bits, 3).?.count);
    try testing.expectEqual(@as(u64, 0b01), bits[0]);
}

test "a dense posting answers as its list does, and only for published points" {
    var store = try Store.init(testing.allocator, 4096);
    defer store.deinit(testing.allocator);
    try store.createIndex(testing.allocator, "a", .keyword, 0);
    var m: [64]u8 = undefined;
    var e: [32]u8 = undefined;
    // 200 points of "dense", past capacity / 32, and 4 of "sparse".
    var off: u32 = 0;
    while (off < 204) : (off += 1) {
        const v = if (off < 200) "dense" else "sparse";
        try store.set(testing.allocator, off, mapBytes(&m, &.{entryBytes(&e, "a", value_string, v)}), 3);
    }
    const field = &store.fields.items[0];
    try testing.expect(field.index.keyword.get("dense").?.dense.len != 0);
    try testing.expectEqual(@as(usize, 0), field.index.keyword.get("sparse").?.dense.len);
    var bits: [64]u64 = undefined;
    try testing.expectEqual(@as(usize, 200), store.select(&keywordFilter("a", "dense"), &bits, 204).?.count);
    try testing.expectEqual(~@as(u64, 0), bits[0]);
    try testing.expectEqual(@as(u64, 0xff), bits[3]);
    try testing.expectEqual(@as(usize, 4), store.select(&keywordFilter("a", "sparse"), &bits, 204).?.count);
    // Indexed but not yet published: the dense answer leaves it out, as
    // `has` leaves a list's out.
    const blob = mapBytes(&m, &.{entryBytes(&e, "a", value_string, "dense")});
    var framed: [64]u8 = undefined;
    const n = writeVarint(&framed, blob.len);
    @memcpy(framed[n..][0..blob.len], blob);
    try store.indexBlob(testing.allocator, 300, framed[0 .. n + blob.len]);
    try testing.expectEqual(@as(usize, 200), store.select(&keywordFilter("a", "dense"), &bits, 301).?.count);
    // A bound below the points cuts the answer at the bound.
    try testing.expectEqual(@as(usize, 70), store.select(&keywordFilter("a", "dense"), &bits, 70).?.count);
    try testing.expectEqual(@as(u64, 0x3f), bits[1]);
}

test "a first payload keeps the postings exact; a clear or a merge does not" {
    var m: [128]u8 = undefined;
    var e: [64]u8 = undefined;
    // Clearing a published payload.
    {
        var store = try Store.init(testing.allocator, 4);
        defer store.deinit(testing.allocator);
        try store.createIndex(testing.allocator, "a", .keyword, 0);
        try store.set(testing.allocator, 0, mapBytes(&m, &.{entryBytes(&e, "a", value_string, "x")}), 3);
        try testing.expect(store.exact_postings.load(.acquire));
        try store.set(testing.allocator, 0, &.{}, 3);
        try testing.expect(!store.exact_postings.load(.acquire));
        var bits: [1]u64 = undefined;
        try testing.expectEqual(@as(usize, 0), store.select(&keywordFilter("a", "x"), &bits, 1).?.count);
    }
    // Merging over a published payload.
    {
        var store = try Store.init(testing.allocator, 4);
        defer store.deinit(testing.allocator);
        try store.createIndex(testing.allocator, "a", .keyword, 0);
        try store.set(testing.allocator, 0, mapBytes(&m, &.{entryBytes(&e, "a", value_string, "x")}), 3);
        try store.merge(testing.allocator, 0, mapBytes(&m, &.{entryBytes(&e, "a", value_string, "y")}), 3);
        try testing.expect(!store.exact_postings.load(.acquire));
        var bits: [1]u64 = undefined;
        try testing.expectEqual(@as(usize, 0), store.select(&keywordFilter("a", "x"), &bits, 1).?.count);
        try testing.expectEqual(@as(usize, 1), store.select(&keywordFilter("a", "y"), &bits, 1).?.count);
    }
}

test "SetPayload merges by key and indexes the merged values" {
    var store = try Store.init(testing.allocator, 8);
    defer store.deinit(testing.allocator);
    try store.createIndex(testing.allocator, "n", .integer, 0);
    var m: [256]u8 = undefined;
    var e1: [64]u8 = undefined;
    var e2: [64]u8 = undefined;
    try store.set(testing.allocator, 3, mapBytes(&m, &.{
        entryBytes(&e1, "a", value_string, "red"),
        entryBytes(&e2, "n", value_integer, 1),
    }), 3);
    // Set `n` to 2 and add `c`: `a` survives, `n` is replaced, `c` appended.
    var e3: [64]u8 = undefined;
    var e4: [64]u8 = undefined;
    try store.merge(testing.allocator, 3, mapBytes(&m, &.{
        entryBytes(&e3, "n", value_integer, 2),
        entryBytes(&e4, "c", value_string, "z"),
    }), 3);
    try testing.expect(store.matches(3, &keywordFilter("a", "red")));
    try testing.expect(store.matches(3, &keywordFilter("c", "z")));
    var f1 = Filter{};
    try f1.add(.must, .{ .key = "n", .match = .{ .integer = 1 } });
    var f2 = Filter{};
    try f2.add(.must, .{ .key = "n", .match = .{ .integer = 2 } });
    try testing.expect(!store.matches(3, &f1));
    try testing.expect(store.matches(3, &f2));
    var bits: [1]u64 = undefined;
    try testing.expectEqual(@as(usize, 1), store.select(&f2, &bits, 4).?.count);
    try testing.expectEqual(@as(usize, 0), store.select(&f1, &bits, 4).?.count);
    // Exactly three entries, `a` first as it was stored.
    var it = BlobIterator.init(store.get(3));
    var n: usize = 0;
    while (it.next()) |entry| : (n += 1) {
        const kv = splitEntry(entry).?;
        const want: []const u8 = switch (n) {
            0 => "a",
            1 => "n",
            2 => "c",
            else => unreachable,
        };
        try testing.expectEqualStrings(want, kv.key);
    }
    try testing.expectEqual(@as(usize, 3), n);
}

test "blobs cross chunk boundaries without moving what readers hold" {
    var store = try Store.init(testing.allocator, 4);
    defer store.deinit(testing.allocator);
    // One entry of ~half a chunk, written three times: the third opens a
    // second chunk, and the first two stay readable where they were.
    const big = try testing.allocator.alloc(u8, chunk_size / 2 - 64);
    defer testing.allocator.free(big);
    @memset(big, 'x');
    const ebuf = try testing.allocator.alloc(u8, big.len + 64);
    defer testing.allocator.free(ebuf);
    var w = wire.Writer.init(ebuf);
    try w.writeStringField(entry_key, "blob");
    const val = try w.beginNested(entry_value, 4);
    try w.writeStringField(value_string, big);
    try w.endNested(val);
    const entry = w.written();
    const mbuf = try testing.allocator.alloc(u8, entry.len + 16);
    defer testing.allocator.free(mbuf);
    const map = mapBytes(mbuf, &.{entry});
    try store.set(testing.allocator, 0, map, 3);
    const first = store.get(0);
    try store.set(testing.allocator, 1, map, 3);
    try store.set(testing.allocator, 2, map, 3);
    try testing.expectEqual(@as(usize, 2), store.chunk_count);
    try testing.expectEqual(first.ptr, store.get(0).ptr);
    var it = BlobIterator.init(store.get(2));
    try testing.expectEqualSlices(u8, entry, it.next().?);
}

const Num = union(enum) { int: i64, dbl: f64 };

/// `{ key: n }`, or `{ key: [n, ...] }` when `as_list`.
fn numEntry(buf: []u8, key: []const u8, nums: []const Num, as_list: bool) []const u8 {
    var w = wire.Writer.init(buf);
    w.writeStringField(entry_key, key) catch unreachable;
    const val = w.beginNested(entry_value, 2) catch unreachable;
    if (as_list) {
        const list = w.beginNested(value_list, 2) catch unreachable;
        for (nums) |n| {
            const v = w.beginNested(1, 2) catch unreachable;
            writeNum(&w, n);
            w.endNested(v) catch unreachable;
        }
        w.endNested(list) catch unreachable;
    } else writeNum(&w, nums[0]);
    w.endNested(val) catch unreachable;
    return w.written();
}

fn writeNum(w: *wire.Writer, n: Num) void {
    switch (n) {
        .int => |i| w.writeVarintFieldAlways(value_integer, @bitCast(i)) catch unreachable,
        .dbl => |d| {
            w.tag(value_double, .fixed64) catch unreachable;
            w.fixed64(@bitCast(d)) catch unreachable;
        },
    }
}

fn rangeFilter(key: []const u8, r: Range) Filter {
    var f = Filter{};
    f.add(.must, .{ .key = key, .match = .{ .range = r } }) catch unreachable;
    return f;
}

test "a range folds its bounds into Qdrant's OrderedFloat order" {
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    // Inclusive and exclusive ends.
    const gt = Range.fromBounds(null, 1, null, null);
    try testing.expect(!gt.contains(1));
    try testing.expect(gt.contains(1.0000001));
    const gte = Range.fromBounds(null, null, 1, null);
    try testing.expect(gte.contains(1));
    try testing.expect(!gte.contains(0.999));
    const lt = Range.fromBounds(2, null, null, null);
    try testing.expect(!lt.contains(2));
    try testing.expect(lt.contains(1.999));
    const lte = Range.fromBounds(null, null, null, 2);
    try testing.expect(lte.contains(2));
    try testing.expect(!lte.contains(2.001));
    // Combined: the strictest of each side holds, whichever order they came.
    const both = Range.fromBounds(5, 1, 1, 6);
    try testing.expect(!both.contains(1));
    try testing.expect(both.contains(4.5));
    try testing.expect(!both.contains(5));
    const looser_strict = Range.fromBounds(null, 0, 1, null);
    try testing.expect(looser_strict.contains(1));
    // An empty interval admits nothing.
    try testing.expect(!Range.fromBounds(1, 2, null, null).contains(1.5));
    // No bounds: every number, infinities included, as `check_range` with
    // every bound `None`.
    const all = Range{};
    try testing.expect(all.contains(-inf));
    try testing.expect(all.contains(inf));
    try testing.expect(all.contains(0));
    // A NaN value never matches; a NaN bound is OrderedFloat's maximum.
    try testing.expect(!all.contains(nan));
    try testing.expect(Range.fromBounds(nan, null, null, null).contains(1e300));
    try testing.expect(!Range.fromBounds(null, nan, null, null).contains(inf));
    try testing.expect(!Range.fromBounds(null, null, nan, null).contains(inf));
    // `-0.0 == 0.0`.
    try testing.expect(!Range.fromBounds(null, 0.0, null, null).contains(-0.0));
    try testing.expect(Range.fromBounds(null, null, -0.0, null).contains(0.0));
}

test "a range matches double and integer payloads, any list element, never a missing field" {
    var store = try Store.init(testing.allocator, 8);
    defer store.deinit(testing.allocator);
    var e1: [64]u8 = undefined;
    var e2: [128]u8 = undefined;
    var m: [256]u8 = undefined;
    // 0: double 0.5; 1: integer 3; 2: [string, 10, 0.25]; 3: string "4";
    // 4: no `x`; 5: NaN; 6: bool.
    try store.set(testing.allocator, 0, mapBytes(&m, &.{entryBytes(&e1, "x", value_double, 0.5)}), 3);
    try store.set(testing.allocator, 1, mapBytes(&m, &.{entryBytes(&e1, "x", value_integer, 3)}), 3);
    {
        var w = wire.Writer.init(&e2);
        try w.writeStringField(entry_key, "x");
        const val = try w.beginNested(entry_value, 2);
        const list = try w.beginNested(value_list, 2);
        const s = try w.beginNested(1, 2);
        try w.writeStringField(value_string, "7");
        try w.endNested(s);
        for ([_]Num{ .{ .int = 10 }, .{ .dbl = 0.25 } }) |n| {
            const v = try w.beginNested(1, 2);
            writeNum(&w, n);
            try w.endNested(v);
        }
        try w.endNested(list);
        try w.endNested(val);
        try store.set(testing.allocator, 2, mapBytes(&m, &.{w.written()}), 3);
    }
    try store.set(testing.allocator, 3, mapBytes(&m, &.{entryBytes(&e1, "x", value_string, "4")}), 3);
    try store.set(testing.allocator, 4, mapBytes(&m, &.{entryBytes(&e1, "y", value_double, 0.5)}), 3);
    try store.set(testing.allocator, 5, mapBytes(&m, &.{entryBytes(&e1, "x", value_double, std.math.nan(f64))}), 3);
    try store.set(testing.allocator, 6, mapBytes(&m, &.{entryBytes(&e1, "x", value_bool, true)}), 3);

    const Case = struct { r: Range, want: [7]bool };
    const cases = [_]Case{
        // Below 1: the double, and the list's 0.25.
        .{ .r = Range.fromBounds(1, null, null, null), .want = .{ true, false, true, false, false, false, false } },
        // At least 3: the integer, and the list's 10.
        .{ .r = Range.fromBounds(null, null, 3, null), .want = .{ false, true, true, false, false, false, false } },
        .{ .r = Range.fromBounds(null, 3, null, null), .want = .{ false, false, true, false, false, false, false } },
        // `(0.5, 3]`: an integer compared as f64, the double's bound excluded.
        .{ .r = Range.fromBounds(null, 0.5, null, 3), .want = .{ false, true, false, false, false, false, false } },
        // No bounds: every numeric value; a string, a bool, NaN and a missing
        // field are not numbers.
        .{ .r = .{}, .want = .{ true, true, true, false, false, false, false } },
    };
    for (cases) |cs| {
        for (cs.want, 0..) |want, off| {
            try testing.expectEqual(want, store.matches(@intCast(off), &rangeFilter("x", cs.r)));
        }
    }
    // must_not on a range holds for the missing field and the NaN; should
    // ORs a range with a keyword match.
    var mn = Filter{};
    try mn.add(.must_not, .{ .key = "x", .match = .{ .range = .{} } });
    for ([_]bool{ false, false, false, true, true, true, true }, 0..) |want, off| {
        try testing.expectEqual(want, store.matches(@intCast(off), &mn));
    }
    var sh = Filter{};
    try sh.add(.should, .{ .key = "x", .match = .{ .range = Range.fromBounds(null, 5, null, null) } });
    try sh.add(.should, .{ .key = "x", .match = .{ .keyword = "4" } });
    for ([_]bool{ false, false, true, true, false, false, false }, 0..) |want, off| {
        try testing.expectEqual(want, store.matches(@intCast(off), &sh));
    }
    // A FieldCondition with both a match and a range admits either.
    var both = Filter{};
    try both.add(.must, .{ .key = "x", .match = .{ .keyword = "4" }, .or_match = .{ .range = Range.fromBounds(null, null, null, 0.5) } });
    for ([_]bool{ true, false, true, true, false, false, false }, 0..) |want, off| {
        try testing.expectEqual(want, store.matches(@intCast(off), &both));
    }
}

/// `{ key: value }`, the value given as its encoded `Value` bytes.
fn rawEntry(buf: []u8, key: []const u8, value: []const u8) []const u8 {
    var w = wire.Writer.init(buf);
    w.writeStringField(entry_key, key) catch unreachable;
    w.tag(entry_value, .length_delimited) catch unreachable;
    w.varint(value.len) catch unreachable;
    w.raw(value) catch unreachable;
    return w.written();
}

/// Encoded `Value`s, by hand: `null_value = 1` (varint), `integer_value =
/// 3`, `string_value = 4`, `struct_value = 6`, `list_value = 7`, a list's
/// elements at `ListValue.values = 1`.
const shape_values = [_][]const u8{
    "\x08\x00", // null
    "", // no kind set: null to Qdrant
    "\x3a\x00", // []
    "\x3a\x04\x0a\x02\x08\x00", // [null]
    "\x3a\x08\x0a\x02\x18\x01\x0a\x02\x08\x00", // [1, null]
    "\x3a\x08\x0a\x02\x18\x01\x0a\x02\x18\x02", // [1, 2]
    "\x22\x01s", // "s"
    "\x32\x09\x0a\x07\x0a\x01b\x12\x02\x08\x00", // {"b": null}
    "\x18\x00", // 0
    "\x3a\x04\x0a\x02\x3a\x00", // [[]]
    "\x3a\x02\x0a\x00", // [<no kind>]
};

/// A store whose point `i < shape_values.len` holds `t` = `shape_values[i]`
/// beside an unrelated key; the next point has only the unrelated key, and
/// the one after it no payload at all.
fn shapeStore() !Store {
    var store = try Store.init(testing.allocator, 16);
    errdefer store.deinit(testing.allocator);
    var e1: [64]u8 = undefined;
    var e2: [64]u8 = undefined;
    var m: [256]u8 = undefined;
    const other = entryBytes(&e2, "o", value_integer, 1);
    for (shape_values, 0..) |v, i| {
        try store.set(testing.allocator, @intCast(i), mapBytes(&m, &.{ rawEntry(&e1, "t", v), other }), 3);
    }
    try store.set(testing.allocator, shape_values.len, mapBytes(&m, &.{other}), 3);
    return store;
}

fn expectAdmits(store: *const Store, clause: Filter.Clause, c: Condition, want: [shape_values.len + 2]bool) !void {
    var f = Filter{};
    try f.add(clause, c);
    for (want, 0..) |w, off| {
        testing.expectEqual(w, store.matches(@intCast(off), &f)) catch |e| {
            std.debug.print("point {d}\n", .{off});
            return e;
        };
    }
}

test "is_empty, is_null and values_count read a value as Qdrant's checks do, a missing field included" {
    var store = try shapeStore();
    defer store.deinit(testing.allocator);
    const T = true;
    const F = false;
    // Columns: null, no kind, [], [null], [1, null], [1, 2], "s", {"b": null},
    // 0, [[]], [<no kind>], `t` missing, no payload.
    //
    // `check_is_empty`: null or an empty list; missing is empty (`check_empty`
    // returns `is_empty`). `[null]` has an element, so it is not.
    try expectAdmits(&store, .must, .{ .key = "t", .match = .{ .is_empty = true } }, .{ T, T, T, F, F, F, F, F, F, F, F, T, T });
    // `is_empty: false` is the complement on a present field, and false on a
    // missing one.
    try expectAdmits(&store, .must, .{ .key = "t", .match = .{ .is_empty = false } }, .{ F, F, F, T, T, T, T, T, T, T, T, F, F });
    // `check_is_null`: null, or a list holding a null; a struct holding one
    // is an object, and a missing field is not null.
    try expectAdmits(&store, .must, .{ .key = "t", .match = .{ .is_null = true } }, .{ T, T, F, T, T, F, F, F, F, F, T, F, F });
    // `is_null: false` on a missing field holds (`check_empty` returns
    // `!is_null`).
    try expectAdmits(&store, .must, .{ .key = "t", .match = .{ .is_null = false } }, .{ F, F, T, F, F, T, T, T, T, T, F, T, T });
    // Counts: 0, 0, 0, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0.
    try expectAdmits(&store, .must, .{ .key = "t", .match = .{ .values_count = ValuesCount.fromBounds(1, null, null, null) } }, .{ T, T, T, F, F, F, F, F, F, F, F, T, T });
    try expectAdmits(&store, .must, .{ .key = "t", .match = .{ .values_count = ValuesCount.fromBounds(null, null, 2, null) } }, .{ F, F, F, F, T, T, F, F, F, F, F, F, F });
    try expectAdmits(&store, .must, .{ .key = "t", .match = .{ .values_count = ValuesCount.fromBounds(null, 0, null, 1) } }, .{ F, F, F, T, F, F, T, T, T, T, T, F, F });
    // No bounds: every count, so every point.
    try expectAdmits(&store, .must, .{ .key = "t", .match = .{ .values_count = .{} } }, .{ T, T, T, T, T, T, T, T, T, T, T, T, T });
    // must_not negates per point, missing field and absent payload alike.
    try expectAdmits(&store, .must_not, .{ .key = "t", .match = .{ .is_empty = true } }, .{ F, F, F, T, T, T, T, T, T, T, T, F, F });
    try expectAdmits(&store, .must_not, .{ .key = "t", .match = .{ .is_null = true } }, .{ F, F, T, F, F, T, T, T, T, T, F, T, T });
    // should ORs a shape condition with a value match.
    var sh = Filter{};
    try sh.add(.should, .{ .key = "t", .match = .{ .is_null = true } });
    try sh.add(.should, .{ .key = "t", .match = .{ .keyword = "s" } });
    for ([_]bool{ T, T, F, T, T, F, T, F, F, F, T, F, F }, 0..) |want, off| {
        try testing.expectEqual(want, store.matches(@intCast(off), &sh));
    }
}

test "values_count folds its bounds, the u64 ends included" {
    const max = std.math.maxInt(u64);
    try testing.expect(!ValuesCount.fromBounds(null, 2, null, null).contains(2));
    try testing.expect(ValuesCount.fromBounds(null, 2, null, null).contains(3));
    try testing.expect(ValuesCount.fromBounds(null, null, 2, null).contains(2));
    try testing.expect(!ValuesCount.fromBounds(2, null, null, null).contains(2));
    try testing.expect(ValuesCount.fromBounds(null, null, null, 2).contains(2));
    try testing.expect(!ValuesCount.fromBounds(null, null, null, 2).contains(3));
    // `lt: 0` admits nothing, `gt: max` nothing, `lte: max` everything.
    try testing.expect(!ValuesCount.fromBounds(0, null, null, null).contains(0));
    try testing.expect(!ValuesCount.fromBounds(null, max, null, null).contains(max));
    try testing.expect(ValuesCount.fromBounds(null, null, null, max).contains(max));
    // The strictest of each side.
    const both = ValuesCount.fromBounds(5, 1, 0, 9);
    try testing.expect(!both.contains(1) and both.contains(2) and both.contains(4) and !both.contains(5));
}

test "a datetime string parses to Qdrant's instant in the forms it accepts, and to null otherwise" {
    try testing.expectEqual(@as(i64, 0), daysFromCivil(1970, 1, 1));
    try testing.expectEqual(@as(i64, -1), daysFromCivil(1969, 12, 31));
    try testing.expectEqual(@as(i64, 11017), daysFromCivil(2000, 3, 1));
    const midnight: i64 = 1_704_067_200; // 2024-01-01T00:00:00Z
    const Case = struct { s: []const u8, want: ?Instant };
    const at = struct {
        fn f(secs: i64, micros: u32) ?Instant {
            return .{ .secs = secs, .micros = micros };
        }
    }.f;
    const cases = [_]Case{
        // The naive forms are UTC.
        .{ .s = "2024-01-01", .want = at(midnight, 0) },
        .{ .s = "2024-01-01T10:20", .want = at(midnight + 37_200, 0) },
        .{ .s = "2024-01-01 10:20", .want = at(midnight + 37_200, 0) },
        .{ .s = "2024-01-01T10:20:30", .want = at(midnight + 37_230, 0) },
        .{ .s = "2024-01-01 10:20:30.5", .want = at(midnight + 37_230, 500_000) },
        // Fraction digits past six are truncated, not rounded.
        .{ .s = "2024-01-01T10:20:30.1234567", .want = at(midnight + 37_230, 123_456) },
        .{ .s = "2024-01-01T10:20:30.9999999999Z", .want = at(midnight + 37_230, 999_999) },
        // Offsets: the instant is UTC.
        .{ .s = "2024-01-01T10:20:30Z", .want = at(midnight + 37_230, 0) },
        .{ .s = "2024-01-01t10:20:30z", .want = at(midnight + 37_230, 0) },
        .{ .s = "2024-01-01T01:00:00+01:00", .want = at(midnight, 0) },
        .{ .s = "2023-12-31 22:30:00-0130", .want = at(midnight, 0) },
        .{ .s = "2024-01-01T05:00:00+05", .want = at(midnight, 0) },
        .{ .s = "2024-01-01t01:00:00+0100", .want = at(midnight, 0) },
        .{ .s = "2024-01-01T00:00:00+23:59", .want = at(midnight - 86_340, 0) },
        // A leap second sorts after :59.999999 and before the next minute.
        .{ .s = "2016-12-31T23:59:60Z", .want = at(1_483_228_799, 1_000_000) },
        .{ .s = "2016-12-31 23:59:60.25", .want = at(1_483_228_799, 1_250_000) },
        // Calendar checks.
        .{ .s = "2024-02-29", .want = at(daysFromCivil(2024, 2, 29) * 86_400, 0) },
        .{ .s = "2000-02-29", .want = at(daysFromCivil(2000, 2, 29) * 86_400, 0) },
        .{ .s = "2023-02-29", .want = null },
        .{ .s = "1900-02-29", .want = null },
        .{ .s = "2024-04-31", .want = null },
        .{ .s = "2024-13-01", .want = null },
        .{ .s = "2024-00-10", .want = null },
        .{ .s = "2024-01-01T24:00:00", .want = null },
        .{ .s = "2024-01-01T10:60", .want = null },
        .{ .s = "2024-01-01T10:20:61", .want = null },
        // Not forms Qdrant accepts: an offset without seconds, a naive time
        // with a lowercase `t`, `+HH` after a lowercase `t`, an empty
        // fraction, an offset of a day or more, trailing bytes.
        .{ .s = "2024-01-01T10:20Z", .want = null },
        .{ .s = "2024-01-01t10:20:30", .want = null },
        .{ .s = "2024-01-01t10:20:30+01", .want = null },
        .{ .s = "2024-01-01T10:20:30.", .want = null },
        .{ .s = "2024-01-01T10:20:30+24:00", .want = null },
        .{ .s = "2024-01-01T10:20:30+01:60", .want = null },
        .{ .s = "2024-01-01T10:20:30+1", .want = null },
        .{ .s = "2024-01-01T10:20:30Zx", .want = null },
        .{ .s = "2024-01-01x", .want = null },
        // Forms chrono takes that this does not (doc comment): never a match.
        .{ .s = "2024-01-01T10:20:30 Z", .want = null },
        .{ .s = "2024-01-01T10:20:30UTC", .want = null },
        .{ .s = " 2024-01-01", .want = null },
        .{ .s = "+12024-01-01", .want = null },
        .{ .s = "keyword", .want = null },
        .{ .s = "", .want = null },
    };
    for (cases) |c| {
        const got = parseDatetime(c.s);
        testing.expectEqual(c.want, got) catch |e| {
            std.debug.print("parsing \"{s}\"\n", .{c.s});
            return e;
        };
    }
}

test "a Timestamp bound is read as Qdrant reads it: microseconds, leap seconds, chrono's range" {
    try testing.expectEqual(Instant{ .secs = 10, .micros = 123_456 }, instantFromTimestamp(10, 123_456_789).?);
    // A negative `nanos` is 0.
    try testing.expectEqual(Instant{ .secs = 10, .micros = 0 }, instantFromTimestamp(10, -5).?);
    // A leap second only on second 59, and below two seconds.
    try testing.expectEqual(Instant{ .secs = 59, .micros = 1_500_000 }, instantFromTimestamp(59, 1_500_000_000).?);
    try testing.expectEqual(Instant{ .secs = -1, .micros = 1_000_000 }, instantFromTimestamp(-1, 1_000_000_000).?);
    try testing.expect(instantFromTimestamp(58, 1_000_000_000) == null);
    try testing.expect(instantFromTimestamp(59, 2_000_000_000) == null);
    // `NaiveDate::MIN` is -262143-01-01 (chrono's own serde test).
    try testing.expect(instantFromTimestamp(timestamp_min_secs, 0) != null);
    try testing.expect(instantFromTimestamp(timestamp_min_secs - 1, 0) == null);
    try testing.expect(instantFromTimestamp(timestamp_max_secs, 0) != null);
    try testing.expect(instantFromTimestamp(timestamp_max_secs + 1, 0) == null);
    try testing.expectEqual(@as(i64, 86_399), @mod(timestamp_max_secs, 86_400));
}

test "a datetime range matches parsed strings and any list element, never a number or a missing field" {
    var store = try Store.init(testing.allocator, 16);
    defer store.deinit(testing.allocator);
    var e1: [128]u8 = undefined;
    var m: [256]u8 = undefined;
    const strs = [_][]const u8{
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00+01:00", // the same instant
        "2024-01-01", // the same instant
        "", // replaced by a list below
        "not a date",
        "", // an integer below
        "", // missing below
        "2023-12-31T23:59:59.9999999Z",
    };
    for (strs, 0..) |s, i| {
        const entry = switch (i) {
            // ["garbage", "2024-06-01T12:00:00.500000Z"]
            3 => rawEntry(&e1, "d", "\x3a\x2a\x0a\x09\x22\x07garbage\x0a\x1d\x22\x1b2024-06-01T12:00:00.500000Z"),
            5 => entryBytes(&e1, "d", value_integer, 1_704_067_200),
            6 => entryBytes(&e1, "x", value_string, "2024-01-01"),
            else => entryBytes(&e1, "d", value_string, s),
        };
        try store.set(testing.allocator, @intCast(i), mapBytes(&m, &.{entry}), 3);
    }
    const midnight = Instant{ .secs = 1_704_067_200, .micros = 0 };
    const last_micro = Instant{ .secs = 1_704_067_199, .micros = 999_999 };
    const Case = struct { clause: Filter.Clause, r: DatetimeRange, want: [8]bool };
    const cases = [_]Case{
        .{ .clause = .must, .r = DatetimeRange.fromBounds(null, null, midnight, null), .want = .{ true, true, true, true, false, false, false, false } },
        .{ .clause = .must, .r = DatetimeRange.fromBounds(null, midnight, null, null), .want = .{ false, false, false, true, false, false, false, false } },
        .{ .clause = .must, .r = DatetimeRange.fromBounds(midnight, null, null, null), .want = .{ false, false, false, false, false, false, false, true } },
        // The payload's seventh fraction digit is truncated: it equals the
        // bound, which `lt` excludes and `lte` admits.
        .{ .clause = .must, .r = DatetimeRange.fromBounds(last_micro, null, null, null), .want = .{ false, false, false, false, false, false, false, false } },
        .{ .clause = .must, .r = DatetimeRange.fromBounds(null, null, null, last_micro), .want = .{ false, false, false, false, false, false, false, true } },
        // No bounds: every value that parses.
        .{ .clause = .must, .r = .{}, .want = .{ true, true, true, true, false, false, false, true } },
        // must_not: the rest, the unparsable, the number and the missing.
        .{ .clause = .must_not, .r = DatetimeRange.fromBounds(null, null, midnight, null), .want = .{ false, false, false, false, true, true, true, true } },
    };
    for (cases) |cs| {
        var f = Filter{};
        try f.add(cs.clause, .{ .key = "d", .match = .{ .datetime = cs.r } });
        for (cs.want, 0..) |want, off| {
            try testing.expectEqual(want, store.matches(@intCast(off), &f));
        }
    }
    // A match ORed with a datetime range, as one FieldCondition.
    var mo = Filter{};
    try mo.add(.must, .{ .key = "d", .match = .{ .keyword = "not a date" }, .or_match = .{ .datetime = DatetimeRange.fromBounds(null, midnight, null, null) } });
    for ([_]bool{ false, false, false, true, true, false, false, false }, 0..) |want, off| {
        try testing.expectEqual(want, store.matches(@intCast(off), &mo));
    }
    // No index answers a datetime or a shape condition, even on an indexed
    // field: the blobs decide.
    try store.createIndex(testing.allocator, "d", .keyword, 8);
    var bits: [1]u64 = undefined;
    var sel = Filter{};
    try sel.add(.must, .{ .key = "d", .match = .{ .datetime = .{} } });
    try testing.expect(store.select(&sel, &bits, 8) == null);
    var se = Filter{};
    try se.add(.must, .{ .key = "d", .match = .{ .is_empty = false } });
    try testing.expect(store.select(&se, &bits, 8) == null);
}

test "a range on an integer field is not read from its postings, and a float field answers it" {
    var store = try Store.init(testing.allocator, 4);
    defer store.deinit(testing.allocator);
    try store.createIndex(testing.allocator, "n", .integer, 0);
    var e: [64]u8 = undefined;
    var m: [128]u8 = undefined;
    // A double under an integer field has no posting, and a range admits it.
    try store.set(testing.allocator, 0, mapBytes(&m, &.{entryBytes(&e, "n", value_double, 2.5)}), 3);
    try store.set(testing.allocator, 1, mapBytes(&m, &.{entryBytes(&e, "n", value_integer, 2)}), 3);
    var bits: [1]u64 = undefined;
    const f = rangeFilter("n", Range.fromBounds(null, null, 2, null));
    try testing.expect(store.select(&f, &bits, 2) == null);
    try testing.expect(store.matches(0, &f));
    try testing.expect(store.matches(1, &f));
    // A float field answers it, and ingests both.
    try store.createIndex(testing.allocator, "f", .float, 2);
    try store.set(testing.allocator, 2, mapBytes(&m, &.{entryBytes(&e, "f", value_double, 2.5)}), 3);
    try store.set(testing.allocator, 3, mapBytes(&m, &.{entryBytes(&e, "f", value_integer, 2)}), 3);
    const g = rangeFilter("f", Range.fromBounds(null, null, 2, null));
    try testing.expectEqual(@as(usize, 2), store.select(&g, &bits, 4).?.count);
    try testing.expectEqual(@as(u64, 0b1100), bits[0]);
    // A match on a float field, or a match-or-range condition, reads no
    // entries either.
    var k = Filter{};
    try k.add(.must, .{ .key = "f", .match = .{ .integer = 2 } });
    try testing.expect(store.select(&k, &bits, 4) == null);
    var mr = Filter{};
    try mr.add(.must, .{ .key = "f", .match = .{ .integer = 2 }, .or_match = .{ .range = Range{} } });
    try testing.expect(store.select(&mr, &bits, 4) == null);
    try testing.expectEqual(Kind.float, store.field("f").?.kind());
    try testing.expectEqual(@as(u32, 3), Kind.float.schemaType());
    try testing.expectEqual(Kind.float, Kind.fromFieldType(2).?);
}

/// The float index against two references: a scan of the blobs (what an
/// unindexed filter answers) and a model of each point's current values.
const FloatModel = struct {
    const n = 6000;
    /// Up to three values per point, `lens` of them in use.
    vals: [][3]f64,
    lens: []u8,
    /// Every non-NaN value the index was given, stale ones included.
    entries: std.ArrayList(FloatEntry) = .empty,
    store: Store,
    indexed: bool = false,
    rnd: std.Random,

    fn init(rnd: std.Random) !FloatModel {
        const vals = try testing.allocator.alloc([3]f64, n);
        errdefer testing.allocator.free(vals);
        const lens = try testing.allocator.alloc(u8, n);
        errdefer testing.allocator.free(lens);
        @memset(lens, 0);
        return .{ .vals = vals, .lens = lens, .store = try Store.init(testing.allocator, n), .rnd = rnd };
    }

    fn deinit(self: *FloatModel) void {
        testing.allocator.free(self.vals);
        testing.allocator.free(self.lens);
        self.entries.deinit(testing.allocator);
        self.store.deinit(testing.allocator);
    }

    /// A value from a small domain, so ties, integer/double equality, the
    /// bounds' edges, `-0.0` and NaN all occur.
    fn value(self: *FloatModel) Num {
        const base: i64 = self.rnd.intRangeAtMost(i64, -40, 40);
        return switch (self.rnd.uintLessThan(u8, 10)) {
            0...4 => .{ .int = base },
            5...7 => .{ .dbl = @as(f64, @floatFromInt(base)) + 0.5 },
            8 => .{ .dbl = @floatFromInt(base) },
            else => .{ .dbl = if (base == 0) std.math.nan(f64) else -0.0 },
        };
    }

    fn write(self: *FloatModel, off: u32) !void {
        var nums: [3]Num = undefined;
        const count = self.rnd.uintLessThan(u8, 4);
        var e1: [128]u8 = undefined;
        var e2: [64]u8 = undefined;
        var m: [256]u8 = undefined;
        if (count == 0) {
            // A cleared payload, or one without the field.
            if (self.rnd.boolean()) {
                try self.store.set(testing.allocator, off, &.{}, 3);
            } else {
                try self.store.set(testing.allocator, off, mapBytes(&m, &.{entryBytes(&e2, "s", value_string, "z")}), 3);
            }
            self.lens[off] = 0;
            return;
        }
        for (nums[0..count], 0..) |*x, i| {
            x.* = self.value();
            self.vals[off][i] = switch (x.*) {
                .int => |v| @floatFromInt(v),
                .dbl => |v| v,
            };
            if (self.indexed and !std.math.isNan(self.vals[off][i])) {
                try self.entries.append(testing.allocator, .{ .v = self.vals[off][i], .off = off });
            }
        }
        self.lens[off] = count;
        const as_list = count > 1 or self.rnd.boolean();
        const entry = numEntry(&e1, "x", nums[0..count], as_list);
        try self.store.set(testing.allocator, off, mapBytes(&m, &.{ entryBytes(&e2, "s", value_string, "z"), entry }), 3);
    }

    fn createIndex(self: *FloatModel, written: u32) !void {
        try self.store.createIndex(testing.allocator, "x", .float, written);
        self.indexed = true;
        for (0..written) |off| {
            for (self.vals[off][0..self.lens[off]]) |v| {
                if (!std.math.isNan(v)) try self.entries.append(testing.allocator, .{ .v = v, .off = @intCast(off) });
            }
        }
    }

    fn modelMatches(self: *const FloatModel, off: usize, r: Range) bool {
        for (self.vals[off][0..self.lens[off]]) |v| if (r.contains(v)) return true;
        return false;
    }

    fn randomRange(self: *FloatModel) Range {
        var b: [4]?f64 = .{ null, null, null, null };
        for (&b) |*x| {
            if (self.rnd.uintLessThan(u8, 3) == 0) {
                const base: f64 = @floatFromInt(self.rnd.intRangeAtMost(i64, -45, 45));
                x.* = if (self.rnd.boolean()) base else base + 0.5;
            }
        }
        if (self.rnd.uintLessThan(u8, 50) == 0) b[self.rnd.uintLessThan(usize, 4)] = std.math.nan(f64);
        return Range.fromBounds(b[0], b[1], b[2], b[3]);
    }

    fn check(self: *FloatModel, written: u32, queries: usize) !void {
        var bits: [(n + 63) / 64]u64 = undefined;
        const field = self.store.field("x").?;
        const middle = Range.fromBounds(null, null, -10, 10);
        for (0..queries) |q| {
            const r = self.randomRange();
            const bound: usize = if (q % 4 == 0) self.rnd.uintAtMost(usize, written) else written;
            // The planner's count is the index's entries in range.
            var want_entries: usize = 0;
            for (self.entries.items) |e| {
                if (r.contains(e.v)) want_entries += 1;
            }
            try testing.expectEqual(want_entries, Store.postingLen(field, .{ .range = r }).?);
            // One condition (the trusted path while nothing was rewritten),
            // then the same range with a must_not beside it (verified).
            var f = rangeFilter("x", r);
            for (0..2) |pass| {
                if (pass == 1) try f.add(.must_not, .{ .key = "x", .match = .{ .range = middle } });
                const sel = self.store.select(&f, &bits, bound).?;
                var want: usize = 0;
                for (0..bound) |off| {
                    const scan = self.store.matches(@intCast(off), &f);
                    const model = self.modelMatches(off, r) and
                        (pass == 0 or !self.modelMatches(off, middle));
                    try testing.expectEqual(model, scan);
                    try testing.expectEqual(scan, testBit(&bits, @intCast(off)));
                    want += @intFromBool(scan);
                }
                try testing.expectEqual(want, sel.count);
            }
        }
    }
};

test "the float index counts and selects what a blob scan admits, across merges, overwrites and clears" {
    var prng = std.Random.DefaultPrng.init(0xf10a7);
    var model = try FloatModel.init(prng.random());
    defer model.deinit();
    const n = FloatModel.n;
    // A backfilled first part, then appends: past `tail_max` entries, so
    // the tail has been merged at least once and holds the latest writes.
    for (0..1000) |off| try model.write(@intCast(off));
    try model.createIndex(1000);
    for (1000..n) |off| try model.write(@intCast(off));
    const fi = &model.store.field("x").?.index.float;
    try testing.expect(fi.sorted.items.len >= FloatIndex.tail_max);
    try testing.expect(fi.tail.items.len > 0);
    try testing.expect(std.sort.isSorted(FloatEntry, fi.sorted.items, {}, FloatEntry.lessThan));
    try testing.expect(model.store.exact_postings.load(.acquire));
    try model.check(n, 150);
    // Overwrites and clears leave stale entries the verification drops.
    for (0..1500) |_| try model.write(model.rnd.uintLessThan(u32, n));
    try testing.expect(!model.store.exact_postings.load(.acquire));
    try testing.expect(std.sort.isSorted(FloatEntry, fi.sorted.items, {}, FloatEntry.lessThan));
    try model.check(n, 150);
}

test "a JSON path parses as Qdrant's json_path parses it, and nothing else does" {
    var buf: [max_path_steps]PathStep = undefined;
    const Ok = struct { raw: []const u8, first: []const u8, steps: []const PathStep };
    const oks = [_]Ok{
        .{ .raw = "a", .first = "a", .steps = &.{} },
        .{ .raw = "foo_123-Bar", .first = "foo_123-Bar", .steps = &.{} },
        .{ .raw = "a.b", .first = "a", .steps = &.{.{ .key = "b" }} },
        .{ .raw = "a[]", .first = "a", .steps = &.{.wildcard} },
        .{ .raw = "a[0]", .first = "a", .steps = &.{.{ .index = 0 }} },
        .{ .raw = "a[007]", .first = "a", .steps = &.{.{ .index = 7 }} },
        .{ .raw = "a[].b[2].c", .first = "a", .steps = &.{ .wildcard, .{ .key = "b" }, .{ .index = 2 }, .{ .key = "c" } } },
        // parse.rs's own test case.
        .{ .raw = "foo[1][50].bar-baz[].\"qux[.]quux\"", .first = "foo", .steps = &.{ .{ .index = 1 }, .{ .index = 50 }, .{ .key = "bar-baz" }, .wildcard, .{ .key = "qux[.]quux" } } },
        // Quoted keys, the first one too, and an empty one.
        .{ .raw = "\"a.b\"", .first = "a.b", .steps = &.{} },
        .{ .raw = "\"a\"", .first = "a", .steps = &.{} },
        .{ .raw = "\"\"", .first = "", .steps = &.{} },
        .{ .raw = "a.\"b c\"[]", .first = "a", .steps = &.{ .{ .key = "b c" }, .wildcard } },
        // Non-ASCII alphanumerics are raw key characters.
        .{ .raw = "été.ü", .first = "été", .steps = &.{.{ .key = "ü" }} },
    };
    for (oks) |ok| {
        const p = try parsePath(ok.raw, &buf);
        try testing.expectEqualStrings(ok.first, p.first);
        try testing.expectEqual(ok.steps.len, p.steps);
        for (ok.steps, buf[0..p.steps]) |want, got| try testing.expect(want.eql(got));
    }
    const bad = [_][]const u8{
        "",      ".a",   "a.",   "a..b", "a b",   " a",                      "a ",
        "[0]",   "[]",   "a[",   "a[0",  "a]",    "a[x]",                    "a[-1]",
        "a[ 0]", "a.[]", "a[]b", "\"a",  "a.\"b", "\"a\\\"b\"",              "\"a\\b\"",
        "a.b.",  "a/b",  "a:b",  "a,b",  "a[0]]", "a[99999999999999999999]",
    };
    for (bad) |raw| {
        testing.expectError(error.InvalidPath, parsePath(raw, &buf)) catch |e| {
            std.debug.print("path {s}\n", .{raw});
            return e;
        };
    }
    try testing.expectError(error.PathTooLong, parsePath(&wildcardPath(max_path_steps + 1), &buf));
    _ = try parsePath(&wildcardPath(max_path_steps), &buf);
}

/// `a` then `k` wildcard steps, for the tests of the step bound.
pub fn wildcardPath(comptime k: usize) [1 + 2 * k]u8 {
    var s: [1 + 2 * k]u8 = undefined;
    s[0] = 'a';
    for (0..k) |i| {
        s[1 + 2 * i] = '[';
        s[2 + 2 * i] = ']';
    }
    return s;
}

/// Hand-built protobuf `Value`s and payload maps, in an arena.
const Vb = struct {
    a: std.mem.Allocator,

    fn wrap(self: Vb, field: u32, body: []const u8) []const u8 {
        var head: [11]u8 = undefined;
        head[0] = @intCast(field << 3 | 2);
        const n = writeVarint(head[1..], body.len);
        return std.mem.concat(self.a, u8, &.{ head[0 .. 1 + n], body }) catch @panic("oom");
    }

    fn str(self: Vb, s: []const u8) []const u8 {
        return self.wrap(value_string, s);
    }

    fn int(self: Vb, v: i64) []const u8 {
        var b: [11]u8 = undefined;
        b[0] = value_integer << 3;
        const n = writeVarint(b[1..], @bitCast(v));
        return self.a.dupe(u8, b[0 .. 1 + n]) catch @panic("oom");
    }

    fn dbl(self: Vb, v: f64) []const u8 {
        var b: [9]u8 = undefined;
        b[0] = value_double << 3 | 1;
        std.mem.writeInt(u64, b[1..9], @bitCast(v), .little);
        return self.a.dupe(u8, &b) catch @panic("oom");
    }

    const nul: []const u8 = "\x08\x00";

    fn list(self: Vb, elems: []const []const u8) []const u8 {
        var body: []const u8 = &.{};
        for (elems) |e| body = std.mem.concat(self.a, u8, &.{ body, self.wrap(1, e) }) catch @panic("oom");
        return self.wrap(value_list, body);
    }

    fn entries(self: Vb, field: u32, kvs: []const [2][]const u8) []const u8 {
        var body: []const u8 = &.{};
        for (kvs) |kv| {
            const entry = std.mem.concat(self.a, u8, &.{ self.wrap(entry_key, kv[0]), self.wrap(entry_value, kv[1]) }) catch @panic("oom");
            body = std.mem.concat(self.a, u8, &.{ body, self.wrap(field, entry) }) catch @panic("oom");
        }
        return body;
    }

    fn obj(self: Vb, kvs: []const [2][]const u8) []const u8 {
        return self.wrap(value_struct, self.entries(1, kvs));
    }

    /// A `PointStruct.payload` (field 3) holding `kvs`.
    fn payload(self: Vb, kvs: []const [2][]const u8) []const u8 {
        return self.entries(3, kvs);
    }
};

/// The payloads the path tests resolve against, by point offset.
fn pathShapes(v: Vb) [14][]const u8 {
    return .{
        // 0: {"a": {"b": "x"}}
        v.payload(&.{.{ "a", v.obj(&.{.{ "b", v.str("x") }}) }}),
        // 1: {"a": [{"b": "x"}, {"b": "y"}]}
        v.payload(&.{.{ "a", v.list(&.{ v.obj(&.{.{ "b", v.str("x") }}), v.obj(&.{.{ "b", v.str("y") }}) }) }}),
        // 2: {"a": [{"b": "y"}, {"c": 1}]}
        v.payload(&.{.{ "a", v.list(&.{ v.obj(&.{.{ "b", v.str("y") }}), v.obj(&.{.{ "c", v.int(1) }}) }) }}),
        // 3: {"a": {"b": ["x", "z"]}}
        v.payload(&.{.{ "a", v.obj(&.{.{ "b", v.list(&.{ v.str("x"), v.str("z") }) }}) }}),
        // 4: {"a": {"b": null}}
        v.payload(&.{.{ "a", v.obj(&.{.{ "b", Vb.nul }}) }}),
        // 5: {"a": {"c": 1}}
        v.payload(&.{.{ "a", v.obj(&.{.{ "c", v.int(1) }}) }}),
        // 6: {"a": [{"b": []}, {"b": 1}]}
        v.payload(&.{.{ "a", v.list(&.{ v.obj(&.{.{ "b", v.list(&.{}) }}), v.obj(&.{.{ "b", v.int(1) }}) }) }}),
        // 7: {"a": [1, null]}
        v.payload(&.{.{ "a", v.list(&.{ v.int(1), Vb.nul }) }}),
        // 8: {"a": "x"}
        v.payload(&.{.{ "a", v.str("x") }}),
        // 9: {"o": 1}
        v.payload(&.{.{ "o", v.int(1) }}),
        // 10: {"a": [[{"b": "x"}]]}
        v.payload(&.{.{ "a", v.list(&.{v.list(&.{v.obj(&.{.{ "b", v.str("x") }})})}) }}),
        // 11: {"a": [{"b": "2024-01-02"}, {"b": 2.5}]}
        v.payload(&.{.{ "a", v.list(&.{ v.obj(&.{.{ "b", v.str("2024-01-02") }}), v.obj(&.{.{ "b", v.dbl(2.5) }}) }) }}),
        // 12: {"a.b": "x"}
        v.payload(&.{.{ "a.b", v.str("x") }}),
        // 13: {"a": {"b": {"c": 5}}}
        v.payload(&.{.{ "a", v.obj(&.{.{ "b", v.obj(&.{.{ "c", v.int(5) }}) }}) }}),
    };
}

const PathCase = struct {
    path: []const u8,
    clause: Filter.Clause = .must,
    match: Match,
    /// Per point of `pathShapes`, `T` where the filter admits it.
    want: *const [14]u8,
};

const new_year = Instant{ .secs = 1_704_067_200, .micros = 0 };

/// Each expectation is Qdrant's `value_get` (json_path/mod.rs:320) then
/// `check_field_condition` (query_checker.rs:203) worked by hand: a key step
/// only into an object, `[i]` and `[]` only into a list, any value
/// matching, and `check_empty` when the path reaches none.
const path_cases = [_]PathCase{
    // Columns: 0 {"a":{"b":"x"}}, 1 [{"b":"x"},{"b":"y"}], 2 [{"b":"y"},{"c":1}],
    // 3 {"b":["x","z"]}, 4 {"b":null}, 5 {"c":1}, 6 [{"b":[]},{"b":1}], 7 [1,null],
    // 8 "x", 9 no "a", 10 [[{"b":"x"}]], 11 [{"b":"2024-01-02"},{"b":2.5}],
    // 12 {"a.b":"x"}, 13 {"b":{"c":5}}.
    // `a.b` reads an object's field, a list's element only through `[]`.
    .{ .path = "a.b", .match = .{ .keyword = "x" }, .want = "TFFTFFFFFFFFFF" },
    .{ .path = "a[].b", .match = .{ .keyword = "x" }, .want = "FTFFFFFFFFFFFF" },
    .{ .path = "a[].b", .clause = .must_not, .match = .{ .keyword = "x" }, .want = "TFTTTTTTTTTTTT" },
    .{ .path = "a[].b", .clause = .should, .match = .{ .keyword = "y" }, .want = "FTTFFFFFFFFFFF" },
    .{ .path = "a[0].b", .match = .{ .keyword = "y" }, .want = "FFTFFFFFFFFFFF" },
    .{ .path = "a[1].b", .match = .{ .keyword = "y" }, .want = "FTFFFFFFFFFFFF" },
    .{ .path = "a[5].b", .match = .{ .keyword = "y" }, .want = "FFFFFFFFFFFFFF" },
    .{ .path = "a[].b", .match = .{ .keywords = "\x0a\x01q\x0a\x01y" }, .want = "FTTFFFFFFFFFFF" },
    .{ .path = "a[].c", .match = .{ .integer = 1 }, .want = "FFTFFFFFFFFFFF" },
    .{ .path = "a.b.c", .match = .{ .integers = "\x0a\x02\x05\x06" }, .want = "FFFFFFFFFFFFFT" },
    // A list at the path's end is one value, matched per element.
    .{ .path = "a[]", .match = .{ .integer = 1 }, .want = "FFFFFFFTFFFFFF" },
    .{ .path = "a", .match = .{ .integer = 1 }, .want = "FFFFFFFTFFFFFF" },
    .{ .path = "a[].b", .match = .{ .range = Range.fromBounds(null, null, 1, null) }, .want = "FFFFFFTFFFFTFF" },
    .{ .path = "a[].b", .match = .{ .datetime = DatetimeRange.fromBounds(null, null, new_year, null) }, .want = "FFFFFFFFFFFTFF" },
    // A quoted key is one top-level key.
    .{ .path = "\"a.b\"", .match = .{ .keyword = "x" }, .want = "FFFFFFFFFFFFTF" },
    // values_count: any one value's count in range, 0 when none.
    .{ .path = "a[].b", .match = .{ .values_count = ValuesCount.fromBounds(1, null, null, null) }, .want = "TFFTTTTTTTTFTT" },
    .{ .path = "a[].b", .match = .{ .values_count = ValuesCount.fromBounds(null, null, 2, null) }, .want = "FFFFFFFFFFFFFF" },
    .{ .path = "a.b", .match = .{ .values_count = ValuesCount.fromBounds(null, null, 2, null) }, .want = "FFFTFFFFFFFFFF" },
    .{ .path = "a[]", .match = .{ .values_count = ValuesCount.fromBounds(null, null, 1, 1) }, .want = "FTTFFFTTFFTTFF" },
    // FieldCondition.is_empty: any value empty, or none.
    .{ .path = "a[].b", .match = .{ .is_empty = true }, .want = "TFFTTTTTTTTFTT" },
    .{ .path = "a[].b", .match = .{ .is_empty = false }, .want = "FTTFFFTFFFFTFF" },
    .{ .path = "a[]", .match = .{ .is_empty = true }, .want = "TFFTTTFTTTFFTT" },
    // Condition.is_empty: every value empty, or none.
    .{ .path = "a[].b", .match = .all_empty, .want = "TFFTTTFTTTTFTT" },
    .{ .path = "a[]", .match = .all_empty, .want = "TFFTTTFFTTFFTT" },
    .{ .path = "a[]", .clause = .must_not, .match = .all_empty, .want = "FTTFFFTTFFTTFF" },
    // is_null: any value null or a list holding one; never when none.
    .{ .path = "a.b", .match = .{ .is_null = true }, .want = "FFFFTFFFFFFFFF" },
    .{ .path = "a.b", .match = .{ .is_null = false }, .want = "TTTTFTTTTTTTTT" },
    .{ .path = "a[]", .match = .{ .is_null = true }, .want = "FFFFFFFTFFFFFF" },
    .{ .path = "a", .match = .{ .is_null = true }, .want = "FFFFFFFTFFFFFF" },
};

test "a path resolves over nested objects and lists for every condition, as Qdrant's value_get and checks do" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const v = Vb{ .a = arena.allocator() };
    var store = try Store.init(testing.allocator, 16);
    defer store.deinit(testing.allocator);
    for (pathShapes(v), 0..) |p, i| try store.set(testing.allocator, @intCast(i), p, 3);
    for (path_cases) |cs| {
        var f = Filter{};
        try f.addPath(cs.clause, .{ .key = cs.path, .match = cs.match });
        for (cs.want, 0..) |w, off| {
            testing.expectEqual(w == 'T', store.matches(@intCast(off), &f)) catch |e| {
                std.debug.print("path {s} point {d}\n", .{ cs.path, off });
                return e;
            };
        }
    }
    // A filter holds several paths' steps side by side.
    var f = Filter{};
    try f.addPath(.must, .{ .key = "a[].b", .match = .{ .keyword = "y" } });
    try f.addPath(.must, .{ .key = "a[1].c", .match = .{ .integer = 1 } });
    try f.addPath(.must_not, .{ .key = "a[0].b", .match = .{ .keyword = "x" } });
    for (0..14) |off| try testing.expectEqual(off == 2, store.matches(@intCast(off), &f));
}

test "an index on a nested path holds what a condition on that path reads, and answers only that path" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const v = Vb{ .a = arena.allocator() };
    var store = try Store.init(testing.allocator, 64);
    defer store.deinit(testing.allocator);
    const shapes = pathShapes(v);
    for (shapes[0..7], 0..) |p, i| try store.set(testing.allocator, @intCast(i), p, 3);
    try store.createIndex(testing.allocator, "a[].b", .keyword, 7);
    try store.createIndex(testing.allocator, "a[].b[]", .float, 7);
    try store.createIndex(testing.allocator, "\"a\".b", .integer, 7);
    for (shapes[7..], 7..) |p, i| try store.set(testing.allocator, @intCast(i), p, 3);
    // The same path however spelled is one field; another kind on it is refused.
    try store.createIndex(testing.allocator, "\"a\"[].\"b\"", .keyword, 14);
    try testing.expectError(error.IndexKindMismatch, store.createIndex(testing.allocator, "a[].b", .integer, 14));
    try testing.expectError(error.InvalidPath, store.createIndex(testing.allocator, "a b", .keyword, 14));
    try testing.expectEqual(@as(usize, 3), store.fields.items.len);
    try testing.expectEqual(Kind.keyword, store.field("a[].\"b\"").?.kind());
    try testing.expect(store.field("a b") == null);

    var bits: [1]u64 = undefined;
    const Sel = struct { path: []const u8, match: Match, indexed: bool };
    const sels = [_]Sel{
        .{ .path = "a[].b", .match = .{ .keyword = "y" }, .indexed = true },
        .{ .path = "a[].b", .match = .{ .keyword = "x" }, .indexed = true },
        .{ .path = "a.b", .match = .{ .keyword = "x" }, .indexed = false },
        .{ .path = "a[]", .match = .{ .keyword = "x" }, .indexed = false },
        .{ .path = "a[].b[]", .match = .{ .range = Range.fromBounds(null, null, 1, null) }, .indexed = true },
        .{ .path = "a.b", .match = .{ .integer = 1 }, .indexed = true },
    };
    for (sels) |s| {
        for (0..2) |pass| {
            var f = Filter{};
            try f.addPath(.must, .{ .key = s.path, .match = s.match });
            // The second pass verifies every entry, the first trusts them.
            if (pass == 1) try f.addPath(.must_not, .{ .key = "o", .match = .{ .integer = 2 } });
            const sel = store.select(&f, &bits, 14);
            try testing.expectEqual(s.indexed, sel != null);
            if (sel == null) continue;
            var want: u64 = 0;
            for (0..14) |off| {
                if (store.matches(@intCast(off), &f)) want |= @as(u64, 1) << @intCast(off);
            }
            try testing.expectEqual(want, bits[0]);
            try testing.expectEqual(@as(usize, @popCount(want)), sel.?.count);
        }
    }
    try testing.expect(store.exact_postings.load(.acquire));
    // `a[].b[]` reaches the elements of a list under `b` only: 6's `[]`.
    try testing.expectEqual(@as(usize, 0), store.field("a[].b[]").?.points);
}

/// The datetime index against a scan of the blobs and a model of each
/// point's current instants, as `FloatModel` checks the float index.
const DatetimeModel = struct {
    const n = 6000;
    const DtItem = SortedIndex(i64, MicrosRange).Item;
    vals: [][3]Instant,
    lens: []u8,
    /// Every instant the index was given, stale ones included.
    entries: std.ArrayList(DtItem) = .empty,
    store: Store,
    indexed: bool = false,
    /// Whether writes may hold leap seconds.
    leap: bool = false,
    rnd: std.Random,
    arena: std.heap.ArenaAllocator,

    fn init(rnd: std.Random) !DatetimeModel {
        const vals = try testing.allocator.alloc([3]Instant, n);
        errdefer testing.allocator.free(vals);
        const lens = try testing.allocator.alloc(u8, n);
        errdefer testing.allocator.free(lens);
        @memset(lens, 0);
        return .{
            .vals = vals,
            .lens = lens,
            .store = try Store.init(testing.allocator, n),
            .rnd = rnd,
            .arena = std.heap.ArenaAllocator.init(testing.allocator),
        };
    }

    fn deinit(self: *DatetimeModel) void {
        testing.allocator.free(self.vals);
        testing.allocator.free(self.lens);
        self.entries.deinit(testing.allocator);
        self.store.deinit(testing.allocator);
        self.arena.deinit();
    }

    /// An instant in a three-minute window, so ties and the bounds' edges
    /// occur, written in one of the forms `parseDatetime` takes.
    fn datetimeString(self: *DatetimeModel) []const u8 {
        const a = self.arena.allocator();
        const minute = self.rnd.uintLessThan(u32, 3);
        var second = self.rnd.uintLessThan(u32, 60);
        if (self.leap and self.rnd.uintLessThan(u8, 10) == 0) second = 60;
        const micros: u32 = switch (self.rnd.uintLessThan(u8, 4)) {
            0 => 0,
            1 => 500_000,
            2 => 999_999,
            else => self.rnd.uintLessThan(u32, 1_000_000),
        };
        return switch (self.rnd.uintLessThan(u8, 3)) {
            0 => std.fmt.allocPrint(a, "2024-01-01T00:{d:0>2}:{d:0>2}.{d:0>6}Z", .{ minute, second, micros }),
            1 => std.fmt.allocPrint(a, "2024-01-01T01:{d:0>2}:{d:0>2}.{d:0>6}+01:00", .{ minute, second, micros }),
            else => std.fmt.allocPrint(a, "2024-01-01 00:{d:0>2}:{d:0>2}.{d:0>6}", .{ minute, second, micros }),
        } catch @panic("oom");
    }

    fn write(self: *DatetimeModel, off: u32) !void {
        const v = Vb{ .a = self.arena.allocator() };
        const count = self.rnd.uintLessThan(u8, 4);
        if (count == 0) {
            // A cleared payload, or one without the field.
            if (self.rnd.boolean()) {
                try self.store.set(testing.allocator, off, &.{}, 3);
            } else {
                try self.store.set(testing.allocator, off, v.payload(&.{.{ "s", v.str("z") }}), 3);
            }
            self.lens[off] = 0;
            return;
        }
        var elems: [4][]const u8 = undefined;
        for (0..count) |i| {
            const s = self.datetimeString();
            const at = parseDatetime(s).?;
            self.vals[off][i] = at;
            if (self.indexed) try self.entries.append(testing.allocator, .{ .v = microsKey(at), .off = off });
            elems[i] = v.str(s);
        }
        self.lens[off] = count;
        // Beside the instants, a string and a number no range admits.
        var len: usize = count;
        if (self.rnd.uintLessThan(u8, 4) == 0) {
            elems[len] = if (self.rnd.boolean()) v.str("not a date") else v.int(1_704_067_200);
            len += 1;
        }
        const value = if (len == 1 and self.rnd.boolean()) elems[0] else v.list(elems[0..len]);
        try self.store.set(testing.allocator, off, v.payload(&.{ .{ "s", v.str("z") }, .{ "d", value } }), 3);
    }

    fn createIndex(self: *DatetimeModel, written: u32) !void {
        try self.store.createIndex(testing.allocator, "d", .datetime, written);
        self.indexed = true;
        for (0..written) |off| {
            for (self.vals[off][0..self.lens[off]]) |at| {
                try self.entries.append(testing.allocator, .{ .v = microsKey(at), .off = @intCast(off) });
            }
        }
    }

    fn modelMatches(self: *const DatetimeModel, off: usize, r: DatetimeRange) bool {
        for (self.vals[off][0..self.lens[off]]) |at| if (r.contains(at)) return true;
        return false;
    }

    /// Bounds in and around the window, some a leap second's.
    fn randomRange(self: *DatetimeModel) DatetimeRange {
        var b: [4]?Instant = .{ null, null, null, null };
        for (&b) |*x| {
            if (self.rnd.uintLessThan(u8, 3) != 0) continue;
            const secs = new_year.secs - 2 + @as(i64, self.rnd.uintLessThan(u32, 185));
            const micros: u32 = switch (self.rnd.uintLessThan(u8, 5)) {
                0 => 0,
                1 => 500_000,
                2 => 999_999,
                3 => self.rnd.uintLessThan(u32, 1_000_000),
                else => if (@mod(secs, 60) == 59) 1_000_000 + self.rnd.uintLessThan(u32, 1_000_000) else 0,
            };
            x.* = .{ .secs = secs, .micros = micros };
        }
        return DatetimeRange.fromBounds(b[0], b[1], b[2], b[3]);
    }

    fn check(self: *DatetimeModel, written: u32, queries: usize) !void {
        var bits: [(n + 63) / 64]u64 = undefined;
        const field = self.store.field("d").?;
        const middle = DatetimeRange.fromBounds(null, null, .{ .secs = new_year.secs + 60, .micros = 0 }, .{ .secs = new_year.secs + 120, .micros = 0 });
        for (0..queries) |q| {
            const r = self.randomRange();
            const bound: usize = if (q % 4 == 0) self.rnd.uintAtMost(usize, written) else written;
            // The planner's count is the index's entries in range.
            var want_entries: usize = 0;
            const mr = MicrosRange.of(r);
            for (self.entries.items) |e| {
                if (mr.contains(e.v)) want_entries += 1;
            }
            try testing.expectEqual(want_entries, Store.postingLen(field, .{ .datetime = r }).?);
            var f = Filter{};
            try f.add(.must, .{ .key = "d", .match = .{ .datetime = r } });
            for (0..2) |pass| {
                if (pass == 1) try f.add(.must_not, .{ .key = "d", .match = .{ .datetime = middle } });
                const sel = self.store.select(&f, &bits, bound).?;
                var want: usize = 0;
                for (0..bound) |off| {
                    const scan = self.store.matches(@intCast(off), &f);
                    const model = self.modelMatches(off, r) and
                        (pass == 0 or !self.modelMatches(off, middle));
                    try testing.expectEqual(model, scan);
                    try testing.expectEqual(scan, testBit(&bits, @intCast(off)));
                    want += @intFromBool(scan);
                }
                try testing.expectEqual(want, sel.count);
            }
        }
    }
};

test "the datetime index counts and selects what a blob scan admits, across merges, overwrites, clears and leap seconds" {
    var prng = std.Random.DefaultPrng.init(0xda7e);
    var model = try DatetimeModel.init(prng.random());
    defer model.deinit();
    const n = DatetimeModel.n;
    for (0..1000) |off| try model.write(@intCast(off));
    try model.createIndex(1000);
    for (1000..n) |off| try model.write(@intCast(off));
    const field = model.store.field("d").?;
    try testing.expectEqual(Kind.datetime, field.kind());
    const di = &field.index.datetime;
    try testing.expect(di.sorted.sorted.items.len >= SortedIndex(i64, MicrosRange).tail_max);
    try testing.expect(di.sorted.tail.items.len > 0);
    try testing.expect(!di.leap);
    // Nothing rewritten and no leap second: the trusted path, exact.
    try testing.expect(model.store.exact_postings.load(.acquire));
    try model.check(n, 150);
    // Overwrites and clears leave stale entries the verification drops.
    for (0..1500) |_| try model.write(model.rnd.uintLessThan(u32, n));
    try testing.expect(!model.store.exact_postings.load(.acquire));
    try model.check(n, 150);
    // Leap seconds: keys tie, so every entry is verified.
    model.leap = true;
    for (0..1500) |_| try model.write(model.rnd.uintLessThan(u32, n));
    try testing.expect(di.leap);
    try testing.expect(std.sort.isSorted(DatetimeModel.DtItem, di.sorted.sorted.items, {}, DatetimeModel.DtItem.lessThan));
    try model.check(n, 150);
}

test "a leap second's key ties with the second's last microsecond, and a range over keys is a superset" {
    const leap = parseDatetime("2016-12-31T23:59:60.25Z").?;
    const last = parseDatetime("2016-12-31T23:59:59.999999Z").?;
    const next = parseDatetime("2017-01-01T00:00:00Z").?;
    try testing.expectEqual(std.math.Order.lt, last.order(leap));
    try testing.expectEqual(std.math.Order.lt, leap.order(next));
    try testing.expectEqual(microsKey(last), microsKey(leap));
    try testing.expectEqual(microsKey(next), microsKey(leap) + 1);
    // `gt: last` excludes `last` and admits `leap`; over keys it is made
    // inclusive only when a bound is a leap second.
    const r = MicrosRange.of(DatetimeRange.fromBounds(null, last, null, null));
    try testing.expect(r.exact and !r.contains(microsKey(last)));
    const rl = MicrosRange.of(DatetimeRange.fromBounds(leap, null, null, null));
    try testing.expect(!rl.exact and rl.contains(microsKey(leap)) and rl.contains(microsKey(last)));
    const open = MicrosRange.of(.{});
    try testing.expect(open.exact and open.contains(std.math.minInt(i64)) and open.contains(std.math.maxInt(i64)));
    try testing.expectEqual(@as(u32, 7), Kind.datetime.schemaType());
    try testing.expectEqual(Kind.datetime, Kind.fromFieldType(6).?);
}
