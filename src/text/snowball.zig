//! The Snowball runtime, ported from `qdrant-rust-stemmers` 1.2.2
//! (`src/snowball/snowball_env.rs`), the crate Qdrant links for its text
//! index's stemmer (decisions.md, 2026-10-07: strawmANN must stem token for
//! token as Qdrant does).
//!
//! Every operation is the crate's, byte for byte, quirks included: `eq_s`
//! compares against the rest of the word rather than up to `limit`, the
//! groupings decode the code point under the cursor, and the among searches
//! are the generated binary searches over byte strings. The one departure is
//! storage: the word lives in a caller-provided buffer instead of a `Cow`, so
//! stemming allocates nothing. The English algorithm never lengthens a word
//! by more than the one `e` Step 1b inserts, which `capacity_slack` covers.

const std = @import("std");

/// How many bytes past the input a stem may need.
pub const capacity_slack = 8;

/// One entry of a generated among table: the string, the index of the entry
/// it is a suffix (or prefix) of, and the result the search returns. None of
/// the English tables carries a routine.
pub const Among = struct {
    s: []const u8,
    substring_i: i32,
    result: i32,
};

pub const Env = struct {
    buf: []u8,
    len: usize,
    cursor: usize,
    limit: usize,
    limit_backward: usize,
    bra: usize,
    ket: usize,

    /// `word` copied into `buf`, which must hold `word.len + capacity_slack`.
    pub fn init(buf: []u8, word: []const u8) Env {
        std.debug.assert(buf.len >= word.len + capacity_slack);
        @memcpy(buf[0..word.len], word);
        return .{
            .buf = buf,
            .len = word.len,
            .cursor = 0,
            .limit = word.len,
            .limit_backward = 0,
            .bra = 0,
            .ket = word.len,
        };
    }

    pub fn current(self: *const Env) []const u8 {
        return self.buf[0..self.len];
    }

    /// Rust's `str::is_char_boundary`.
    fn isCharBoundary(self: *const Env, i: usize) bool {
        if (i == 0 or i == self.len) return true;
        if (i > self.len) return false;
        return (self.buf[i] & 0xC0) != 0x80;
    }

    /// The code point starting at byte `i`, or null at the end. Only its
    /// order against the groupings' ASCII bounds matters, so a non-ASCII
    /// character is any value above them.
    fn charAt(self: *const Env, i: usize) ?u32 {
        if (i >= self.len) return null;
        const b = self.buf[i];
        if (b < 0x80) return b;
        return std.unicode.utf8Decode(self.buf[i..][0 .. std.unicode.utf8ByteSequenceLength(b) catch return 0x80]) catch 0x80;
    }

    fn replaceS(self: *Env, bra: usize, ket: usize, s: []const u8) i64 {
        const adjustment: i64 = @as(i64, @intCast(s.len)) - (@as(i64, @intCast(ket)) - @as(i64, @intCast(bra)));
        const new_len: usize = @intCast(@as(i64, @intCast(self.len)) + adjustment);
        std.debug.assert(new_len <= self.buf.len);
        // Move the tail first, in the direction that does not overwrite it.
        const tail = self.len - ket;
        if (adjustment > 0) {
            std.mem.copyBackwards(u8, self.buf[bra + s.len ..][0..tail], self.buf[ket..][0..tail]);
        } else if (adjustment < 0) {
            std.mem.copyForwards(u8, self.buf[bra + s.len ..][0..tail], self.buf[ket..][0..tail]);
        }
        @memcpy(self.buf[bra..][0..s.len], s);
        self.len = new_len;
        self.limit = @intCast(@as(i64, @intCast(self.limit)) + adjustment);
        if (self.cursor >= ket) {
            self.cursor = @intCast(@as(i64, @intCast(self.cursor)) + adjustment);
        } else if (self.cursor > bra) {
            self.cursor = bra;
        }
        return adjustment;
    }

    pub fn eq_s(self: *Env, s: []const u8) bool {
        if (self.cursor >= self.limit) return false;
        if (!std.mem.startsWith(u8, self.current()[self.cursor..], s)) return false;
        self.cursor += s.len;
        while (!self.isCharBoundary(self.cursor)) self.cursor += 1;
        return true;
    }

    pub fn eq_s_b(self: *Env, s: []const u8) bool {
        if (@as(i64, @intCast(self.cursor)) - @as(i64, @intCast(self.limit_backward)) < @as(i64, @intCast(s.len))) return false;
        const at = self.cursor - s.len;
        if (!self.isCharBoundary(at) or !std.mem.startsWith(u8, self.current()[at..], s)) return false;
        self.cursor = at;
        return true;
    }

    pub fn slice_from(self: *Env, s: []const u8) bool {
        _ = self.replaceS(self.bra, self.ket, s);
        return true;
    }

    pub fn slice_del(self: *Env) bool {
        return self.slice_from("");
    }

    pub fn insert(self: *Env, bra: usize, ket: usize, s: []const u8) void {
        const adjustment = self.replaceS(bra, ket, s);
        if (bra <= self.bra) self.bra = @intCast(@as(i64, @intCast(self.bra)) + adjustment);
        if (bra <= self.ket) self.ket = @intCast(@as(i64, @intCast(self.ket)) + adjustment);
    }

    pub fn next_char(self: *Env) void {
        self.cursor += 1;
        while (!self.isCharBoundary(self.cursor)) self.cursor += 1;
    }

    pub fn previous_char(self: *Env) void {
        self.cursor -= 1;
        while (!self.isCharBoundary(self.cursor)) self.cursor -= 1;
    }

    pub fn byte_index_for_hop(self: *const Env, delta_in: i32) i32 {
        var delta = delta_in;
        if (delta > 0) {
            var res = self.cursor;
            while (delta > 0) {
                res += 1;
                delta -= 1;
                while (res <= self.len and !self.isCharBoundary(res)) res += 1;
            }
            return @intCast(res);
        } else if (delta < 0) {
            var res: i64 = @intCast(self.cursor);
            while (delta < 0) {
                res -= 1;
                delta += 1;
                while (res >= 0 and !self.isCharBoundary(@intCast(res))) res -= 1;
            }
            return @intCast(res);
        }
        return @intCast(self.cursor);
    }

    fn inGroupingAt(chars: []const u8, min: u32, max: u32, ch_in: u32) bool {
        if (ch_in > max or ch_in < min) return false;
        const ch = ch_in - min;
        return (chars[ch >> 3] & (@as(u8, 1) << @intCast(ch & 0x7))) != 0;
    }

    pub fn in_grouping(self: *Env, chars: []const u8, min: u32, max: u32) bool {
        if (self.cursor >= self.limit) return false;
        const ch = self.charAt(self.cursor) orelse return false;
        if (!inGroupingAt(chars, min, max, ch)) return false;
        self.next_char();
        return true;
    }

    pub fn in_grouping_b(self: *Env, chars: []const u8, min: u32, max: u32) bool {
        if (self.cursor <= self.limit_backward) return false;
        self.previous_char();
        const ch = self.charAt(self.cursor) orelse return false;
        self.next_char();
        if (!inGroupingAt(chars, min, max, ch)) return false;
        self.previous_char();
        return true;
    }

    pub fn out_grouping(self: *Env, chars: []const u8, min: u32, max: u32) bool {
        if (self.cursor >= self.limit) return false;
        const ch = self.charAt(self.cursor) orelse return false;
        if (inGroupingAt(chars, min, max, ch)) return false;
        self.next_char();
        return true;
    }

    pub fn out_grouping_b(self: *Env, chars: []const u8, min: u32, max: u32) bool {
        if (self.cursor <= self.limit_backward) return false;
        self.previous_char();
        const ch = self.charAt(self.cursor) orelse return false;
        self.next_char();
        if (inGroupingAt(chars, min, max, ch)) return false;
        self.previous_char();
        return true;
    }

    pub fn find_among(self: *Env, amongs: []const Among) i32 {
        var i: i32 = 0;
        var j: i32 = @intCast(amongs.len);
        const c = self.cursor;
        const l = self.limit;
        var common_i: usize = 0;
        var common_j: usize = 0;
        var first_key_inspected = false;
        while (true) {
            const k = i + ((j - i) >> 1);
            var diff: i32 = 0;
            var common = @min(common_i, common_j);
            const w = amongs[@intCast(k)];
            var lvar = common;
            while (lvar < w.s.len) : (lvar += 1) {
                if (c + common == l) {
                    diff = -1;
                    break;
                }
                diff = @as(i32, self.buf[c + common]) - @as(i32, w.s[lvar]);
                if (diff != 0) break;
                common += 1;
            }
            if (diff < 0) {
                j = k;
                common_j = common;
            } else {
                i = k;
                common_i = common;
            }
            if (j - i <= 1) {
                if (i > 0) break;
                if (j == i) break;
                if (first_key_inspected) break;
                first_key_inspected = true;
            }
        }
        while (true) {
            const w = amongs[@intCast(i)];
            if (common_i >= w.s.len) {
                self.cursor = c + w.s.len;
                return w.result;
            }
            i = w.substring_i;
            if (i < 0) return 0;
        }
    }

    pub fn find_among_b(self: *Env, amongs: []const Among) i32 {
        var i: i32 = 0;
        var j: i32 = @intCast(amongs.len);
        const c = self.cursor;
        const lb = self.limit_backward;
        var common_i: usize = 0;
        var common_j: usize = 0;
        var first_key_inspected = false;
        while (true) {
            const k = i + ((j - i) >> 1);
            var diff: i32 = 0;
            var common = @min(common_i, common_j);
            const w = amongs[@intCast(k)];
            // `for lvar in (0..w.len() - common).rev()`.
            var n = w.s.len - common;
            while (n > 0) {
                n -= 1;
                const lvar = n;
                if (c - common == lb) {
                    diff = -1;
                    break;
                }
                diff = @as(i32, self.buf[c - common - 1]) - @as(i32, w.s[lvar]);
                if (diff != 0) break;
                common += 1;
            }
            if (diff < 0) {
                j = k;
                common_j = common;
            } else {
                i = k;
                common_i = common;
            }
            if (j - i <= 1) {
                if (i > 0) break;
                if (j == i) break;
                if (first_key_inspected) break;
                first_key_inspected = true;
            }
        }
        while (true) {
            const w = amongs[@intCast(i)];
            if (common_i >= w.s.len) {
                self.cursor = c - w.s.len;
                return w.result;
            }
            i = w.substring_i;
            if (i < 0) return 0;
        }
    }
};
