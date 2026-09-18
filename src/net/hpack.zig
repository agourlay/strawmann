//! HPACK (RFC 7541), header compression for HTTP/2.
//!
//! §6.1 stages the transport as "M1: nghttp2 via Zig's C interop" then
//! "M6: hand-rolled `h2`, if and only if profiling shows framing/HPACK on the
//! critical path". **nghttp2 is not available on the build host** (no headers,
//! no pkg-config entry), so the native implementation is not an optimisation
//! brought forward, it is the only way to have a transport at all. The M6
//! design constraints therefore apply from the start, which is the cheaper
//! order anyway: they are structural, and retrofitting them is what M6 was
//! budgeted for.
//!
//! §6.1's design rules, all of which this file follows:
//!
//!   "Design it for zero per-request allocation: arena per stream reset on
//!    completion, fixed-size header table, HPACK encoding using static-table
//!    indices plus literal-without-indexing (always legal, avoids maintaining
//!    an encoder dynamic table)."
//!
//! ## Asymmetric by design
//!
//! **Decoding** must be complete: tonic (which `qdrant-client` uses) Huffman-
//! encodes header values and indexes into its dynamic table, so a server that
//! skipped either would fail on the first request. Full support is mandatory.
//!
//! **Encoding** is deliberately minimal: static-table indices where one fits,
//! literal-without-indexing otherwise, never Huffman, never inserting into a
//! dynamic table. That is always legal HPACK, it makes the encoder stateless,
//! and it removes an entire class of desynchronisation bug, an encoder
//! dynamic table that drifts out of step with the peer's decoder corrupts every
//! subsequent header block on the connection, and the failure appears far from
//! its cause. Response headers here are a handful of short, mostly-static
//! fields, so the bytes saved would not repay that risk.

const std = @import("std");

pub const Error = error{
    /// Integer or string ran past the end of the block.
    Truncated,
    /// An HPACK integer that does not fit the target width.
    IntegerOverflow,
    /// An index that names no static or dynamic entry.
    InvalidIndex,
    /// Huffman bits that decode to no symbol, or an invalid EOS.
    InvalidHuffman,
    /// A dynamic table size update larger than the peer's SETTINGS allow.
    TableSizeExceeded,
    /// Header block larger than the fixed decode arena.
    HeaderTooLarge,
    /// More headers in one block than the fixed array holds.
    TooManyHeaders,
};

// =========================================================================
// Static table (RFC 7541 Appendix A)
// =========================================================================

pub const StaticEntry = struct { name: []const u8, value: []const u8 };

/// Indices are 1-based on the wire; this array is 0-based, so wire index `i`
/// is `static_table[i - 1]`.
pub const static_table = [_]StaticEntry{
    .{ .name = ":authority", .value = "" }, // 1
    .{ .name = ":method", .value = "GET" }, // 2
    .{ .name = ":method", .value = "POST" }, // 3
    .{ .name = ":path", .value = "/" }, // 4
    .{ .name = ":path", .value = "/index.html" }, // 5
    .{ .name = ":scheme", .value = "http" }, // 6
    .{ .name = ":scheme", .value = "https" }, // 7
    .{ .name = ":status", .value = "200" }, // 8
    .{ .name = ":status", .value = "204" }, // 9
    .{ .name = ":status", .value = "206" }, // 10
    .{ .name = ":status", .value = "304" }, // 11
    .{ .name = ":status", .value = "400" }, // 12
    .{ .name = ":status", .value = "404" }, // 13
    .{ .name = ":status", .value = "500" }, // 14
    .{ .name = "accept-charset", .value = "" }, // 15
    .{ .name = "accept-encoding", .value = "gzip, deflate" }, // 16
    .{ .name = "accept-language", .value = "" }, // 17
    .{ .name = "accept-ranges", .value = "" }, // 18
    .{ .name = "accept", .value = "" }, // 19
    .{ .name = "access-control-allow-origin", .value = "" }, // 20
    .{ .name = "age", .value = "" }, // 21
    .{ .name = "allow", .value = "" }, // 22
    .{ .name = "authorization", .value = "" }, // 23
    .{ .name = "cache-control", .value = "" }, // 24
    .{ .name = "content-disposition", .value = "" }, // 25
    .{ .name = "content-encoding", .value = "" }, // 26
    .{ .name = "content-language", .value = "" }, // 27
    .{ .name = "content-length", .value = "" }, // 28
    .{ .name = "content-location", .value = "" }, // 29
    .{ .name = "content-range", .value = "" }, // 30
    .{ .name = "content-type", .value = "" }, // 31
    .{ .name = "cookie", .value = "" }, // 32
    .{ .name = "date", .value = "" }, // 33
    .{ .name = "etag", .value = "" }, // 34
    .{ .name = "expect", .value = "" }, // 35
    .{ .name = "expires", .value = "" }, // 36
    .{ .name = "from", .value = "" }, // 37
    .{ .name = "host", .value = "" }, // 38
    .{ .name = "if-match", .value = "" }, // 39
    .{ .name = "if-modified-since", .value = "" }, // 40
    .{ .name = "if-none-match", .value = "" }, // 41
    .{ .name = "if-range", .value = "" }, // 42
    .{ .name = "if-unmodified-since", .value = "" }, // 43
    .{ .name = "last-modified", .value = "" }, // 44
    .{ .name = "link", .value = "" }, // 45
    .{ .name = "location", .value = "" }, // 46
    .{ .name = "max-forwards", .value = "" }, // 47
    .{ .name = "proxy-authenticate", .value = "" }, // 48
    .{ .name = "proxy-authorization", .value = "" }, // 49
    .{ .name = "range", .value = "" }, // 50
    .{ .name = "referer", .value = "" }, // 51
    .{ .name = "refresh", .value = "" }, // 52
    .{ .name = "retry-after", .value = "" }, // 53
    .{ .name = "server", .value = "" }, // 54
    .{ .name = "set-cookie", .value = "" }, // 55
    .{ .name = "strict-transport-security", .value = "" }, // 56
    .{ .name = "transfer-encoding", .value = "" }, // 57
    .{ .name = "user-agent", .value = "" }, // 58
    .{ .name = "vary", .value = "" }, // 59
    .{ .name = "via", .value = "" }, // 60
    .{ .name = "www-authenticate", .value = "" }, // 61
};

// =========================================================================
// Huffman code (RFC 7541 Appendix B)
// =========================================================================

/// `{ code, bit length }` per symbol, 0..255 plus EOS at 256.
const huffman_table = blk: {
    const E = struct { u32, u8 };
    break :blk [257]E{
        .{ 0x1ff8, 13 },     .{ 0x7fffd8, 23 },   .{ 0xfffffe2, 28 },  .{ 0xfffffe3, 28 },
        .{ 0xfffffe4, 28 },  .{ 0xfffffe5, 28 },  .{ 0xfffffe6, 28 },  .{ 0xfffffe7, 28 },
        .{ 0xfffffe8, 28 },  .{ 0xffffea, 24 },   .{ 0x3ffffffc, 30 }, .{ 0xfffffe9, 28 },
        .{ 0xfffffea, 28 },  .{ 0x3ffffffd, 30 }, .{ 0xfffffeb, 28 },  .{ 0xfffffec, 28 },
        .{ 0xfffffed, 28 },  .{ 0xfffffee, 28 },  .{ 0xfffffef, 28 },  .{ 0xffffff0, 28 },
        .{ 0xffffff1, 28 },  .{ 0xffffff2, 28 },  .{ 0x3ffffffe, 30 }, .{ 0xffffff3, 28 },
        .{ 0xffffff4, 28 },  .{ 0xffffff5, 28 },  .{ 0xffffff6, 28 },  .{ 0xffffff7, 28 },
        .{ 0xffffff8, 28 },  .{ 0xffffff9, 28 },  .{ 0xffffffa, 28 },  .{ 0xffffffb, 28 },
        .{ 0x14, 6 },        .{ 0x3f8, 10 },      .{ 0x3f9, 10 },      .{ 0xffa, 12 },
        .{ 0x1ff9, 13 },     .{ 0x15, 6 },        .{ 0xf8, 8 },        .{ 0x7fa, 11 },
        .{ 0x3fa, 10 },      .{ 0x3fb, 10 },      .{ 0xf9, 8 },        .{ 0x7fb, 11 },
        .{ 0xfa, 8 },        .{ 0x16, 6 },        .{ 0x17, 6 },        .{ 0x18, 6 },
        .{ 0x0, 5 },         .{ 0x1, 5 },         .{ 0x2, 5 },         .{ 0x19, 6 },
        .{ 0x1a, 6 },        .{ 0x1b, 6 },        .{ 0x1c, 6 },        .{ 0x1d, 6 },
        .{ 0x1e, 6 },        .{ 0x1f, 6 },        .{ 0x5c, 7 },        .{ 0xfb, 8 },
        .{ 0x7ffc, 15 },     .{ 0x20, 6 },        .{ 0xffb, 12 },      .{ 0x3fc, 10 },
        .{ 0x1ffa, 13 },     .{ 0x21, 6 },        .{ 0x5d, 7 },        .{ 0x5e, 7 },
        .{ 0x5f, 7 },        .{ 0x60, 7 },        .{ 0x61, 7 },        .{ 0x62, 7 },
        .{ 0x63, 7 },        .{ 0x64, 7 },        .{ 0x65, 7 },        .{ 0x66, 7 },
        .{ 0x67, 7 },        .{ 0x68, 7 },        .{ 0x69, 7 },        .{ 0x6a, 7 },
        .{ 0x6b, 7 },        .{ 0x6c, 7 },        .{ 0x6d, 7 },        .{ 0x6e, 7 },
        .{ 0x6f, 7 },        .{ 0x70, 7 },        .{ 0x71, 7 },        .{ 0x72, 7 },
        .{ 0xfc, 8 },        .{ 0x73, 7 },        .{ 0xfd, 8 },        .{ 0x1ffb, 13 },
        .{ 0x7fff0, 19 },    .{ 0x1ffc, 13 },     .{ 0x3ffc, 14 },     .{ 0x22, 6 },
        .{ 0x7ffd, 15 },     .{ 0x3, 5 },         .{ 0x23, 6 },        .{ 0x4, 5 },
        .{ 0x24, 6 },        .{ 0x5, 5 },         .{ 0x25, 6 },        .{ 0x26, 6 },
        .{ 0x27, 6 },        .{ 0x6, 5 },         .{ 0x74, 7 },        .{ 0x75, 7 },
        .{ 0x28, 6 },        .{ 0x29, 6 },        .{ 0x2a, 6 },        .{ 0x7, 5 },
        .{ 0x2b, 6 },        .{ 0x76, 7 },        .{ 0x2c, 6 },        .{ 0x8, 5 },
        .{ 0x9, 5 },         .{ 0x2d, 6 },        .{ 0x77, 7 },        .{ 0x78, 7 },
        .{ 0x79, 7 },        .{ 0x7a, 7 },        .{ 0x7b, 7 },        .{ 0x7ffe, 15 },
        .{ 0x7fc, 11 },      .{ 0x3ffd, 14 },     .{ 0x1ffd, 13 },     .{ 0xffffffc, 28 },
        .{ 0xfffe6, 20 },    .{ 0x3fffd2, 22 },   .{ 0xfffe7, 20 },    .{ 0xfffe8, 20 },
        .{ 0x3fffd3, 22 },   .{ 0x3fffd4, 22 },   .{ 0x3fffd5, 22 },   .{ 0x7fffd9, 23 },
        .{ 0x3fffd6, 22 },   .{ 0x7fffda, 23 },   .{ 0x7fffdb, 23 },   .{ 0x7fffdc, 23 },
        .{ 0x7fffdd, 23 },   .{ 0x7fffde, 23 },   .{ 0xffffeb, 24 },   .{ 0x7fffdf, 23 },
        .{ 0xffffec, 24 },   .{ 0xffffed, 24 },   .{ 0x3fffd7, 22 },   .{ 0x7fffe0, 23 },
        .{ 0xffffee, 24 },   .{ 0x7fffe1, 23 },   .{ 0x7fffe2, 23 },   .{ 0x7fffe3, 23 },
        .{ 0x7fffe4, 23 },   .{ 0x1fffdc, 21 },   .{ 0x3fffd8, 22 },   .{ 0x7fffe5, 23 },
        .{ 0x3fffd9, 22 },   .{ 0x7fffe6, 23 },   .{ 0x7fffe7, 23 },   .{ 0xffffef, 24 },
        .{ 0x3fffda, 22 },   .{ 0x1fffdd, 21 },   .{ 0xfffe9, 20 },    .{ 0x3fffdb, 22 },
        .{ 0x3fffdc, 22 },   .{ 0x7fffe8, 23 },   .{ 0x7fffe9, 23 },   .{ 0x1fffde, 21 },
        .{ 0x7fffea, 23 },   .{ 0x3fffdd, 22 },   .{ 0x3fffde, 22 },   .{ 0xfffff0, 24 },
        .{ 0x1fffdf, 21 },   .{ 0x3fffdf, 22 },   .{ 0x7fffeb, 23 },   .{ 0x7fffec, 23 },
        .{ 0x1fffe0, 21 },   .{ 0x1fffe1, 21 },   .{ 0x3fffe0, 22 },   .{ 0x1fffe2, 21 },
        .{ 0x7fffed, 23 },   .{ 0x3fffe1, 22 },   .{ 0x7fffee, 23 },   .{ 0x7fffef, 23 },
        .{ 0xfffea, 20 },    .{ 0x3fffe2, 22 },   .{ 0x3fffe3, 22 },   .{ 0x3fffe4, 22 },
        .{ 0x7ffff0, 23 },   .{ 0x3fffe5, 22 },   .{ 0x3fffe6, 22 },   .{ 0x7ffff1, 23 },
        .{ 0x3ffffe0, 26 },  .{ 0x3ffffe1, 26 },  .{ 0xfffeb, 20 },    .{ 0x7fff1, 19 },
        .{ 0x3fffe7, 22 },   .{ 0x7ffff2, 23 },   .{ 0x3fffe8, 22 },   .{ 0x1ffffec, 25 },
        .{ 0x3ffffe2, 26 },  .{ 0x3ffffe3, 26 },  .{ 0x3ffffe4, 26 },  .{ 0x7ffffde, 27 },
        .{ 0x7ffffdf, 27 },  .{ 0x3ffffe5, 26 },  .{ 0xfffff1, 24 },   .{ 0x1ffffed, 25 },
        .{ 0x7fff2, 19 },    .{ 0x1fffe3, 21 },   .{ 0x3ffffe6, 26 },  .{ 0x7ffffe0, 27 },
        .{ 0x7ffffe1, 27 },  .{ 0x3ffffe7, 26 },  .{ 0x7ffffe2, 27 },  .{ 0xfffff2, 24 },
        .{ 0x1fffe4, 21 },   .{ 0x1fffe5, 21 },   .{ 0x3ffffe8, 26 },  .{ 0x3ffffe9, 26 },
        .{ 0xffffffd, 28 },  .{ 0x7ffffe3, 27 },  .{ 0x7ffffe4, 27 },  .{ 0x7ffffe5, 27 },
        .{ 0xfffec, 20 },    .{ 0xfffff3, 24 },   .{ 0xfffed, 20 },    .{ 0x1fffe6, 21 },
        .{ 0x3fffe9, 22 },   .{ 0x1fffe7, 21 },   .{ 0x1fffe8, 21 },   .{ 0x7ffff3, 23 },
        .{ 0x3fffea, 22 },   .{ 0x3fffeb, 22 },   .{ 0x1ffffee, 25 },  .{ 0x1ffffef, 25 },
        .{ 0xfffff4, 24 },   .{ 0xfffff5, 24 },   .{ 0x3ffffea, 26 },  .{ 0x7ffff4, 23 },
        .{ 0x3ffffeb, 26 },  .{ 0x7ffffe6, 27 },  .{ 0x3ffffec, 26 },  .{ 0x3ffffed, 26 },
        .{ 0x7ffffe7, 27 },  .{ 0x7ffffe8, 27 },  .{ 0x7ffffe9, 27 },  .{ 0x7ffffea, 27 },
        .{ 0x7ffffeb, 27 },  .{ 0xffffffe, 28 },  .{ 0x7ffffec, 27 },  .{ 0x7ffffed, 27 },
        .{ 0x7ffffee, 27 },  .{ 0x7ffffef, 27 },  .{ 0x7fffff0, 27 },  .{ 0x3ffffee, 26 },
        .{ 0x3fffffff, 30 },
    };
};

/// Canonical-code decode tables, built at comptime from `huffman_table`.
///
/// A canonical Huffman code can be decoded without a tree: for each bit length
/// `L`, all codes of that length are consecutive, so knowing the first code and
/// the first symbol index at each length is enough. Reading bit by bit is
/// O(bits) rather than the O(1) a multi-level lookup table would give, but
/// header strings here are tens of bytes and §6.1 is explicit that the native
/// transport should only be optimised "if and only if profiling shows
/// framing/HPACK on the critical path". This is the version that is obviously
/// correct; the lookup-table version is what to write if W0 says it matters.
const HuffDecode = struct {
    /// Number of codes of each bit length, 0..30.
    count: [31]u16,
    /// Smallest code of each bit length.
    first_code: [31]u32,
    /// Index into `sorted` of the first symbol of each bit length.
    first_index: [31]u16,
    /// Symbols ordered by (bit length, code).
    sorted: [257]u16,
};

const huff_decode: HuffDecode = blk: {
    @setEvalBranchQuota(100_000);
    var d: HuffDecode = .{
        .count = @splat(0),
        .first_code = @splat(0),
        .first_index = @splat(0),
        .sorted = @splat(0),
    };
    for (huffman_table) |e| d.count[e[1]] += 1;

    var code: u32 = 0;
    var index: u16 = 0;
    for (1..31) |len| {
        d.first_code[len] = code;
        d.first_index[len] = index;
        // Symbols of this length, in ascending code order. The table is already
        // in code order within a length because it is a canonical code.
        for (huffman_table, 0..) |e, sym| {
            if (e[1] == len) {
                d.sorted[index] = @intCast(sym);
                index += 1;
            }
        }
        code = (code + d.count[len]) << 1;
    }
    break :blk d;
};

/// Decode a Huffman-encoded string into `out`. Returns the decoded length.
pub fn huffmanDecode(out: []u8, src: []const u8) Error!usize {
    var n: usize = 0;
    var code: u32 = 0;
    var len: u5 = 0;

    for (src) |byte| {
        var bit: i32 = 7;
        while (bit >= 0) : (bit -= 1) {
            code = (code << 1) | ((byte >> @intCast(bit)) & 1);
            if (len == 30) return Error.InvalidHuffman;
            len += 1;

            const cnt = huff_decode.count[len];
            if (cnt == 0) continue;
            const first = huff_decode.first_code[len];
            if (code >= first and code - first < cnt) {
                const sym = huff_decode.sorted[huff_decode.first_index[len] + (code - first)];
                // 256 is EOS. RFC 7541 §5.2: "A Huffman-encoded string literal
                // containing the EOS symbol MUST be treated as a decoding
                // error." Accepting it would let a peer smuggle a zero byte
                // past length accounting.
                if (sym == 256) return Error.InvalidHuffman;
                if (n >= out.len) return Error.HeaderTooLarge;
                out[n] = @intCast(sym);
                n += 1;
                code = 0;
                len = 0;
            }
        }
    }

    // RFC 7541 §5.2: the remainder must be fewer than 8 bits and must be the
    // most-significant bits of the EOS code (all ones). Anything else is
    // padding that carries information, which is not allowed.
    if (len > 7) return Error.InvalidHuffman;
    if (len > 0) {
        const expected = (@as(u32, 1) << len) - 1;
        if (code != expected) return Error.InvalidHuffman;
    }
    return n;
}

/// Upper bound on the decoded length of `n` Huffman-encoded bytes.
///
/// The shortest code is 5 bits, so 8 bits can produce at most 8/5 symbols.
pub fn huffmanMaxDecodedLen(n: usize) usize {
    return n * 8 / 5 + 1;
}

// =========================================================================
// Integer coding (RFC 7541 §5.1)
// =========================================================================

/// Decode an HPACK integer with an `n`-bit prefix.
///
/// `first` is the byte containing the prefix; `src` is everything after it.
/// Returns the value and how many continuation bytes were consumed.
pub fn decodeInteger(comptime prefix_bits: u4, first: u8, src: []const u8) Error!struct { value: u32, consumed: usize } {
    // `1 << 8` overflows u8, so the 8-bit-prefix case (used by
    // `encodeInteger` for whole-byte values) is spelled out.
    const mask: u8 = comptime if (prefix_bits >= 8) 0xff else (@as(u8, 1) << @intCast(prefix_bits)) - 1;
    const prefix = first & mask;
    if (prefix < mask) return .{ .value = prefix, .consumed = 0 };

    var value: u64 = mask;
    var shift: u6 = 0;
    var i: usize = 0;
    while (i < src.len) {
        const b = src[i];
        i += 1;
        value += @as(u64, b & 0x7f) << shift;
        // A value beyond u32 is either a malformed frame or an attempt to make
        // us allocate absurdly. §8.8 fuzzes "oversized length prefixes"; this
        // is where that lands.
        if (value > std.math.maxInt(u32)) return Error.IntegerOverflow;
        if (b & 0x80 == 0) return .{ .value = @intCast(value), .consumed = i };
        shift += 7;
        if (shift > 28) return Error.IntegerOverflow;
    }
    return Error.Truncated;
}

/// Encode an HPACK integer with an `n`-bit prefix into `out`.
/// `flags` supplies the bits above the prefix.
pub fn encodeInteger(out: []u8, comptime prefix_bits: u4, flags: u8, value: u32) usize {
    const mask: u8 = comptime if (prefix_bits >= 8) 0xff else (@as(u8, 1) << @intCast(prefix_bits)) - 1;
    if (value < mask) {
        out[0] = flags | @as(u8, @intCast(value));
        return 1;
    }
    out[0] = flags | mask;
    var v = value - mask;
    var n: usize = 1;
    while (v >= 0x80) {
        out[n] = @as(u8, @truncate(v)) | 0x80;
        v >>= 7;
        n += 1;
    }
    out[n] = @truncate(v);
    return n + 1;
}

// =========================================================================
// Dynamic table
// =========================================================================

/// §6.1: "fixed-size header table". No allocation, no growth.
///
/// Entries are stored newest-first in a ring; wire index `61 + i` names the
/// `i`-th newest. RFC 7541 §4.1 defines an entry's size as
/// `name.len + value.len + 32`, and eviction is driven by that accounting
/// rather than by entry count.
pub const DynamicTable = struct {
    /// Backing storage for names and values, used as a bump arena that is
    /// rebuilt on eviction. Sized for the default 4096-byte SETTINGS value with
    /// headroom for the 32-byte-per-entry overhead not being stored here.
    storage: [16 * 1024]u8 = undefined,
    storage_used: usize = 0,

    entries: [256]Entry = undefined,
    count: usize = 0,

    /// Current accounted size, per RFC 7541 §4.1.
    size: usize = 0,
    /// Limit from the peer's `SETTINGS_HEADER_TABLE_SIZE`, adjustable downward
    /// by a dynamic table size update.
    max_size: usize = default_table_size,

    pub const Entry = struct {
        name: []const u8,
        value: []const u8,

        pub fn accountedSize(self: Entry) usize {
            return self.name.len + self.value.len + 32;
        }
    };

    pub fn init() DynamicTable {
        return .{};
    }

    pub fn reset(self: *DynamicTable) void {
        self.storage_used = 0;
        self.count = 0;
        self.size = 0;
    }

    pub fn setMaxSize(self: *DynamicTable, n: usize) void {
        self.max_size = n;
        self.evictTo(n);
    }

    fn evictTo(self: *DynamicTable, limit: usize) void {
        while (self.size > limit and self.count > 0) {
            self.count -= 1;
            self.size -= self.entries[self.count].accountedSize();
        }
        if (self.count == 0) self.storage_used = 0;
    }

    /// Insert at the front. Returns false when the entry cannot be stored.
    ///
    /// `name` and `value` may alias this table's own storage, a
    /// literal-with-indexing header can take its name from an existing entry -
    /// so they are copied into a stack staging buffer before any compaction
    /// runs. Without that, `compact()` rewrites `storage` underneath them and
    /// the subsequent `@memcpy` has overlapping source and destination: an
    /// aliasing panic in safe builds, and a corrupted entry name either way.
    pub fn insert(self: *DynamicTable, name_in: []const u8, value_in: []const u8) bool {
        var staging: [max_entry_bytes]u8 = undefined;
        // RFC 7541 §4.4: an entry larger than the maximum size empties the
        // table and is not inserted. That is not an error. Checked before the
        // copy: `max_size` never exceeds `default_table_size`, so anything
        // that would not fit the staging buffer lands here too. Returning
        // `false` for that case left the old entries in place, one index off
        // from the peer's view of the table for the rest of the connection.
        if (name_in.len + value_in.len + 32 > self.max_size) {
            self.reset();
            return true;
        }
        std.debug.assert(name_in.len + value_in.len <= staging.len);
        @memcpy(staging[0..name_in.len], name_in);
        @memcpy(staging[name_in.len..][0..value_in.len], value_in);
        const name = staging[0..name_in.len];
        const value = staging[name_in.len..][0..value_in.len];

        const entry_size = name.len + value.len + 32;
        self.evictTo(self.max_size - entry_size);

        // Compact the arena when it cannot fit the new strings. Entries are
        // copied in place; a ring of slices into a bump arena is much simpler
        // to reason about than a true ring buffer of bytes, and the compaction
        // is O(table size) on a table bounded at 4 KiB.
        if (self.storage_used + name.len + value.len > self.storage.len) {
            self.compact();
            if (self.storage_used + name.len + value.len > self.storage.len) return false;
        }
        if (self.count == self.entries.len) {
            self.count -= 1;
            self.size -= self.entries[self.count].accountedSize();
        }

        const n_start = self.storage_used;
        @memcpy(self.storage[n_start..][0..name.len], name);
        const v_start = n_start + name.len;
        @memcpy(self.storage[v_start..][0..value.len], value);
        self.storage_used = v_start + value.len;

        // Shift to make room at the front.
        var i = self.count;
        while (i > 0) : (i -= 1) self.entries[i] = self.entries[i - 1];
        self.entries[0] = .{
            .name = self.storage[n_start..v_start],
            .value = self.storage[v_start..self.storage_used],
        };
        self.count += 1;
        self.size += entry_size;
        return true;
    }

    fn compact(self: *DynamicTable) void {
        var tmp: [16 * 1024]u8 = undefined;
        var used: usize = 0;
        for (self.entries[0..self.count]) |*e| {
            const n_start = used;
            @memcpy(tmp[n_start..][0..e.name.len], e.name);
            const v_start = n_start + e.name.len;
            @memcpy(tmp[v_start..][0..e.value.len], e.value);
            used = v_start + e.value.len;
            e.name = self.storage[n_start..v_start];
            e.value = self.storage[v_start..used];
        }
        @memcpy(self.storage[0..used], tmp[0..used]);
        self.storage_used = used;
    }

    pub fn get(self: *const DynamicTable, i: usize) ?Entry {
        if (i >= self.count) return null;
        return self.entries[i];
    }
};

// =========================================================================
// Decoder
// =========================================================================

pub const Header = struct {
    name: []const u8,
    value: []const u8,
};

pub const max_headers = 64;

/// Upper bound on one dynamic-table entry's strings.
///
/// The table's accounted size is capped at 4096 including 32 bytes of overhead
/// per entry, so no storable entry can exceed that in strings alone.
pub const max_entry_bytes = 4096;

/// The dynamic table size we advertise in SETTINGS_HEADER_TABLE_SIZE and the
/// ceiling the decoder enforces on a Dynamic Table Size Update. One constant,
/// so the two cannot drift apart: `h2.Settings.header_table_size` defaults to
/// it and `max_entry_bytes` must be at least it minus the 32-byte overhead.
pub const default_table_size: usize = 4096;

/// Decodes one header block into a fixed arena and a fixed header array.
///
/// §6.1: "arena per stream reset on completion". `reset` returns the decoder to
/// its initial state without touching the dynamic table, which must persist for
/// the life of the connection, HPACK is stateful across requests, and clearing
/// it between them would desynchronise the peer.
pub const Decoder = struct {
    table: DynamicTable = DynamicTable.init(),

    /// Scratch for Huffman-decoded and copied strings.
    arena: [32 * 1024]u8 = undefined,
    arena_used: usize = 0,

    headers: [max_headers]Header = undefined,
    header_count: usize = 0,

    pub fn init() Decoder {
        return .{};
    }

    /// Reset per-block state. The dynamic table survives.
    pub fn resetBlock(self: *Decoder) void {
        self.arena_used = 0;
        self.header_count = 0;
    }

    fn alloc(self: *Decoder, n: usize) Error![]u8 {
        if (self.arena_used + n > self.arena.len) return Error.HeaderTooLarge;
        const s = self.arena[self.arena_used..][0..n];
        self.arena_used += n;
        return s;
    }

    fn addHeader(self: *Decoder, name: []const u8, value: []const u8) Error!void {
        if (self.header_count >= max_headers) return Error.TooManyHeaders;
        self.headers[self.header_count] = .{ .name = name, .value = value };
        self.header_count += 1;
    }

    /// Read a string literal (§5.2): one bit of Huffman flag, 7-bit length.
    fn readString(self: *Decoder, src: []const u8, pos: *usize) Error![]const u8 {
        if (pos.* >= src.len) return Error.Truncated;
        const first = src[pos.*];
        const huffman = (first & 0x80) != 0;
        const int = try decodeInteger(7, first, src[pos.* + 1 ..]);
        pos.* += 1 + int.consumed;
        const len = int.value;
        if (pos.* + len > src.len) return Error.Truncated;
        const raw = src[pos.*..][0..len];
        pos.* += len;

        if (!huffman) {
            // Copy into the arena. Borrowing the frame buffer would be faster,
            // but the frame is reused for the next read and header values must
            // outlive it, a request's `:path` is read after the whole block is
            // parsed.
            const dst = try self.alloc(raw.len);
            @memcpy(dst, raw);
            return dst;
        }
        const dst = try self.alloc(huffmanMaxDecodedLen(raw.len));
        const n = try huffmanDecode(dst, raw);
        // Give back the unused tail so the arena is not wasted on the
        // conservative bound.
        self.arena_used -= dst.len - n;
        return dst[0..n];
    }

    /// The name of the entry at `index`, as a slice that outlives the block.
    ///
    /// A static name is a program constant. A dynamic name is a slice into
    /// the table's bump arena, which `compact()` (or an eviction that empties
    /// the table) relocates on a later insert *in the same block*, and the
    /// literal that named it is reported to the caller after the whole block
    /// has been decoded. So a dynamic name is copied into the arena, whose
    /// slices are stable for exactly the lifetime `decode` promises.
    fn copyName(self: *Decoder, index: u32) Error![]const u8 {
        return (try self.copyEntry(index, false)).name;
    }

    /// Both strings of the entry at `index`, stable for the block. Same
    /// argument as `copyName`: an indexed dynamic header (§6.1) reported as
    /// slices into the table's storage is rewritten by a later `compact()`
    /// in the same block. `with_value` false skips the value copy for the
    /// literal branches, which read their own value.
    fn copyEntry(self: *Decoder, index: u32, with_value: bool) Error!DynamicTable.Entry {
        const e = try self.lookup(index);
        if (index <= static_table.len) return e;
        const name = try self.alloc(e.name.len);
        @memcpy(name, e.name);
        if (!with_value) return .{ .name = name, .value = e.value };
        const value = try self.alloc(e.value.len);
        @memcpy(value, e.value);
        return .{ .name = name, .value = value };
    }

    fn lookup(self: *const Decoder, index: u32) Error!DynamicTable.Entry {
        if (index == 0) return Error.InvalidIndex;
        if (index <= static_table.len) {
            const e = static_table[index - 1];
            return .{ .name = e.name, .value = e.value };
        }
        return self.table.get(index - static_table.len - 1) orelse Error.InvalidIndex;
    }

    /// Decode a complete header block. Returns the decoded headers, which
    /// borrow from the decoder's arena and remain valid until `resetBlock`.
    pub fn decode(self: *Decoder, src: []const u8) Error![]const Header {
        self.resetBlock();
        var pos: usize = 0;

        while (pos < src.len) {
            const b = src[pos];

            if (b & 0x80 != 0) {
                // §6.1: Indexed Header Field.
                const int = try decodeInteger(7, b, src[pos + 1 ..]);
                pos += 1 + int.consumed;
                // Copy a dynamic entry: `lookup` alone would hand out slices
                // into the table's storage, which a later insert in this same
                // block can relocate (see `copyEntry`).
                const e = try self.copyEntry(int.value, true);
                try self.addHeader(e.name, e.value);
            } else if (b & 0xc0 == 0x40) {
                // §6.2.1: Literal Header Field with Incremental Indexing.
                const int = try decodeInteger(6, b, src[pos + 1 ..]);
                pos += 1 + int.consumed;
                const name = if (int.value == 0)
                    try self.readString(src, &pos)
                else
                    try self.copyName(int.value);
                const value = try self.readString(src, &pos);
                _ = self.table.insert(name, value);
                // Report the *arena* slices, never the table's copies.
                //
                // `addHeader` stores slices without copying, and the table's
                // storage is relocated in place by `compact()` on a later
                // insert in the same block. Handing out `table.get(0)` here
                // meant a header decoded early in a block could be silently
                // rewritten by a header decoded later in it, and `:path` is
                // exactly what tonic sends this way, so the symptom is a
                // request routed to the wrong method.
                //
                // The arena slices are stable for the life of the block, which
                // is precisely the lifetime `decode` promises its caller.
                try self.addHeader(name, value);
            } else if (b & 0xe0 == 0x20) {
                // §6.3: Dynamic Table Size Update.
                const int = try decodeInteger(5, b, src[pos + 1 ..]);
                pos += 1 + int.consumed;
                // The bound is what we advertised as SETTINGS_HEADER_TABLE_SIZE
                // (`h2.Settings.header_table_size` is this same constant); a
                // peer may only shrink the table below that, never grow it.
                if (int.value > default_table_size) return Error.TableSizeExceeded;
                self.table.setMaxSize(int.value);
            } else {
                // §6.2.2 / §6.2.3: Literal without Indexing / Never Indexed.
                // Both are decoded identically; "never indexed" only constrains
                // intermediaries, which we are not.
                const int = try decodeInteger(4, b, src[pos + 1 ..]);
                pos += 1 + int.consumed;
                const name = if (int.value == 0)
                    try self.readString(src, &pos)
                else
                    try self.copyName(int.value);
                const value = try self.readString(src, &pos);
                try self.addHeader(name, value);
            }
        }
        return self.headers[0..self.header_count];
    }
};

// =========================================================================
// Encoder
// =========================================================================

/// Stateless HPACK encoder: static indices plus literal-without-indexing.
/// See the module doc for why it never uses Huffman or the dynamic table.
pub const Encoder = struct {
    /// Emit a header, using a static-table index when the exact name+value
    /// pair is present.
    pub fn writeHeader(out: []u8, name: []const u8, value: []const u8) Error!usize {
        if (staticIndexFull(name, value)) |idx| {
            if (out.len < 8) return Error.HeaderTooLarge;
            return encodeInteger(out, 7, 0x80, idx);
        }
        var n: usize = 0;
        if (staticIndexName(name)) |idx| {
            // Literal without indexing, name from the static table: 0000 prefix.
            if (out.len < n + 8) return Error.HeaderTooLarge;
            n += encodeInteger(out[n..], 4, 0x00, idx);
        } else {
            if (out.len < n + 8) return Error.HeaderTooLarge;
            out[n] = 0x00;
            n += 1;
            n += try writeRawString(out[n..], name);
        }
        n += try writeRawString(out[n..], value);
        return n;
    }

    fn writeRawString(out: []u8, s: []const u8) Error!usize {
        if (out.len < 8) return Error.HeaderTooLarge;
        // High bit clear = not Huffman.
        var n = encodeInteger(out, 7, 0x00, @intCast(s.len));
        if (out.len < n + s.len) return Error.HeaderTooLarge;
        @memcpy(out[n..][0..s.len], s);
        n += s.len;
        return n;
    }

    fn staticIndexFull(name: []const u8, value: []const u8) ?u32 {
        for (static_table, 1..) |e, i| {
            if (e.value.len != 0 and std.mem.eql(u8, e.name, name) and std.mem.eql(u8, e.value, value)) {
                return @intCast(i);
            }
        }
        return null;
    }

    fn staticIndexName(name: []const u8) ?u32 {
        for (static_table, 1..) |e, i| {
            if (std.mem.eql(u8, e.name, name)) return @intCast(i);
        }
        return null;
    }
};

// =========================================================================
// Tests
// =========================================================================

const testing = std.testing;

test "huffman table is a valid canonical prefix code (Kraft equality)" {
    // Σ 2^-len over all 257 symbols must be exactly 1 for a complete code.
    // Scaled to 2^30 to stay in integers.
    var sum: u64 = 0;
    for (huffman_table) |e| sum += @as(u64, 1) << @intCast(30 - e[1]);
    try testing.expectEqual(@as(u64, 1) << 30, sum);
}

test "huffman decode: RFC 7541 C.4.1 :authority value" {
    // "www.example.com" Huffman-encoded.
    const enc = [_]u8{ 0xf1, 0xe3, 0xc2, 0xe5, 0xf2, 0x3a, 0x6b, 0xa0, 0xab, 0x90, 0xf4, 0xff };
    var out: [64]u8 = undefined;
    const n = try huffmanDecode(&out, &enc);
    try testing.expectEqualStrings("www.example.com", out[0..n]);
}

test "huffman decode: RFC 7541 C.4.2 and C.4.3 vectors" {
    var out: [64]u8 = undefined;
    {
        const enc = [_]u8{ 0xa8, 0xeb, 0x10, 0x64, 0x9c, 0xbf };
        const n = try huffmanDecode(&out, &enc);
        try testing.expectEqualStrings("no-cache", out[0..n]);
    }
    {
        const enc = [_]u8{ 0x25, 0xa8, 0x49, 0xe9, 0x5b, 0xa9, 0x7d, 0x7f };
        const n = try huffmanDecode(&out, &enc);
        try testing.expectEqualStrings("custom-key", out[0..n]);
    }
    {
        const enc = [_]u8{ 0x25, 0xa8, 0x49, 0xe9, 0x5b, 0xb8, 0xe8, 0xb4, 0xbf };
        const n = try huffmanDecode(&out, &enc);
        try testing.expectEqualStrings("custom-value", out[0..n]);
    }
}

test "huffman decode rejects EOS and bad padding" {
    // All-ones for 30 bits is EOS.
    const eos = [_]u8{ 0xff, 0xff, 0xff, 0xfc };
    var out: [64]u8 = undefined;
    try testing.expectError(Error.InvalidHuffman, huffmanDecode(&out, &eos));

    // Padding with a zero bit is not the MSBs of EOS.
    // 'a' is 00011 (5 bits); pad the remaining 3 bits with 000 instead of 111.
    const bad_pad = [_]u8{0b00011_000};
    try testing.expectError(Error.InvalidHuffman, huffmanDecode(&out, &bad_pad));
}

test "hpack integer coding round-trips, including RFC examples" {
    // RFC 7541 C.1.1: 10 with a 5-bit prefix -> single byte.
    var buf: [16]u8 = undefined;
    var n = encodeInteger(&buf, 5, 0x00, 10);
    try testing.expectEqual(@as(usize, 1), n);
    var d = try decodeInteger(5, buf[0], buf[1..n]);
    try testing.expectEqual(@as(u32, 10), d.value);

    // C.1.2: 1337 with a 5-bit prefix -> 31, 154, 10.
    n = encodeInteger(&buf, 5, 0x00, 1337);
    try testing.expectEqual(@as(usize, 3), n);
    try testing.expectEqual(@as(u8, 31), buf[0]);
    try testing.expectEqual(@as(u8, 154), buf[1]);
    try testing.expectEqual(@as(u8, 10), buf[2]);
    d = try decodeInteger(5, buf[0], buf[1..n]);
    try testing.expectEqual(@as(u32, 1337), d.value);

    // C.1.3: 42 with an 8-bit prefix.
    n = encodeInteger(&buf, 8, 0x00, 42);
    try testing.expectEqual(@as(usize, 1), n);
    d = try decodeInteger(8, buf[0], buf[1..n]);
    try testing.expectEqual(@as(u32, 42), d.value);

    // Round-trip a spread of magnitudes at every prefix width we use.
    inline for ([_]u4{ 4, 5, 6, 7 }) |bits| {
        for ([_]u32{ 0, 1, 14, 15, 16, 127, 128, 255, 256, 65535, 1 << 20 }) |v| {
            const m = encodeInteger(&buf, bits, 0x00, v);
            const got = try decodeInteger(bits, buf[0], buf[1..m]);
            try testing.expectEqual(v, got.value);
            try testing.expectEqual(m - 1, got.consumed);
        }
    }
}

test "hpack integer rejects oversized encodings" {
    // §8.8 fuzzes "oversized length prefixes".
    const huge = [_]u8{ 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
    try testing.expectError(Error.IntegerOverflow, decodeInteger(7, 0xff, &huge));

    const truncated = [_]u8{ 0x80, 0x80 };
    try testing.expectError(Error.Truncated, decodeInteger(7, 0xff, &truncated));
}

test "decoder handles RFC 7541 C.3 request sequence with a dynamic table" {
    var dec = Decoder.init();

    // C.3.1: :method GET:scheme http:path /:authority www.example.com
    const b1 = [_]u8{
        0x82, 0x86, 0x84, 0x41, 0x0f, 0x77, 0x77, 0x77,
        0x2e, 0x65, 0x78, 0x61, 0x6d, 0x70, 0x6c, 0x65,
        0x2e, 0x63, 0x6f, 0x6d,
    };
    const h1 = try dec.decode(&b1);
    try testing.expectEqual(@as(usize, 4), h1.len);
    try testing.expectEqualStrings(":method", h1[0].name);
    try testing.expectEqualStrings("GET", h1[0].value);
    try testing.expectEqualStrings(":scheme", h1[1].name);
    try testing.expectEqualStrings("http", h1[1].value);
    try testing.expectEqualStrings(":path", h1[2].name);
    try testing.expectEqualStrings("/", h1[2].value);
    try testing.expectEqualStrings(":authority", h1[3].name);
    try testing.expectEqualStrings("www.example.com", h1[3].value);
    // One entry inserted, size 57 per the RFC.
    try testing.expectEqual(@as(usize, 1), dec.table.count);
    try testing.expectEqual(@as(usize, 57), dec.table.size);

    // C.3.2: same, plus cache-control: no-cache. Index 62 names the entry
    // inserted above, which only works if table state survived resetBlock.
    const b2 = [_]u8{
        0x82, 0x86, 0x84, 0xbe, 0x58, 0x08, 0x6e, 0x6f,
        0x2d, 0x63, 0x61, 0x63, 0x68, 0x65,
    };
    const h2 = try dec.decode(&b2);
    try testing.expectEqual(@as(usize, 5), h2.len);
    try testing.expectEqualStrings(":authority", h2[3].name);
    try testing.expectEqualStrings("www.example.com", h2[3].value);
    try testing.expectEqualStrings("cache-control", h2[4].name);
    try testing.expectEqualStrings("no-cache", h2[4].value);
    try testing.expectEqual(@as(usize, 2), dec.table.count);
    try testing.expectEqual(@as(usize, 110), dec.table.size);
}

test "decoder handles the Huffman-encoded C.4 request sequence" {
    var dec = Decoder.init();
    const b = [_]u8{
        0x82, 0x86, 0x84, 0x41, 0x8c, 0xf1, 0xe3, 0xc2,
        0xe5, 0xf2, 0x3a, 0x6b, 0xa0, 0xab, 0x90, 0xf4,
        0xff,
    };
    const h = try dec.decode(&b);
    try testing.expectEqual(@as(usize, 4), h.len);
    try testing.expectEqualStrings("www.example.com", h[3].value);
}

test "decoder rejects an index naming nothing" {
    var dec = Decoder.init();
    // Index 62 with an empty dynamic table.
    const b = [_]u8{0xbe};
    try testing.expectError(Error.InvalidIndex, dec.decode(&b));

    // Index 0 is never valid.
    const b0 = [_]u8{0x80};
    try testing.expectError(Error.InvalidIndex, dec.decode(&b0));
}

test "decoder handles the gRPC request headers a tonic client actually sends" {
    // Reconstructed from what qdrant-client/tonic emits: literal-with-indexing
    // for :path, static indices for :method POST and te: trailers is a literal.
    var dec = Decoder.init();

    var buf: [512]u8 = undefined;
    var n: usize = 0;
    // :method POST (static index 3)
    n += encodeInteger(buf[n..], 7, 0x80, 3);
    // :scheme http (static index 6)
    n += encodeInteger(buf[n..], 7, 0x80, 6);
    // :path -> literal with incremental indexing, name index 4
    n += encodeInteger(buf[n..], 6, 0x40, 4);
    const path = "/qdrant.Points/QueryBatch";
    n += encodeInteger(buf[n..], 7, 0x00, path.len);
    @memcpy(buf[n..][0..path.len], path);
    n += path.len;
    // content-type application/grpc -> literal without indexing, name index 31
    n += encodeInteger(buf[n..], 4, 0x00, 31);
    const ct = "application/grpc";
    n += encodeInteger(buf[n..], 7, 0x00, ct.len);
    @memcpy(buf[n..][0..ct.len], ct);
    n += ct.len;
    // te: trailers -> fully literal
    buf[n] = 0x00;
    n += 1;
    const te = "te";
    n += encodeInteger(buf[n..], 7, 0x00, te.len);
    @memcpy(buf[n..][0..te.len], te);
    n += te.len;
    const tev = "trailers";
    n += encodeInteger(buf[n..], 7, 0x00, tev.len);
    @memcpy(buf[n..][0..tev.len], tev);
    n += tev.len;

    const h = try dec.decode(buf[0..n]);
    try testing.expectEqual(@as(usize, 5), h.len);
    try testing.expectEqualStrings("POST", h[0].value);
    try testing.expectEqualStrings(":path", h[2].name);
    try testing.expectEqualStrings(path, h[2].value);
    try testing.expectEqualStrings("content-type", h[3].name);
    try testing.expectEqualStrings("application/grpc", h[3].value);
    try testing.expectEqualStrings("te", h[4].name);
    try testing.expectEqualStrings("trailers", h[4].value);
}

test "encoder output round-trips through the decoder" {
    var buf: [512]u8 = undefined;
    var n: usize = 0;
    n += try Encoder.writeHeader(buf[n..], ":status", "200");
    n += try Encoder.writeHeader(buf[n..], "content-type", "application/grpc");
    n += try Encoder.writeHeader(buf[n..], "grpc-status", "0");
    n += try Encoder.writeHeader(buf[n..], "grpc-message", "");

    var dec = Decoder.init();
    const h = try dec.decode(buf[0..n]);
    try testing.expectEqual(@as(usize, 4), h.len);
    try testing.expectEqualStrings(":status", h[0].name);
    try testing.expectEqualStrings("200", h[0].value);
    try testing.expectEqualStrings("content-type", h[1].name);
    try testing.expectEqualStrings("application/grpc", h[1].value);
    try testing.expectEqualStrings("grpc-status", h[2].name);
    try testing.expectEqualStrings("0", h[2].value);
    try testing.expectEqualStrings("grpc-message", h[3].name);
    try testing.expectEqualStrings("", h[3].value);
}

test "encoder uses a single byte for a full static match" {
    var buf: [64]u8 = undefined;
    // :status 200 is static index 8.
    const n = try Encoder.writeHeader(&buf, ":status", "200");
    try testing.expectEqual(@as(usize, 1), n);
    try testing.expectEqual(@as(u8, 0x88), buf[0]);
}

test "encoder never inserts into a dynamic table, so it stays stateless" {
    // Encoding the same headers twice must produce identical bytes. An encoder
    // that indexed would produce different (shorter) output the second time,
    // and would then depend on the peer's decoder tracking it.
    var a: [256]u8 = undefined;
    var b: [256]u8 = undefined;
    const na = try Encoder.writeHeader(&a, "grpc-message", "collection not found");
    const nb = try Encoder.writeHeader(&b, "grpc-message", "collection not found");
    try testing.expectEqual(na, nb);
    try testing.expectEqualSlices(u8, a[0..na], b[0..nb]);
}

test "dynamic table evicts by accounted size, not entry count" {
    var t = DynamicTable.init();
    t.setMaxSize(100);
    // Each entry is 32 + name + value. Two 20-byte entries fit (2 x 52 = 104 >
    // 100), so the second evicts the first.
    _ = t.insert("aaaaaaaaaa", "aaaaaaaaaa"); // 52
    try testing.expectEqual(@as(usize, 1), t.count);
    _ = t.insert("bbbbbbbbbb", "bbbbbbbbbb"); // 52; 104 > 100 -> evict
    try testing.expectEqual(@as(usize, 1), t.count);
    try testing.expectEqualStrings("bbbbbbbbbb", t.get(0).?.name);
}

test "dynamic table: an oversized entry empties the table per RFC 7541 §4.4" {
    var t = DynamicTable.init();
    t.setMaxSize(100);
    _ = t.insert("small", "value");
    try testing.expectEqual(@as(usize, 1), t.count);
    const big = [_]u8{'x'} ** 200;
    _ = t.insert(&big, "v");
    try testing.expectEqual(@as(usize, 0), t.count);
    try testing.expectEqual(@as(usize, 0), t.size);

    // The same at the default 4096-byte size, with an entry too large for
    // the staging buffer as well: it used to be refused *without* emptying
    // the table, leaving every index one off from the peer's view.
    var t2 = DynamicTable.init();
    try testing.expect(t2.insert("a", "b"));
    try testing.expectEqual(@as(usize, 1), t2.count);
    const huge = [_]u8{'y'} ** (max_entry_bytes + 1);
    try testing.expect(t2.insert(&huge, ""));
    try testing.expectEqual(@as(usize, 0), t2.count);
    try testing.expectEqual(@as(usize, 0), t2.size);
    try testing.expectEqual(@as(usize, 0), t2.storage_used);
    // And through the decoder: a literal-with-indexing that large must
    // decode fine and leave the table empty, not desynchronised.
    var dec = Decoder.init();
    var blk: [16]u8 = undefined;
    var n: usize = 0;
    n += encodeInteger(blk[n..], 6, 0x40, 4); // literal w/ indexing, name = :path
    n += encodeInteger(blk[n..], 7, 0x00, 4096);
    var wire: [16 + 4096]u8 = undefined;
    @memcpy(wire[0..n], blk[0..n]);
    @memset(wire[n..][0..4096], 'z');
    const hs = try dec.decode(wire[0 .. n + 4096]);
    try testing.expectEqual(@as(usize, 1), hs.len);
    try testing.expectEqual(@as(usize, 0), dec.table.count);
}

test "dynamic table size update is honoured and bounded" {
    var dec = Decoder.init();
    // 0x20 | 31 followed by continuation -> size update. Use 0x3f 0xe1 0x1f
    // = prefix 5 bits all set, value 4096.
    var buf: [8]u8 = undefined;
    const n = encodeInteger(&buf, 5, 0x20, 4096);
    _ = try dec.decode(buf[0..n]);
    try testing.expectEqual(@as(usize, 4096), dec.table.max_size);

    // Larger than the default SETTINGS value must be refused.
    const n2 = encodeInteger(&buf, 5, 0x20, 8192);
    try testing.expectError(Error.TableSizeExceeded, dec.decode(buf[0..n2]));
}

test "decoder bounds the header count" {
    var dec = Decoder.init();
    var buf: [1024]u8 = undefined;
    var n: usize = 0;
    for (0..max_headers + 5) |_| {
        n += encodeInteger(buf[n..], 7, 0x80, 2); // :method GET
    }
    try testing.expectError(Error.TooManyHeaders, dec.decode(buf[0..n]));
}

test "a literal naming a dynamic entry survives table compaction later in the block" {
    // Header 2 takes its *name* from dynamic index 62 (header 1's entry),
    // which used to be reported as a slice into the table's bump arena. The
    // rest of the block inserts enough to run the 16 KiB arena out and
    // `compact()` rewrites it in place, so by the time the caller reads the
    // headers, header 2's name pointed at whatever now lives at that offset.
    var dec = Decoder.init();
    var buf: [24 * 1024]u8 = undefined;
    var n: usize = 0;

    // Header 1: literal with indexing, new name "x-first", 200-byte value.
    const val = [_]u8{'v'} ** 200;
    n += encodeInteger(buf[n..], 6, 0x40, 0);
    n += encodeInteger(buf[n..], 7, 0x00, 7);
    @memcpy(buf[n..][0..7], "x-first");
    n += 7;
    n += encodeInteger(buf[n..], 7, 0x00, val.len);
    @memcpy(buf[n..][0..val.len], &val);
    n += val.len;

    // Header 2: literal with indexing, name = dynamic index 62 ("x-first").
    n += encodeInteger(buf[n..], 6, 0x40, 62);
    n += encodeInteger(buf[n..], 7, 0x00, 6);
    @memcpy(buf[n..][0..6], "second");
    n += 6;

    // Header 3: literal *without* indexing, name = dynamic index 62 again
    // (now "x-first" from header 2's entry, same name).
    n += encodeInteger(buf[n..], 4, 0x00, 62);
    n += encodeInteger(buf[n..], 7, 0x00, 5);
    @memcpy(buf[n..][0..5], "third");
    n += 5;

    // Then 60 more indexed literals of ~280 bytes each: 16 KiB+ of strings,
    // more than the arena holds, so compaction runs mid-block. Each name is
    // distinct so an aliased slice would visibly change.
    const filler = [_]u8{'f'} ** 270;
    for (0..60) |i| {
        var name: [8]u8 = undefined;
        _ = std.fmt.bufPrint(&name, "n{d:0>7}", .{i}) catch unreachable;
        n += encodeInteger(buf[n..], 6, 0x40, 0);
        n += encodeInteger(buf[n..], 7, 0x00, name.len);
        @memcpy(buf[n..][0..name.len], &name);
        n += name.len;
        n += encodeInteger(buf[n..], 7, 0x00, filler.len);
        @memcpy(buf[n..][0..filler.len], &filler);
        n += filler.len;
    }

    const hdrs = try dec.decode(buf[0..n]);
    try testing.expectEqual(@as(usize, 63), hdrs.len);
    try testing.expectEqualStrings("x-first", hdrs[0].name);
    try testing.expectEqualStrings("x-first", hdrs[1].name);
    try testing.expectEqualStrings("second", hdrs[1].value);
    try testing.expectEqualStrings("x-first", hdrs[2].name);
    try testing.expectEqualStrings("third", hdrs[2].value);
    // The table did compact: its arena is smaller than the sum of every
    // string inserted, and the oldest entries have been evicted.
    try testing.expect(dec.table.storage_used < 60 * 278);
    try testing.expect(dec.table.count < 62);
}

test "an indexed dynamic header survives table compaction later in the block" {
    // Header 1 inserts (":path", 200-byte value). Header 2 is §6.1 indexed
    // 62, naming that entry: name *and* value used to be reported as slices
    // into the table's storage. The rest of the block inserts more than the
    // 16 KiB table arena holds, so `compact()` rewrites it mid-block and an
    // aliased header 2 would read as filler bytes.
    var dec = Decoder.init();
    var buf: [24 * 1024]u8 = undefined;
    var n: usize = 0;

    const val = [_]u8{'p'} ** 200;
    n += encodeInteger(buf[n..], 6, 0x40, 4); // name = static 4, ":path"
    n += encodeInteger(buf[n..], 7, 0x00, val.len);
    @memcpy(buf[n..][0..val.len], &val);
    n += val.len;

    n += encodeInteger(buf[n..], 7, 0x80, 62); // indexed dynamic

    const filler = [_]u8{'f'} ** 270;
    for (0..60) |i| {
        var name: [8]u8 = undefined;
        _ = std.fmt.bufPrint(&name, "n{d:0>7}", .{i}) catch unreachable;
        n += encodeInteger(buf[n..], 6, 0x40, 0);
        n += encodeInteger(buf[n..], 7, 0x00, name.len);
        @memcpy(buf[n..][0..name.len], &name);
        n += name.len;
        n += encodeInteger(buf[n..], 7, 0x00, filler.len);
        @memcpy(buf[n..][0..filler.len], &filler);
        n += filler.len;
    }

    const hdrs = try dec.decode(buf[0..n]);
    try testing.expectEqual(@as(usize, 62), hdrs.len);
    try testing.expectEqualStrings(":path", hdrs[0].name);
    try testing.expectEqualStrings(&val, hdrs[0].value);
    try testing.expectEqualStrings(":path", hdrs[1].name);
    try testing.expectEqualStrings(&val, hdrs[1].value);
    try testing.expect(dec.table.storage_used < 60 * 278);
}
