//! TurboQuant, as Qdrant 1.19 serves it through `TurboQuantization { bits }`.
//!
//! The reference is Qdrant's `lib/quantization/src/encoded_vectors_tq.rs` and
//! `turboquant/` at `878843e6`, on the path its collection API takes: TQ+
//! (per-coordinate error correction) for every width, the rotation over the
//! whole zero-padded vector. A vector is padded, rotated by a fixed randomised
//! Hadamard transform so its coordinates are near N(0, 1) after a length
//! rescale, shifted and scaled per coordinate (TQ+), and each coordinate is
//! coded against a fixed Lloyd-Max codebook for N(0, 1). A query is rotated
//! and corrected the same way but kept in floating point, and scored against
//! the codes asymmetrically.
//!
//! Matched to Qdrant: the padding rule, the rotation (block-diagonal
//! Walsh-Hadamard over power-of-two chunks, three permutations from a Knuth
//! MMIX LCG at Qdrant's seeds), the codebooks and their midpoint boundaries,
//! the TQ+ quantile targets, the per-vector scalars (`sf`, `l2`, `xm`) and the
//! score formulas.
//!
//! Not matched, and why it does not matter to a comparison:
//!   * Qdrant estimates each coordinate's quantiles with a streaming P-square
//!     estimator over vectors drawn by an OS-seeded RNG, so its own codes
//!     differ from one build to the next. This takes the exact quantiles of a
//!     deterministic strided sample of the same size: the same targets,
//!     reproducibly.
//!   * Qdrant integerises the query (16-bit) and scores against an integer
//!     rounding of the codebook (4-bit's top centroid becomes 2.7117 for
//!     2.733). This keeps the query and the codebook in f32, which is a
//!     little more precise and does not change what the codes are.

const std = @import("std");
const builtin = @import("builtin");

/// Bits per coordinate, as Qdrant's `TurboQuantBitSize` names them.
pub const Bits = enum {
    b1,
    b1_5,
    b2,
    b4,

    /// Code bits per padded coordinate: 1.5 bits is one-bit codes over a
    /// vector padded to 1.5x its width (`TQ/mod.rs:24-32`).
    pub fn codeBits(self: Bits) usize {
        return switch (self) {
            .b1, .b1_5 => 1,
            .b2 => 2,
            .b4 => 4,
        };
    }

    /// The padded width `D` the rotation spans (`TQ/encoding.rs:195-202`).
    pub fn paddedDim(self: Bits, d: usize) usize {
        return switch (self) {
            .b1 => std.mem.alignForward(usize, d, 8),
            .b1_5 => std.mem.alignForward(usize, d * 3 / 2, 8),
            .b2 => std.mem.alignForward(usize, d, 4),
            .b4 => std.mem.alignForward(usize, d, 2),
        };
    }

    /// Code bytes per stored vector.
    pub fn codeBytes(self: Bits, d: usize) usize {
        return self.paddedDim(d) * self.codeBits() / 8;
    }

    /// Vectors the TQ+ fit samples (`TQ/mod.rs:62-68`).
    pub fn sampleSize(self: Bits) usize {
        return switch (self) {
            .b1, .b1_5 => 2048,
            .b2 => 4096,
            .b4 => 8192,
        };
    }

    /// The fixed Lloyd-Max codebook for N(0, 1) (`TQ/lloyd_max.rs:3-29`).
    pub fn centroids(self: Bits) []const f32 {
        return switch (self) {
            .b1, .b1_5 => &c1,
            .b2 => &c2,
            .b4 => &c4,
        };
    }

    fn boundaries(self: Bits) []const f32 {
        return switch (self) {
            .b1, .b1_5 => &b1_bounds,
            .b2 => &b2_bounds,
            .b4 => &b4_bounds,
        };
    }

    /// The TQ+ lower quantile target, `Φ(−max|C|)`: `(1 − quantile) / 2` with
    /// Qdrant's `quantile = 2·Φ(c_outer) − 1` (`Q/encoded_vectors_tq.rs:158-237`,
    /// its erf approximation evaluated).
    fn loProb(self: Bits) f64 {
        return switch (self) {
            .b1, .b1_5 => 0.2124687,
            .b2 => 0.0655217,
            .b4 => 0.0031381,
        };
    }
};

const c1 = [_]f32{ -0.7978846, 0.7978846 };
const c2 = [_]f32{ -1.510, -0.4528, 0.4528, 1.510 };
const c4 = [_]f32{
    -2.733, -2.069, -1.618, -1.256, -0.9424, -0.6568, -0.3881, -0.1284,
    0.1284, 0.3881, 0.6568, 0.9424, 1.256,   1.618,   2.069,   2.733,
};
const b1_bounds = midpoints(&c1);
const b2_bounds = midpoints(&c2);
const b4_bounds = midpoints(&c4);

fn midpoints(comptime c: []const f32) [c.len - 1]f32 {
    var out: [c.len - 1]f32 = undefined;
    for (0..c.len - 1) |i| out[i] = (c[i] + c[i + 1]) / 2;
    return out;
}

/// A value's code: the number of boundaries it strictly exceeds, ties to the
/// lower code (`TQ/encoding.rs:124-128`).
fn codeOf(bounds: []const f32, v: f32) u8 {
    var k: u8 = 0;
    for (bounds) |b| {
        if (v > b) k += 1;
    }
    return k;
}

// -------------------------------------------------------------------------
// Rotation
// -------------------------------------------------------------------------

/// Qdrant's rotation seeds and LCG (`TQ/rotation.rs:10`, `TQ/permutation.rs`).
const permutation_seeds = [3]u64{ 654605292835415893, 8636605637963351413, 1775280196666917949 };
const lcg_a: u64 = 6364136223846793005;
const lcg_c: u64 = 1442695040888963407;

/// `R = H·P2·H·P1·H·P0·H` over `n` coordinates: `H` the normalised
/// Walsh-Hadamard transform applied to each power-of-two chunk of the buffer
/// (the set bits of `n`, largest first), each `P` a Fisher-Yates permutation.
pub const Rotation = struct {
    n: usize,
    /// Three gather maps, `dst[k] = src[map[k]]`, back to back.
    maps: []u32,

    pub fn init(alloc: std.mem.Allocator, n: usize) !Rotation {
        const maps = try alloc.alloc(u32, 3 * n);
        for (0..3) |p| {
            const m = maps[p * n ..][0..n];
            for (m, 0..) |*x, i| x.* = @intCast(i);
            var state = permutation_seeds[p];
            var i = n;
            while (i > 1) {
                i -= 1;
                state = state *% lcg_a +% lcg_c;
                const j: usize = @intCast((state >> 32) % (i + 1));
                std.mem.swap(u32, &m[i], &m[j]);
            }
        }
        return .{ .n = n, .maps = maps };
    }

    pub fn deinit(self: *Rotation, alloc: std.mem.Allocator) void {
        alloc.free(self.maps);
    }

    /// Rotate `x` in place; `tmp` is scratch of the same length.
    pub fn apply(self: *const Rotation, x: []f64, tmp: []f64) void {
        std.debug.assert(x.len == self.n and tmp.len >= self.n);
        whtChunks(x);
        for (0..3) |p| {
            const m = self.maps[p * self.n ..][0..self.n];
            for (tmp[0..self.n], m) |*t, k| t.* = x[k];
            @memcpy(x, tmp[0..self.n]);
            whtChunks(x);
        }
    }
};

/// The normalised Walsh-Hadamard transform of each power-of-two chunk, in
/// f64 (`TQ/rotation.rs:158-173, 222-235, 269-281`).
fn whtChunks(x: []f64) void {
    var off: usize = 0;
    var rest = x.len;
    while (rest > 0) {
        const s = std.math.floorPowerOfTwo(usize, rest);
        const c = x[off..][0..s];
        var h: usize = 1;
        while (h < s) : (h *= 2) {
            var i: usize = 0;
            while (i < s) : (i += 2 * h) {
                for (i..i + h) |j| {
                    const a = c[j];
                    const b = c[j + h];
                    c[j] = a + b;
                    c[j + h] = a - b;
                }
            }
        }
        const norm = 1.0 / @sqrt(@as(f64, @floatFromInt(s)));
        for (c) |*v| v.* *= norm;
        off += s;
        rest -= s;
    }
}

// -------------------------------------------------------------------------
// The trained model
// -------------------------------------------------------------------------

/// What distinguishes the per-vector scalars: Qdrant keeps Cosine apart from
/// Dot for TurboQuant, and only Euclid stores the raw norm.
pub const Kind = enum { cosine, dot, euclid };

/// Everything encoding and scoring need beside the codes.
pub const Model = struct {
    bits: Bits,
    kind: Kind,
    dim: usize,
    padded: usize,
    rot: Rotation,
    /// TQ+: per padded coordinate, `(x + shift) · scale` before coding.
    shift: []f32,
    scale: []f32,

    pub fn deinit(self: *Model, alloc: std.mem.Allocator) void {
        self.rot.deinit(alloc);
        alloc.free(self.shift);
        alloc.free(self.scale);
    }

    /// Scratch `encode` and `prepareQuery` need: two f64 buffers of `padded`.
    pub fn scratchLen(self: *const Model) usize {
        return 2 * self.padded;
    }

    /// Steps 1-4 of the stored encoding: pad, rotate, and (not for cosine)
    /// rescale to `√D` length. Returns the raw norm, or null for cosine.
    fn rotateRescaled(self: *const Model, v: []const f32, buf: []f64, tmp: []f64) ?f32 {
        for (buf[0..v.len], v) |*b, x| b.* = x;
        @memset(buf[v.len..self.padded], 0);
        self.rot.apply(buf[0..self.padded], tmp);
        var l2: ?f32 = null;
        if (self.kind != .cosine) {
            var s: f64 = 0;
            for (buf[0..self.padded]) |x| s += x * x;
            l2 = @floatCast(@sqrt(s));
        }
        const len: f64 = if (l2) |l| l else 1.0;
        if (len > 0) {
            const k = @sqrt(@as(f64, @floatFromInt(self.padded))) / len;
            for (buf[0..self.padded]) |*x| x.* *= k;
        }
        return l2;
    }

    /// Encode `v` into `codes` (`codeBytes` bytes) and its three scalars
    /// (`TQ/quantization.rs:172-293`).
    pub fn encode(self: *const Model, v: []const f32, codes: []u8, scratch: []f64) Scalars {
        const buf = scratch[0..self.padded];
        const tmp = scratch[self.padded..][0..self.padded];
        const l2 = self.rotateRescaled(v, buf, tmp);
        var energy: f64 = 0;
        for (buf) |x| energy += x * x;
        var xm: f32 = 0;
        if (energy >= 1e-12) {
            var acc: f64 = 0;
            for (buf, self.shift) |x, sh| acc += x * -@as(f64, sh);
            xm = @floatCast(acc);
            for (buf, self.shift, self.scale) |*x, sh, sc| x.* = (x.* + sh) * sc;
        }
        const bounds = self.bits.boundaries();
        const cents = self.bits.centroids();
        const cb = self.bits.codeBits();
        @memset(codes, 0);
        var cn2: f64 = 0;
        for (buf, 0..) |x, i| {
            const code = codeOf(bounds, @floatCast(x));
            const bit = i * cb;
            codes[bit / 8] |= code << @intCast(bit % 8);
            const r = @as(f64, cents[code]) / @as(f64, self.scale[i]) - @as(f64, self.shift[i]);
            cn2 += r * r;
        }
        var cn: f32 = @floatCast(@sqrt(cn2));
        if (self.kind == .cosine and energy < 1e-12) cn = @floatCast(@sqrt(@as(f64, @floatFromInt(self.padded))));
        const sf: f32 = if (cn == 0) 0 else switch (self.kind) {
            .cosine => 1.0 / cn,
            .dot, .euclid => l2.? / cn,
        };
        return .{ .sf = sf, .l2 = l2 orelse 0, .xm = xm };
    }

    /// The query side (`TQ/quantization.rs:501-569`): rotated, not rescaled,
    /// TQ+-corrected, kept in f32. `out` is `padded` long.
    pub fn prepareQuery(self: *const Model, q: []const f32, out: []f32, scratch: []f64) PreparedQuery {
        const buf = scratch[0..self.padded];
        const tmp = scratch[self.padded..][0..self.padded];
        for (buf[0..q.len], q) |*b, x| b.* = x;
        @memset(buf[q.len..], 0);
        self.rot.apply(buf, tmp);
        var n2: f64 = 0;
        var ec: f64 = 0;
        var sum: f32 = 0;
        for (buf, self.shift, self.scale, out[0..self.padded]) |x, sh, sc, *o| {
            n2 += x * x;
            ec += x * -@as(f64, sh);
            o.* = @floatCast(x / sc);
            sum += o.*;
        }
        return .{ .q = out[0..self.padded], .ec = @floatCast(ec), .qn2 = @floatCast(n2), .sum = sum };
    }

    /// Score one stored vector: Qdrant's `score_from_raw_dot` before its
    /// `invert`, as a similarity, higher better (`TQ/quantization.rs:574-696`).
    pub fn score(self: *const Model, pq: *const PreparedQuery, codes: []const u8, s: Scalars) f32 {
        const dot = rawDot(self.bits, pq, codes) + pq.ec;
        return switch (self.kind) {
            .cosine, .dot => dot * s.sf,
            .euclid => -(pq.qn2 + s.l2 * s.l2 - 2 * dot * s.sf),
        };
    }
};

/// The per-vector scalars Qdrant stores after the codes.
pub const Scalars = struct { sf: f32, l2: f32, xm: f32 };

pub const PreparedQuery = struct {
    q: []const f32,
    ec: f32,
    qn2: f32,
    /// `Σ q`, for the one-bit kernel.
    sum: f32,
    /// The integer form the 2- and 4-bit kernels score (`prepareInt`): empty
    /// where the target has no `vpshufb` and `vpdpbusd` at 512 bits.
    planes: []const i8 = &.{},
    /// `Σ qi`, the integer query's sum, for the codebook's offset.
    sum_i: i64 = 0,
    /// `1 / (q_scale · c_scale)`.
    post: f32 = 0,
};

/// Qdrant's x86 integer codebooks, offset 128 (`query4bit/mod.rs:57-73`,
/// `query2bit/mod.rs:52-60`), and their scales `128 / max|C|`.
const int4 = [16]u8{ 0, 31, 52, 69, 84, 97, 110, 122, 134, 146, 159, 172, 187, 204, 225, 255 };
const int2 = [4]u8{ 0, 90, 166, 255 };
const int4_scale: f32 = 128.0 / 2.733;
const int2_scale: f32 = 128.0 / 1.510;
/// Qdrant's x86 query range for the 2- and 4-bit kernels.
const qmax: f32 = 8127;

/// Coordinates per block of the integer kernel: 64 code bytes.
fn blockCoords(bits: Bits) usize {
    return if (bits == .b4) 128 else 256;
}

/// Plane bytes `prepareInt` needs for a model's `padded` width.
pub fn planeBytes(bits: Bits, padded: usize) usize {
    return if (bits == .b4 or bits == .b2) 2 * std.mem.alignForward(usize, padded, blockCoords(bits)) else 0;
}

pub fn hasIntKernel() bool {
    return builtin.zig_backend == .stage2_llvm and builtin.cpu.has(.x86, .avx512bw) and
        builtin.cpu.has(.x86, .avx512vnni);
}

/// Integerise the prepared query for the 2- and 4-bit kernels: Qdrant's
/// `QMAX / max|q|` scale, each value split as `128·hi + lo` with `hi` signed
/// and `lo` in [0, 127], both bytes, laid out block by block in the order the
/// code stream is unpacked (`dotInt`). Zero past `padded`.
pub fn prepareInt(pq: *PreparedQuery, bits: Bits, planes: []i8) void {
    if (!(bits == .b4 or bits == .b2) or !comptime hasIntKernel()) return;
    const q = pq.q;
    var amax: f32 = std.math.floatEps(f32);
    for (q) |x| amax = @max(amax, @abs(x));
    const q_scale = qmax / amax;
    const bc = blockCoords(bits);
    const streams: usize = if (bits == .b4) 2 else 4;
    const per = bc / streams; // coordinates per stream per block: 64
    const out = planes[0..planeBytes(bits, q.len)];
    @memset(out, 0);
    var sum: i64 = 0;
    for (q, 0..) |x, i| {
        const r = @round(x * q_scale); // half away from zero, as Qdrant's
        const qi: i32 = @intFromFloat(std.math.clamp(r, -qmax, qmax));
        sum += qi;
        const hi: i32 = @divFloor(qi, 128);
        const lo: i32 = qi - 128 * hi;
        const block = i / bc;
        const within = i % bc;
        const stream = within % streams;
        const lane = within / streams;
        const base = block * 2 * bc + stream * 2 * per;
        out[base + lane] = @intCast(hi);
        out[base + per + lane] = @intCast(lo);
    }
    pq.planes = out;
    pq.sum_i = sum;
    pq.post = 1.0 / (q_scale * (if (bits == .b4) int4_scale else int2_scale));
}

/// Train a model: the rotation, then the TQ+ shift and scale from the exact
/// per-coordinate quantiles of `sample` (rows of `dim` f32).
pub fn train(alloc: std.mem.Allocator, bits: Bits, kind: Kind, dim: usize, sample: []const []const f32) !Model {
    const padded = bits.paddedDim(dim);
    var rot = try Rotation.init(alloc, padded);
    errdefer rot.deinit(alloc);
    const shift = try alloc.alloc(f32, padded);
    errdefer alloc.free(shift);
    const scale = try alloc.alloc(f32, padded);
    errdefer alloc.free(scale);
    var model = Model{ .bits = bits, .kind = kind, .dim = dim, .padded = padded, .rot = rot, .shift = shift, .scale = scale };
    @memset(shift, 0);
    @memset(scale, 1);
    const n = sample.len;
    if (n == 0) return model;

    // Column-major rotated, rescaled sample: `cols[i * n + r]`.
    const cols = try alloc.alloc(f32, padded * n);
    defer alloc.free(cols);
    const scratch = try alloc.alloc(f64, 2 * padded);
    defer alloc.free(scratch);
    for (sample, 0..) |v, r| {
        _ = model.rotateRescaled(v, scratch[0..padded], scratch[padded..]);
        for (scratch[0..padded], 0..) |x, i| cols[i * n + r] = @floatCast(x);
    }
    const lo_p = bits.loProb();
    const c_outer = bits.centroids()[bits.centroids().len - 1];
    for (0..padded) |i| {
        const col = cols[i * n ..][0..n];
        std.mem.sort(f32, col, {}, std.sort.asc(f32));
        const lo = quantile(col, lo_p);
        const hi = quantile(col, 1 - lo_p);
        shift[i] = -(lo + hi) / 2;
        scale[i] = if (hi - lo > 1e-3) (2 * c_outer) / (hi - lo) else 1.0;
    }
    return model;
}

/// Linear-interpolated quantile of sorted `v`.
fn quantile(v: []const f32, p: f64) f32 {
    if (v.len == 1) return v[0];
    const k = p * @as(f64, @floatFromInt(v.len - 1));
    const i: usize = @intFromFloat(@floor(k));
    const j = @min(i + 1, v.len - 1);
    const t: f32 = @floatCast(k - @floor(k));
    return v[i] + (v[j] - v[i]) * t;
}

// -------------------------------------------------------------------------
// Kernels: Σ q_i · C[code_i]
// -------------------------------------------------------------------------

fn hasVpermps() bool {
    return builtin.zig_backend == .stage2_llvm and builtin.cpu.has(.x86, .avx512f);
}

/// `out[j] = tbl[idx[j] & 15]`: one `vpermps zmm`, the whole 4-bit codebook
/// in one register (Zig has no runtime-indexed `@shuffle`, see `pq_adc.zig`).
inline fn permute16(tbl: @Vector(16, f32), idx: @Vector(16, u32)) @Vector(16, f32) {
    return asm ("vpermps %[tbl], %[idx], %[out]"
        : [out] "=v" (-> @Vector(16, f32)),
        : [tbl] "v" (tbl),
          [idx] "v" (idx),
    );
}

pub fn rawDot(bits: Bits, pq: *const PreparedQuery, codes: []const u8) f32 {
    const q = pq.q;
    if (comptime hasIntKernel()) {
        if (pq.planes.len > 0) return switch (bits) {
            .b4 => dotInt(4, pq, codes),
            .b2 => dotInt(2, pq, codes),
            .b1, .b1_5 => unreachable,
        };
    }
    return switch (bits) {
        .b1, .b1_5 => dot1(q, pq.sum, codes),
        .b2 => dotLut(2, q, codes),
        .b4 => dotLut(4, q, codes),
    };
}

/// One-bit codes: `C1 · Σ q_i·(2b_i − 1) = C1 · (2·Σ_{b_i=1} q_i − Σq)`.
fn dot1(q: []const f32, sum: f32, codes: []const u8) f32 {
    const V = @Vector(16, f32);
    const zero: V = @splat(0);
    var acc = [4]V{ zero, zero, zero, zero };
    var i: usize = 0;
    while (i + 64 <= q.len) : (i += 64) {
        inline for (0..4) |u| {
            const at = i + 16 * u;
            const m: u16 = @as(u16, codes[at / 8]) | (@as(u16, codes[at / 8 + 1]) << 8);
            const qv: V = q[at..][0..16].*;
            acc[u] += @select(f32, @as(@Vector(16, bool), @bitCast(m)), qv, zero);
        }
    }
    while (i + 16 <= q.len) : (i += 16) {
        const m: u16 = @as(u16, codes[i / 8]) | (@as(u16, codes[i / 8 + 1]) << 8);
        const qv: V = q[i..][0..16].*;
        acc[0] += @select(f32, @as(@Vector(16, bool), @bitCast(m)), qv, zero);
    }
    var ones = @reduce(.Add, (acc[0] + acc[1]) + (acc[2] + acc[3]));
    while (i < q.len) : (i += 1) {
        if ((codes[i / 8] >> @intCast(i % 8)) & 1 == 1) ones += q[i];
    }
    return c1[1] * (2 * ones - sum);
}

/// `Σ qi · (c_u8 − 128)` over the integer planes, 64 code bytes a step:
/// `vpshufb` maps every code to Qdrant's integer centroid, and `vpdpbusd`
/// multiplies them into each plane. The codes are read in whole 64-byte
/// steps; the store keeps 64 bytes of slack after its last row, and the
/// planes are zero past the vector, so the bytes past a row contribute
/// nothing.
fn dotInt(comptime cb: comptime_int, pq: *const PreparedQuery, codes: []const u8) f32 {
    const B = @Vector(64, u8);
    const S = @Vector(64, i8);
    const A = @Vector(16, i32);
    const tbl_arr: [16]u8 = blk: {
        var t: [16]u8 = @splat(0);
        if (cb == 4) t = int4 else {
            for (int2, 0..) |v, k| t[k] = v;
        }
        break :blk t;
    };
    var tbl4: [64]u8 = undefined;
    for (0..4) |l| @memcpy(tbl4[l * 16 ..][0..16], &tbl_arr);
    const tbl: B = tbl4;
    const streams = 8 / cb;
    const bc: usize = 64 * streams;
    const mask: B = @splat((1 << cb) - 1);
    var acc_hi: A = @splat(0);
    var acc_lo: A = @splat(0);
    var i: usize = 0;
    var off: usize = 0;
    const planes = pq.planes;
    while (i < pq.q.len) : (i += bc) {
        const raw: B = codes.ptr[off..][0..64].*;
        const base = (i / bc) * 2 * bc;
        inline for (0..streams) |k| {
            const idx = (raw >> @as(@Vector(64, u3), @splat(@intCast(k * cb)))) & mask;
            const cu = shuffle64(tbl, idx);
            const hi: S = planes[base + k * 128 ..][0..64].*;
            const lo: S = planes[base + k * 128 + 64 ..][0..64].*;
            acc_hi = dpbusd(acc_hi, cu, hi);
            acc_lo = dpbusd(acc_lo, cu, lo);
        }
        off += 64;
    }
    const s: i64 = 128 * @as(i64, @reduce(.Add, acc_hi)) + @as(i64, @reduce(.Add, acc_lo)) - 128 * pq.sum_i;
    return pq.post * @as(f32, @floatFromInt(s));
}

inline fn shuffle64(tbl: @Vector(64, u8), idx: @Vector(64, u8)) @Vector(64, u8) {
    return asm ("vpshufb %[idx], %[tbl], %[out]"
        : [out] "=v" (-> @Vector(64, u8)),
        : [tbl] "v" (tbl),
          [idx] "v" (idx),
    );
}

inline fn dpbusd(acc: @Vector(16, i32), a: @Vector(64, u8), b: @Vector(64, i8)) @Vector(16, i32) {
    return asm ("vpdpbusd %[b], %[a], %[acc]"
        : [acc] "=v" (-> @Vector(16, i32)),
        : [_] "0" (acc),
          [a] "v" (a),
          [b] "v" (b),
    );
}

/// Sixteen codes as `u32` lanes, whole-vector: masks and shifts over the
/// bytes, one comptime interleave, one zero-extension. Built lane by lane it
/// was sixteen extracts and inserts per `vpermps`, and 4-bit read 0.37x
/// Qdrant (dbpedia-100K, 2026-09-30).
inline fn unpack(comptime cb: comptime_int, raw: @Vector(16 * cb / 8, u8)) @Vector(16, u32) {
    if (cb == 4) {
        // byte j holds codes 2j (low nibble) and 2j + 1 (high).
        const lo = raw & @as(@Vector(8, u8), @splat(0x0f));
        const hi = raw >> @as(@Vector(8, u3), @splat(4));
        const m = comptime blk: {
            var mm: [16]i32 = undefined;
            for (0..8) |j| {
                mm[2 * j] = j;
                mm[2 * j + 1] = ~@as(i32, j);
            }
            break :blk mm;
        };
        const v: @Vector(16, u8) = @shuffle(u8, lo, hi, m);
        return v;
    }
    // Two bits: byte j holds codes 4j..4j+3 at shifts 0, 2, 4, 6.
    const mask: @Vector(4, u8) = @splat(0x03);
    const k0 = raw & mask;
    const k1 = (raw >> @as(@Vector(4, u3), @splat(2))) & mask;
    const k2 = (raw >> @as(@Vector(4, u3), @splat(4))) & mask;
    const k3 = raw >> @as(@Vector(4, u3), @splat(6));
    const pair = comptime blk: {
        var mm: [8]i32 = undefined;
        for (0..4) |j| {
            mm[2 * j] = j;
            mm[2 * j + 1] = ~@as(i32, j);
        }
        break :blk mm;
    };
    const ab: @Vector(8, u8) = @shuffle(u8, k0, k1, pair); // k0[j], k1[j]
    const cd: @Vector(8, u8) = @shuffle(u8, k2, k3, pair); // k2[j], k3[j]
    const quad = comptime blk: {
        var mm: [16]i32 = undefined;
        for (0..4) |j| {
            mm[4 * j] = 2 * j;
            mm[4 * j + 1] = 2 * j + 1;
            mm[4 * j + 2] = ~@as(i32, 2 * j);
            mm[4 * j + 3] = ~@as(i32, 2 * j + 1);
        }
        break :blk mm;
    };
    const v: @Vector(16, u8) = @shuffle(u8, ab, cd, quad);
    return v;
}

/// Two- and four-bit codes, 16 coordinates at a time through `vpermps`.
fn dotLut(comptime cb: comptime_int, q: []const f32, codes: []const u8) f32 {
    const cents = if (cb == 4) c4[0..] else c2[0..];
    var i: usize = 0;
    var total: f32 = 0;
    if (comptime hasVpermps()) {
        var tbl_arr: [16]f32 = @splat(0);
        for (cents, 0..) |c, k| tbl_arr[k] = c;
        const tbl: @Vector(16, f32) = tbl_arr;
        // Four accumulators and fused multiply-adds: one accumulator and a
        // separate `vmulps` + `vaddps` serialised every step on the add's
        // latency, 71% of 4-bit's search in `score` (dbpedia-100K, W7).
        const V = @Vector(16, f32);
        var acc = [4]V{ @splat(0), @splat(0), @splat(0), @splat(0) };
        const per_byte = 8 / cb;
        const bytes = 16 / per_byte;
        while (i + 64 <= q.len) : (i += 64) {
            inline for (0..4) |u| {
                const at = i + 16 * u;
                const raw: @Vector(bytes, u8) = codes[at / per_byte ..][0..bytes].*;
                const qv: V = q[at..][0..16].*;
                acc[u] = @mulAdd(V, permute16(tbl, unpack(cb, raw)), qv, acc[u]);
            }
        }
        while (i + 16 <= q.len) : (i += 16) {
            const raw: @Vector(bytes, u8) = codes[i / per_byte ..][0..bytes].*;
            const qv: V = q[i..][0..16].*;
            acc[0] = @mulAdd(V, permute16(tbl, unpack(cb, raw)), qv, acc[0]);
        }
        total = @reduce(.Add, (acc[0] + acc[1]) + (acc[2] + acc[3]));
    }
    while (i < q.len) : (i += 1) {
        const bit = i * cb;
        const code = (codes[bit / 8] >> @intCast(bit % 8)) & ((1 << cb) - 1);
        total += q[i] * cents[code];
    }
    return total;
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

test "padding, code sizes and codebooks are Qdrant's" {
    try testing.expectEqual(@as(usize, 2304), Bits.b1_5.paddedDim(1536));
    try testing.expectEqual(@as(usize, 768), Bits.b1_5.paddedDim(512));
    try testing.expectEqual(@as(usize, 768), Bits.b4.codeBytes(1536));
    try testing.expectEqual(@as(usize, 384), Bits.b2.codeBytes(1536));
    try testing.expectEqual(@as(usize, 288), Bits.b1_5.codeBytes(1536));
    try testing.expectEqual(@as(usize, 192), Bits.b1.codeBytes(1536));
    try testing.expectEqual(@as(usize, 130), Bits.b4.paddedDim(129));
    // Boundaries are the midpoints, and codes are "boundaries exceeded".
    try testing.expectApproxEqAbs(@as(f32, -0.9814), b2_bounds[0], 1e-6);
    try testing.expectEqual(@as(u8, 0), codeOf(&b1_bounds, 0.0));
    try testing.expectEqual(@as(u8, 1), codeOf(&b1_bounds, 1e-9));
    try testing.expectEqual(@as(u8, 15), codeOf(&b4_bounds, 9.0));
    try testing.expectEqual(@as(u8, 7), codeOf(&b4_bounds, 0.0));
}

test "the rotation is orthonormal and its maps are permutations" {
    var rot = try Rotation.init(testing.allocator, 24);
    defer rot.deinit(testing.allocator);
    // Each map is a permutation.
    for (0..3) |p| {
        var seen = [_]bool{false} ** 24;
        for (rot.maps[p * 24 ..][0..24]) |k| seen[k] = true;
        for (seen) |s| try testing.expect(s);
    }
    // Norm preserved to f64 precision (chunks 16 + 8).
    var x: [24]f64 = undefined;
    var tmp: [24]f64 = undefined;
    var n0: f64 = 0;
    for (&x, 0..) |*v, i| {
        v.* = @as(f64, @floatFromInt(i)) - 7.5;
        n0 += v.* * v.*;
    }
    rot.apply(&x, &tmp);
    var n1: f64 = 0;
    for (x) |v| n1 += v * v;
    try testing.expectApproxEqRel(n0, n1, 1e-12);
}

test "the kernels agree with the scalar reference and the score tracks the dot product" {
    const dim = 64;
    var prng = std.Random.DefaultPrng.init(0x7b0);
    const rnd = prng.random();
    const n = 300;
    const data = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(data);
    for (data) |*x| x.* = rnd.floatNorm(f32);
    var rows: [n][]const f32 = undefined;
    for (&rows, 0..) |*r, i| r.* = data[i * dim ..][0..dim];
    inline for (.{ Bits.b1, Bits.b1_5, Bits.b2, Bits.b4 }) |bits| {
        var m = try train(testing.allocator, bits, .dot, dim, &rows);
        defer m.deinit(testing.allocator);
        const scratch = try testing.allocator.alloc(f64, m.scratchLen());
        defer testing.allocator.free(scratch);
        const codes = try testing.allocator.alloc(u8, bits.codeBytes(dim) * n);
        defer testing.allocator.free(codes);
        var sc: [n]Scalars = undefined;
        for (0..n) |r| sc[r] = m.encode(rows[r], codes[r * bits.codeBytes(dim) ..][0..bits.codeBytes(dim)], scratch);
        const qbuf = try testing.allocator.alloc(f32, m.padded);
        defer testing.allocator.free(qbuf);
        const pq = m.prepareQuery(rows[0], qbuf, scratch);
        // Rank agreement with the exact dot product against row 0.
        var agree: usize = 0;
        var pairs: usize = 0;
        for (1..80) |a| for (a + 1..80) |b| {
            const ea = dotF(rows[0], rows[a]);
            const eb = dotF(rows[0], rows[b]);
            const qa = m.score(&pq, codes[a * bits.codeBytes(dim) ..][0..bits.codeBytes(dim)], sc[a]);
            const qb = m.score(&pq, codes[b * bits.codeBytes(dim) ..][0..bits.codeBytes(dim)], sc[b]);
            pairs += 1;
            if ((ea > eb) == (qa > qb)) agree += 1;
        };
        const want: f64 = switch (bits) {
            .b4 => 0.95,
            .b2 => 0.88,
            .b1, .b1_5 => 0.75,
        };
        try testing.expect(@as(f64, @floatFromInt(agree)) / @as(f64, @floatFromInt(pairs)) > want);
        // The SIMD kernel and the scalar tail agree on a width with both.
        if (bits == .b4 or bits == .b2) {
            const cb = bits.codeBits();
            var ref: f32 = 0;
            const row = codes[3 * bits.codeBytes(dim) ..][0..bits.codeBytes(dim)];
            for (0..m.padded) |i| {
                const bit = i * cb;
                const code = (row[bit / 8] >> @intCast(bit % 8)) & ((@as(u8, 1) << @intCast(cb)) - 1);
                ref += pq.q[i] * bits.centroids()[code];
            }
            try testing.expectApproxEqRel(ref, rawDot(bits, &pq, row), 1e-4);
        }
    }
}

test "the integer kernel agrees with the float one to the integer codebook's rounding" {
    if (!comptime hasIntKernel()) return error.SkipZigTest;
    const dim = 300; // blocks of 128 and 256 with a tail in both
    var prng = std.Random.DefaultPrng.init(0x1a7);
    const rnd = prng.random();
    const n = 64;
    const data = try testing.allocator.alloc(f32, n * dim);
    defer testing.allocator.free(data);
    for (data) |*x| x.* = rnd.floatNorm(f32);
    var rows: [n][]const f32 = undefined;
    for (&rows, 0..) |*r, i| r.* = data[i * dim ..][0..dim];
    inline for (.{ Bits.b2, Bits.b4 }) |bits| {
        var m = try train(testing.allocator, bits, .dot, dim, &rows);
        defer m.deinit(testing.allocator);
        const scratch = try testing.allocator.alloc(f64, m.scratchLen());
        defer testing.allocator.free(scratch);
        const rb = bits.codeBytes(dim);
        const codes = try testing.allocator.alloc(u8, rb * n + 64);
        defer testing.allocator.free(codes);
        @memset(codes, 0xff); // slack garbage must not count
        for (0..n) |r| _ = m.encode(rows[r], codes[r * rb ..][0..rb], scratch);
        const qbuf = try testing.allocator.alloc(f32, m.padded);
        defer testing.allocator.free(qbuf);
        const planes = try testing.allocator.alloc(i8, planeBytes(bits, m.padded));
        defer testing.allocator.free(planes);
        var float_q = m.prepareQuery(rows[1], qbuf, scratch);
        var int_q = float_q;
        prepareInt(&int_q, bits, planes);
        try testing.expect(int_q.planes.len > 0);
        float_q.planes = &.{};
        var worst: f32 = 0;
        var scale: f32 = 0;
        for (0..n) |r| {
            const row = codes[r * rb ..][0..rb];
            const f = rawDot(bits, &float_q, row);
            const i = rawDot(bits, &int_q, row);
            worst = @max(worst, @abs(f - i));
            scale = @max(scale, @abs(f));
        }
        // The integer codebook rounds 4-bit's top centroid to 2.7117 for
        // 2.733: under 2% of the largest score.
        try testing.expect(worst < 0.02 * scale);
    }
}

fn dotF(a: []const f32, b: []const f32) f32 {
    var s: f32 = 0;
    for (a, b) |x, y| s += x * y;
    return s;
}
