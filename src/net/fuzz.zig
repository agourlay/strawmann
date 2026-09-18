//! §8.8, protocol fuzzing.
//!
//! > "**Protocol fuzzing:** malformed HTTP/2 frames, truncated HPACK, oversized
//! > length prefixes, wrong wire types, deeply nested messages, dimension
//! > mismatches. Target: **no crash, no hang, no UB**, a clean gRPC error. Run
//! > against a `Debug` build so Zig's safety checks are live."
//!
//! The target is the interesting part. These are not correctness tests, a
//! fuzzer has no oracle for what a malformed frame *should* decode to. What it
//! can check is that the decoder terminates, stays in bounds, and returns an
//! error rather than a plausible value. Zig's Debug-mode safety checks turn
//! every out-of-bounds access, integer overflow and invalid enum cast into a
//! panic, which is what makes "no UB" mechanically checkable rather than a
//! hope: `zig build test` runs these in Debug by default.
//!
//! The entry points take a `[]const u8` so `std.testing.fuzz` could drive
//! them, but nothing wires them to it and `build.zig` has no fuzz step: every
//! test below seeds its own `DefaultPrng` with a literal, so the whole file is
//! deterministic and the runner's `--seed` reaches none of it. Worth knowing
//! before chasing a flake: a `zig build test` failure here reproduces at any
//! seed or not at all.

const std = @import("std");
const h2 = @import("h2.zig");
const hpack = @import("hpack.zig");
const grpc = @import("grpc.zig");
const proto = @import("../proto/proto.zig");

/// Feed arbitrary bytes to the HPACK decoder.
///
/// Must terminate and must not read out of bounds, whatever the input. A
/// successful decode of garbage is fine, HPACK has no checksum, so many byte
/// strings are valid header blocks.
pub fn fuzzHpack(input: []const u8) void {
    var dec = hpack.Decoder.init();
    const headers = dec.decode(input) catch return;
    // Touch every result so a decoder that returned a slice past its arena
    // trips the bounds check rather than passing silently.
    var total: usize = 0;
    for (headers) |h| {
        total +%= h.name.len +% h.value.len;
        if (h.name.len > 0) total +%= h.name[0];
        if (h.value.len > 0) total +%= h.value[h.value.len - 1];
    }
    std.mem.doNotOptimizeAway(total);
}

/// Feed arbitrary bytes to the Huffman decoder.
pub fn fuzzHuffman(input: []const u8) void {
    var out: [4096]u8 = undefined;
    const n = hpack.huffmanDecode(&out, input[0..@min(input.len, 512)]) catch return;
    std.mem.doNotOptimizeAway(out[0..n]);
}

/// Feed arbitrary bytes to the HTTP/2 frame reader as a whole connection.
///
/// This is the entry point that matters most: it is exactly what a hostile peer
/// controls, and it drives frame parsing, HPACK, flow control and stream state
/// together.
pub fn fuzzConnection(alloc: std.mem.Allocator, input: []const u8) !void {
    const stream_bufs = try alloc.alloc([]u8, 4);
    defer alloc.free(stream_bufs);
    for (stream_bufs) |*b| b.* = try alloc.alloc(u8, 4096);
    defer for (stream_bufs) |b| alloc.free(b);

    var c = h2.Connection.init(stream_bufs);
    const ob = try alloc.alloc(u8, 64 * 1024);
    defer alloc.free(ob);
    var out = h2.OutBuf.init(ob);

    var pos: usize = 0;
    // Optionally consume a preface, so the fuzzer reaches frame parsing both
    // with and without one.
    if (input.len >= h2.client_preface.len and
        std.mem.eql(u8, input[0..h2.client_preface.len], h2.client_preface))
    {
        pos += (c.consumePreface(input) catch return) orelse return;
    } else {
        c.preface_seen = true;
    }

    // A bound on iterations, not on bytes: the property under test is that the
    // reader always makes progress or asks for more. A reader that returned
    // `consumed = 0` forever with bytes available would hang a real server, so
    // the loop detects it rather than spinning.
    var iterations: usize = 0;
    while (pos < input.len) {
        iterations += 1;
        if (iterations > input.len + 16) {
            std.debug.panic("frame reader made no progress on {d} bytes", .{input.len - pos});
        }
        const r = c.readFrame(input[pos..], &out) catch return;
        if (r.consumed == 0) break; // needs more bytes; correct termination
        pos += r.consumed;
        switch (r.event) {
            .request => |idx| {
                // Exercise the layers a real server would run next.
                const s = &c.streams[idx];
                const body = grpc.decodeMessage(s.bodyBytes()) catch {
                    c.closeStream(idx);
                    continue;
                };
                fuzzProtoMessages(body);
                c.closeStream(idx);
            },
            .goaway => return,
            else => {},
        }
        // The output buffer is finite; a real connection flushes here.
        if (out.len > ob.len / 2) out.reset();
    }
}

/// Feed arbitrary bytes to every top-level protobuf decoder.
///
/// §8.8 lists "wrong wire types, deeply nested messages, dimension mismatches"
/// All three land here.
pub fn fuzzProtoMessages(input: []const u8) void {
    const msg = proto.messages;

    {
        var r = proto.Reader.init(input);
        if (msg.UpsertPoints.decode(&r)) |up| {
            var it = up.pointIterator();
            var n: usize = 0;
            while (it.next() catch null) |pt| {
                n += 1;
                if (n > 4096) break; // a self-referential frame must not loop forever
                if (pt.vectors.single) |v| {
                    var dst: [64]f32 = undefined;
                    _ = v.dense.copyInto(&dst) catch {};
                }
            }
        } else |_| {}
    }
    {
        var r = proto.Reader.init(input);
        if (msg.QueryBatchPoints.decode(&r)) |qb| {
            _ = qb.countQueries() catch {};
            var it = qb.queryIterator();
            var n: usize = 0;
            while (it.next() catch null) |_| {
                n += 1;
                if (n > 4096) break;
            }
        } else |_| {}
    }
    {
        var r = proto.Reader.init(input);
        _ = msg.VectorParams.decode(&r) catch {};
    }
    {
        var r = proto.Reader.init(input);
        _ = msg.SearchParams.decode(&r) catch {};
    }
    {
        var r = proto.Reader.init(input);
        _ = msg.PointId.decode(&r) catch {};
    }
}

// -------------------------------------------------------------------------
// The checked-in regression corpus.
//
// §8.8: "Shrink failing cases to a minimal reproducer and check it into the
// regression corpus." Each entry below is named for the hazard it covers, so a
// future failure points at a property rather than at an opaque blob.
// -------------------------------------------------------------------------

const corpus = struct {
    /// §8.8: "oversized length prefixes".
    const oversized_varint = [_]u8{ 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
    /// A length-delimited field claiming far more bytes than the buffer holds.
    const lying_length = [_]u8{ 0x0a, 0xff, 0xff, 0xff, 0x7f, 0x01 };
    /// §8.8: "wrong wire types", field 1 as varint where a message is expected.
    const wrong_wire_type = [_]u8{ 0x08, 0x01 };
    /// §8.8: "truncated HPACK".
    const truncated_hpack = [_]u8{ 0x40, 0x0a, 'a', 'b' };
    /// A Huffman string whose declared length exceeds the block.
    const hpack_huffman_overrun = [_]u8{ 0x00, 0x8f, 0xff, 0xff };
    /// An HPACK index naming nothing.
    const hpack_bad_index = [_]u8{0xfe};
    /// A frame header declaring a length larger than max_frame_size.
    const huge_frame = [_]u8{ 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };
    /// SETTINGS with a length that is not a multiple of 6.
    const bad_settings = [_]u8{ 0x00, 0x00, 0x05, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 1, 2, 3, 4, 5 };
    /// A WINDOW_UPDATE of zero, which RFC 9113 forbids.
    const zero_window_update = [_]u8{ 0x00, 0x00, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0 };
    /// A DATA frame whose pad length exceeds its payload.
    const bad_padding = [_]u8{ 0x00, 0x00, 0x02, 0x00, 0x08, 0x00, 0x00, 0x00, 0x01, 0xff, 0x00 };
    /// §8.8: "deeply nested messages". 64 nested length-delimited fields, past
    /// `wire.max_depth`.
    const deep_nesting = blk: {
        @setEvalBranchQuota(10_000);
        var buf: [128]u8 = undefined;
        for (0..64) |i| {
            buf[i * 2] = 0x0a; // field 1, length-delimited
            buf[i * 2 + 1] = @intCast(126 - i * 2);
        }
        break :blk buf;
    };
    /// A gRPC prefix declaring more bytes than follow.
    const grpc_short_message = [_]u8{ 0x00, 0x00, 0x00, 0xff, 0xff, 0x01 };
    /// The compressed flag set, which §6.1 says never happens.
    const grpc_compressed = [_]u8{ 0x01, 0x00, 0x00, 0x00, 0x01, 0x00 };
    /// §8.8: "dimension mismatches", a packed float run of non-multiple-of-4
    /// length.
    const ragged_floats = [_]u8{ 0x0a, 0x03, 0x00, 0x00, 0x00 };
};

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const testing = std.testing;

test "corpus: every checked-in reproducer terminates without UB" {
    // Running in Debug (the default for `zig build test`) makes Zig's safety
    // checks live, which is what turns "no UB" into something a test can
    // actually observe.
    const cases: []const []const u8 = &.{
        &corpus.oversized_varint,
        &corpus.lying_length,
        &corpus.wrong_wire_type,
        &corpus.truncated_hpack,
        &corpus.hpack_huffman_overrun,
        &corpus.hpack_bad_index,
        &corpus.huge_frame,
        &corpus.bad_settings,
        &corpus.zero_window_update,
        &corpus.bad_padding,
        &corpus.deep_nesting,
        &corpus.grpc_short_message,
        &corpus.grpc_compressed,
        &corpus.ragged_floats,
    };
    for (cases) |c| {
        fuzzHpack(c);
        fuzzHuffman(c);
        fuzzProtoMessages(c);
        try fuzzConnection(testing.allocator, c);
    }
}

test "§8.8: a deeply nested message is refused, not recursed" {
    // Without `wire.max_depth` this is a stack overflow, which is a crash
    // rather than "a clean gRPC error".
    var r = proto.Reader.init(&corpus.deep_nesting);
    // Whatever it returns, it must return.
    _ = proto.messages.VectorParams.decode(&r) catch {};

    // And the depth limit is reachable directly.
    var deep = proto.Reader.init(&[_]u8{0x00});
    deep.depth = proto.wire.max_depth - 1;
    try testing.expectError(proto.wire.Error.TooDeep, deep.nested());
}

test "§8.8: oversized length prefixes are rejected rather than allocated" {
    var r = proto.Reader.init(&corpus.oversized_varint);
    try testing.expectError(proto.wire.Error.InvalidVarint, r.varint());

    var r2 = proto.Reader.init(&corpus.lying_length);
    _ = try r2.tag();
    try testing.expectError(proto.wire.Error.InvalidLength, r2.bytes());
}

test "§8.8: a compressed gRPC message is refused" {
    // §6.1: "Compression is not enabled." Accepting the flag and ignoring it
    // would hand gzip bytes to the protobuf decoder.
    try testing.expectError(
        grpc.Error.CompressionUnsupported,
        grpc.decodeMessage(&corpus.grpc_compressed),
    );
}

test "fuzz: random bytes never crash the HPACK decoder" {
    var prng = std.Random.DefaultPrng.init(0xf022);
    const rnd = prng.random();
    var buf: [512]u8 = undefined;
    for (0..2000) |_| {
        const n = rnd.uintLessThan(usize, buf.len);
        rnd.bytes(buf[0..n]);
        fuzzHpack(buf[0..n]);
        fuzzHuffman(buf[0..n]);
    }
}

test "fuzz: random bytes never crash the protobuf decoders" {
    var prng = std.Random.DefaultPrng.init(0x9b0);
    const rnd = prng.random();
    var buf: [1024]u8 = undefined;
    for (0..2000) |_| {
        const n = rnd.uintLessThan(usize, buf.len);
        rnd.bytes(buf[0..n]);
        fuzzProtoMessages(buf[0..n]);
    }
}

test "fuzz: random bytes never crash or hang the HTTP/2 connection" {
    var prng = std.Random.DefaultPrng.init(0x1122);
    const rnd = prng.random();
    var buf: [2048]u8 = undefined;
    for (0..500) |_| {
        const n = rnd.uintLessThan(usize, buf.len);
        rnd.bytes(buf[0..n]);
        try fuzzConnection(testing.allocator, buf[0..n]);
    }
}

test "fuzz: structurally valid frames with random payloads" {
    // Pure random bytes rarely produce a parseable frame header, so most of the
    // budget above is spent on the first `readFrame` rejecting garbage. This
    // generates well-formed *headers* with random payloads, which is what
    // actually reaches the per-frame handlers, HPACK and the stream machinery.
    var prng = std.Random.DefaultPrng.init(0x3344);
    const rnd = prng.random();

    var buf: [4096]u8 = undefined;
    for (0..1000) |_| {
        var pos: usize = 0;
        const frames = 1 + rnd.uintLessThan(usize, 6);
        for (0..frames) |_| {
            if (pos + h2.frame_header_len + 64 > buf.len) break;
            const payload_len = rnd.uintLessThan(u32, 48);
            const h = h2.FrameHeader{
                .length = payload_len,
                .frame_type = @enumFromInt(rnd.uintLessThan(u8, 12)),
                .flags = rnd.int(u8),
                .stream_id = rnd.uintLessThan(u31, 8),
            };
            h.write(buf[pos..][0..h2.frame_header_len]);
            pos += h2.frame_header_len;
            rnd.bytes(buf[pos..][0..payload_len]);
            pos += payload_len;
        }
        try fuzzConnection(testing.allocator, buf[0..pos]);
    }
}

test "fuzz: mutations of a valid request stay safe" {
    // §8.8's differential-fuzzing spirit applied locally: start from a request
    // the server accepts and flip bytes. Mutation from a valid seed reaches far
    // deeper into the decoders than random bytes do.
    var seed_buf: [512]u8 = undefined;
    var w = proto.Writer.init(&seed_buf);
    try w.writeStringField(1, "bench");
    const qp = try w.beginNested(2, 3);
    try w.writeStringField(1, "bench");
    {
        const query = try w.beginNested(3, 3);
        const nearest = try w.beginNested(1, 3);
        const dv = try w.beginNested(2, 3);
        try w.writePackedFloats(1, &[_]f32{ 1, 2, 3, 4 });
        try w.endNested(dv);
        try w.endNested(nearest);
        try w.endNested(query);
    }
    try w.writeVarintFieldAlways(8, 10);
    try w.endNested(qp);
    const seed = w.written();

    var prng = std.Random.DefaultPrng.init(0x5566);
    const rnd = prng.random();
    var mutated: [512]u8 = undefined;

    for (0..5000) |_| {
        @memcpy(mutated[0..seed.len], seed);
        const flips = 1 + rnd.uintLessThan(usize, 4);
        for (0..flips) |_| {
            const i = rnd.uintLessThan(usize, seed.len);
            mutated[i] = rnd.int(u8);
        }
        fuzzProtoMessages(mutated[0..seed.len]);
    }
}

// -------------------------------------------------------------------------
// Randomised stream interleaving against a model of the server
// -------------------------------------------------------------------------

/// A random but *legal* client: several streams in flight, HEADERS split into
/// CONTINUATION, DATA in pieces, RST_STREAM before or after END_STREAM,
/// SETTINGS changes, WINDOW_UPDATEs and PINGs interleaved, and the resulting
/// bytes fed to the connection in random-sized reads. Around it, a model of
/// what `server.zig` does with the events, so the invariants a real server
/// depends on can be asserted directly:
///
///   * every stream that ends cleanly and is never reset gets exactly one
///     response; a reset one gets at most one; a refused one gets none;
///   * a slot handed out by `.request` is never handed out again while the
///     model still owns it, and keeps its stream id and a terminal state
///     until the model frees it, whatever the peer sends in the meantime.
///
/// This is the shape of bug the RST_STREAM slot reuse was: nothing in a
/// single-frame unit test, only visible when a reset lands while a worker
/// holds the slot and the next HEADERS happens to pick it.
fn interleavedStreamsRound(alloc: std.mem.Allocator, seed: u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();

    const pool = 4;
    const stream_bufs = try alloc.alloc([]u8, pool);
    defer alloc.free(stream_bufs);
    for (stream_bufs) |*b| b.* = try alloc.alloc(u8, 512);
    defer for (stream_bufs) |b| alloc.free(b);
    var c = h2.Connection.init(stream_bufs);
    c.preface_seen = true;

    const ob = try alloc.alloc(u8, 256 * 1024);
    defer alloc.free(ob);
    var out = h2.OutBuf.init(ob);

    // The client's view of each stream it opened. The id is assigned when
    // its HEADERS goes on the wire, not when the model creates it: RFC 9113
    // §5.1.1 requires ids to increase in the order streams are opened, and
    // the server now enforces it.
    const ClientStream = struct {
        id: u31 = 0,
        /// Frames left to send: header pieces, then data pieces.
        header_pieces: u8,
        data_pieces: u8,
        /// The last data piece carries END_STREAM.
        ended: bool = false,
        /// RST_STREAM sent. Before END_STREAM no response is owed; after
        /// it, the reset may or may not beat the completion, so at most one.
        reset: bool = false,
        /// Bytes of DATA sent, so the model knows which overflow.
        sent: usize = 0,
    };
    var live: [8]ClientStream = undefined;
    var live_n: usize = 0;
    var next_id: u31 = 1;

    // What the server model saw and did.
    const max_ids = 4096;
    const responses = try alloc.alloc(u8, max_ids);
    defer alloc.free(responses);
    @memset(responses, 0);
    const refused = try alloc.alloc(bool, max_ids);
    defer alloc.free(refused);
    @memset(refused, false);
    // What the client believes each stream is owed once it is done with it.
    const owed = try alloc.alloc(Owed, max_ids);
    defer alloc.free(owed);
    @memset(owed, .unknown);
    var owned = [_]?u31{null} ** pool;

    // The wire, built frame by frame, then fed in random reads.
    const wire = try alloc.alloc(u8, 1 << 20);
    defer alloc.free(wire);
    var wire_len: usize = 0;
    var wo = h2.OutBuf.init(wire);

    var hb: [64]u8 = undefined;
    const path = "/qdrant.Points/QueryBatch";
    var hn: usize = 0;
    hn += hpack.encodeInteger(hb[hn..], 6, 0x40, 4);
    hn += hpack.encodeInteger(hb[hn..], 7, 0x00, path.len);
    @memcpy(hb[hn..][0..path.len], path);
    hn += path.len;

    const frames = 60 + rnd.uintLessThan(usize, 200);
    for (0..frames) |_| {
        // Occasionally open a new stream.
        if (live_n < live.len and (live_n == 0 or rnd.uintLessThan(u8, 3) == 0)) {
            live[live_n] = .{
                .header_pieces = 1 + rnd.uintLessThan(u8, 3),
                .data_pieces = rnd.uintLessThan(u8, 4),
            };
            live_n += 1;
        }
        // Pick a stream and advance it, or send a control frame.
        switch (rnd.uintLessThan(u8, 10)) {
            0 => {
                // SETTINGS: a new initial window size, or a bigger frame size.
                var p: [6]u8 = undefined;
                if (rnd.boolean()) {
                    std.mem.writeInt(u16, p[0..2], @intFromEnum(h2.SettingId.initial_window_size), .big);
                    std.mem.writeInt(u32, p[2..6], rnd.uintLessThan(u32, 1 << 20), .big);
                } else {
                    std.mem.writeInt(u16, p[0..2], @intFromEnum(h2.SettingId.max_frame_size), .big);
                    std.mem.writeInt(u32, p[2..6], 16384 + rnd.uintLessThan(u32, 1 << 20), .big);
                }
                try wo.frame(.settings, 0, 0, &p);
            },
            1 => {
                // WINDOW_UPDATE on the connection or a stream, live or
                // already retired. Never an id that has not been opened yet:
                // RFC 9113 §5.1 makes a credit for an idle stream a
                // PROTOCOL_ERROR, which ends the connection instead of
                // exercising the interleaving this round is about. Same
                // reason ids are assigned in wire order above.
                var p: [4]u8 = undefined;
                std.mem.writeInt(u32, &p, 1 + rnd.uintLessThan(u32, 65535), .big);
                const opened = (next_id - 1) / 2;
                const sid: u31 = if (opened == 0 or rnd.boolean()) 0 else 1 + 2 * rnd.uintLessThan(u31, opened);
                try wo.frame(.window_update, 0, sid, &p);
            },
            2 => {
                var p: [8]u8 = undefined;
                rnd.bytes(&p);
                try wo.frame(.ping, 0, 0, &p);
            },
            else => {
                if (live_n == 0) continue;
                const li = rnd.uintLessThan(usize, live_n);
                const st = &live[li];
                if (st.header_pieces > 0) {
                    // HEADERS then CONTINUATIONs, contiguous on the wire.
                    st.id = next_id;
                    next_id += 2;
                    const pieces = st.header_pieces;
                    var at: usize = 0;
                    for (0..pieces) |k| {
                        const last = k + 1 == pieces;
                        const take = if (last) hn - at else (hn - at) / (pieces - k);
                        var flags: u8 = 0;
                        if (last) flags |= h2.Flags.end_headers;
                        // A bodyless request carries END_STREAM on the HEADERS.
                        if (k == 0 and st.data_pieces == 0) {
                            flags |= h2.Flags.end_stream;
                            st.ended = true;
                        }
                        try wo.frame(if (k == 0) .headers else .continuation, flags, st.id, hb[at..][0..take]);
                        at += take;
                    }
                    st.header_pieces = 0;
                } else if (!st.reset and rnd.uintLessThan(u8, 6) == 0) {
                    var p: [4]u8 = undefined;
                    std.mem.writeInt(u32, &p, @intFromEnum(h2.ErrorCode.cancel), .big);
                    try wo.frame(.rst_stream, 0, st.id, &p);
                    st.reset = true;
                } else if (st.data_pieces > 0 and !st.reset) {
                    // Sometimes more than the 512-byte request buffer, to
                    // drive the fail-fast path.
                    const n = rnd.uintLessThan(usize, 300);
                    var p: [300]u8 = undefined;
                    rnd.bytes(p[0..n]);
                    st.data_pieces -= 1;
                    const flags: u8 = if (st.data_pieces == 0) h2.Flags.end_stream else 0;
                    if (st.data_pieces == 0) st.ended = true;
                    st.sent += n;
                    try wo.frame(.data, flags, st.id, p[0..n]);
                }
                // Retire finished client streams so new ones open.
                if (st.header_pieces == 0 and (st.ended or st.reset)) {
                    if (st.reset or rnd.boolean()) {
                        // Remember what is owed before forgetting it.
                        owed[st.id] = if (st.ended and !st.reset) .one else .at_most_one;
                        live[li] = live[live_n - 1];
                        live_n -= 1;
                    }
                }
            },
        }
    }
    // Retire the rest. One whose HEADERS never went out has no id and is
    // owed nothing.
    for (live[0..live_n]) |st| {
        if (st.header_pieces > 0) continue;
        owed[st.id] = if (st.ended and !st.reset) .one else .at_most_one;
    }
    wire_len = wo.len;

    // Feed the wire in random-sized reads through a compacting input buffer,
    // exactly as the server does, running the model on every event.
    var in: [8192]u8 = undefined;
    var in_len: usize = 0;
    var fed: usize = 0;
    var steps: usize = 0;
    while (fed < wire_len or in_len > 0) {
        steps += 1;
        if (steps > wire_len * 4 + 64) return error.NoProgress;
        // Read.
        if (fed < wire_len and in_len < in.len) {
            const want = @min(1 + rnd.uintLessThan(usize, 700), @min(in.len - in_len, wire_len - fed));
            @memcpy(in[in_len..][0..want], wire[fed..][0..want]);
            in_len += want;
            fed += want;
        }
        // Parse.
        var pos: usize = 0;
        while (pos < in_len) {
            const r = try c.readFrame(in[pos..in_len], &out);
            if (r.consumed == 0) break;
            pos += r.consumed;
            switch (r.event) {
                .request => |idx| {
                    if (owned[idx] != null) return error.SlotReusedWhileOwned;
                    owned[idx] = c.streams[idx].id;
                },
                .request_too_large => |idx| {
                    // The I/O thread answers and frees the slot itself.
                    const id = c.streams[idx].id;
                    try grpc.writeError(&out, id, .resource_exhausted, "too large");
                    responses[id] += 1;
                    c.closeStream(idx);
                },
                .goaway => return error.UnexpectedGoaway,
                .reset, .need_more, .progress => {},
            }
            // Owned slots keep their id and a terminal state.
            for (owned, 0..) |o, i| {
                const id = o orelse continue;
                if (c.streams[i].id != id) return error.OwnedSlotRenamed;
                if (c.streams[i].state != .half_closed_remote and c.streams[i].state != .closed) return error.OwnedSlotReopened;
                if (!c.streams[i].dispatched) return error.OwnedSlotUndispatched;
            }
            // Refusals are visible on the wire.
            try noteRefusals(&out, refused);
            if (out.len > ob.len / 2) out.reset();
        }
        if (pos > 0) {
            std.mem.copyForwards(u8, in[0 .. in_len - pos], in[pos..in_len]);
            in_len -= pos;
        }
        if (fed >= wire_len and pos == 0 and in_len > 0) return error.TrailingGarbage;

        // The worker pool completes some owned requests, in random order.
        if (rnd.uintLessThan(u8, 3) == 0) try completeOne(&c, &out, &owned, responses, rnd);
    }
    // Drain every remaining completion.
    while (true) {
        var any = false;
        for (owned) |o| any = any or o != null;
        if (!any) break;
        try completeOne(&c, &out, &owned, responses, rnd);
    }
    try noteRefusals(&out, refused);

    // Now the accounting: what each stream was owed, against what it got.
    for (owed, 0..) |o, id| {
        const got = responses[id];
        if (refused[id]) {
            if (got != 0) return error.RefusedStreamAnswered;
            continue;
        }
        switch (o) {
            .unknown => if (got != 0) return error.UnknownStreamAnswered,
            .one => if (got != 1) return error.WrongResponseCount,
            .at_most_one => if (got > 1) return error.AnsweredTwice,
        }
    }
}

const Owed = enum { unknown, one, at_most_one };

/// Pick a random owned slot and do what `drainCompletions` does with it.
fn completeOne(c: *h2.Connection, out: *h2.OutBuf, owned: []?u31, responses: []u8, rnd: std.Random) !void {
    var candidates: [h2.max_streams]usize = undefined;
    var n: usize = 0;
    for (owned, 0..) |o, i| {
        if (o != null) {
            candidates[n] = i;
            n += 1;
        }
    }
    if (n == 0) return;
    const idx = candidates[rnd.uintLessThan(usize, n)];
    const id = owned[idx].?;
    if (c.streams[idx].state == .closed) {
        // Reset by the peer while the worker ran: dropped.
    } else {
        try grpc.writeError(out, id, .ok, "");
        responses[id] += 1;
    }
    c.closeStream(idx);
    owned[idx] = null;
    if (out.len > out.buf.len / 2) out.reset();
}

fn noteRefusals(out: *const h2.OutBuf, refused: []bool) !void {
    var pos: usize = 0;
    while (pos + h2.frame_header_len <= out.len) {
        const h = h2.FrameHeader.parse(out.buf[pos..][0..h2.frame_header_len]);
        if (h.frame_type == .rst_stream and h.length == 4) {
            const code = std.mem.readInt(u32, out.buf[pos + h2.frame_header_len ..][0..4], .big);
            if (code == @intFromEnum(h2.ErrorCode.refused_stream)) refused[h.stream_id] = true;
        }
        pos += h2.frame_header_len + h.length;
    }
}

test "fuzz: interleaved streams, resets and random reads keep one response per request" {
    for (0..300) |i| {
        interleavedStreamsRound(testing.allocator, 0x7700 + i) catch |e| {
            std.debug.print("seed {d}: {s}\n", .{ 0x7700 + i, @errorName(e) });
            return e;
        };
    }
}
