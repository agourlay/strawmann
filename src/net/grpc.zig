//! gRPC over HTTP/2, message framing, status codes, and trailers.
//!
//! §6.1: "**gRPC framing** is a 5-byte prefix (compressed-flag + big-endian u32
//! length) per message, and the status lives in **trailers** (`grpc-status`,
//! `grpc-message`) in a second HEADERS frame with END_STREAM. Tonic sends
//! `te: trailers` and expects trailers even on success."
//!
//! That last clause is the one that costs a day if missed. A gRPC response is
//! *three* HTTP/2 frames, not two:
//!
//!   1. HEADERS , `:status: 200`, `content-type: application/grpc`
//!   2. DATA    , 5-byte prefix + the protobuf message
//!   3. HEADERS , `grpc-status: 0`, with END_STREAM   <-- the trailers
//!
//! Omitting (3) leaves the client waiting forever on a request that visibly
//! returned data, which presents as a timeout rather than as a protocol error
//!, §2's "failures are legible rather than showing up as timeouts" is exactly
//! this hazard.
//!
//! An error response omits (2) entirely and carries the status in (1) as
//! "trailers-only", which is what tonic expects for a failed unary call.

const std = @import("std");
const h2 = @import("h2.zig");
const hpack = @import("hpack.zig");

/// The 5-byte gRPC message prefix.
pub const prefix_len = 5;

/// gRPC status codes (grpc/status.h).
pub const Status = enum(u32) {
    ok = 0,
    cancelled = 1,
    unknown = 2,
    invalid_argument = 3,
    deadline_exceeded = 4,
    not_found = 5,
    already_exists = 6,
    permission_denied = 7,
    resource_exhausted = 8,
    failed_precondition = 9,
    aborted = 10,
    out_of_range = 11,
    /// §1: "Anything in this list that `bfb` can be told to emit is handled by
    /// returning a clean `UNIMPLEMENTED`, never by silently degrading."
    unimplemented = 12,
    internal = 13,
    unavailable = 14,
    data_loss = 15,
    unauthenticated = 16,

    /// Decimal text, for the `grpc-status` header value. Small enough to be a
    /// table rather than a formatted integer, which keeps the response path
    /// allocation-free.
    pub fn text(self: Status) []const u8 {
        return switch (self) {
            .ok => "0",
            .cancelled => "1",
            .unknown => "2",
            .invalid_argument => "3",
            .deadline_exceeded => "4",
            .not_found => "5",
            .already_exists => "6",
            .permission_denied => "7",
            .resource_exhausted => "8",
            .failed_precondition => "9",
            .aborted => "10",
            .out_of_range => "11",
            .unimplemented => "12",
            .internal => "13",
            .unavailable => "14",
            .data_loss => "15",
            .unauthenticated => "16",
        };
    }
};

pub const Error = error{
    /// The 5-byte prefix declares a length the frame does not contain.
    IncompleteMessage,
    /// The compressed flag is set. §2/§6.1: "Compression is not enabled."
    /// Accepting the flag and ignoring it would hand the protobuf decoder
    /// gzip bytes, which fails far from the cause.
    CompressionUnsupported,
    OutputFull,
    /// The body does not fit the peer's current send window.
    WindowExhausted,
};

/// Strip the 5-byte prefix from a request body.
///
/// bfb issues only unary RPCs, so a request body is exactly one length-prefixed
/// message. A body carrying more than one is a client bug rather than a stream
/// we need to handle.
pub fn decodeMessage(body: []const u8) Error![]const u8 {
    if (body.len < prefix_len) return Error.IncompleteMessage;
    if (body[0] != 0) return Error.CompressionUnsupported;
    const len = std.mem.readInt(u32, body[1..5], .big);
    // Subtract rather than add. `prefix_len + len` is computed in u32 because
    // `len` is, so a wire-supplied length of 0xFFFFFFFB..0xFFFFFFFF overflows
    // *before* the bounds check, a panic in safe builds and, in ReleaseFast, a
    // wrap that passes the check and yields a ~4 GiB out-of-bounds slice from a
    // single 5-byte DATA frame.
    if (len > body.len - prefix_len) return Error.IncompleteMessage;
    return body[prefix_len..][0..len];
}

/// Write the 5-byte prefix for a message of `len` bytes.
pub fn writePrefix(out: *[prefix_len]u8, len: u32) void {
    out[0] = 0; // not compressed
    std.mem.writeInt(u32, out[1..5], len, .big);
}

/// The RPC path, split into service and method.
pub const Path = struct {
    service: []const u8,
    method: []const u8,

    /// Parse `/qdrant.Points/QueryBatch`.
    pub fn parse(path: []const u8) ?Path {
        if (path.len == 0 or path[0] != '/') return null;
        const rest = path[1..];
        const slash = std.mem.indexOfScalar(u8, rest, '/') orelse return null;
        const service = rest[0..slash];
        const method = rest[slash + 1 ..];
        if (service.len == 0 or method.len == 0) return null;
        return .{ .service = service, .method = method };
    }

    pub fn eql(self: Path, service: []const u8, method: []const u8) bool {
        return std.mem.eql(u8, self.service, service) and std.mem.eql(u8, self.method, method);
    }
};

/// Test scaffolding: a successful unary response in one go, HEADERS, DATA,
/// trailers.
///
/// `body` is the encoded protobuf message *without* the gRPC prefix; this
/// function writes the prefix. It requires the whole body to fit the send
/// windows and the output buffer right now, and fails with `WindowExhausted`
/// or `OutputFull` otherwise, having written nothing.
///
/// Not `pub`, and not a path the server has: the server writes the pieces
/// separately through `writeResponseHeaders`, `writeData` and `writeTrailers`
/// so a body that outruns the window or the buffer is parked and resumed
/// rather than refused. It stayed exported long enough to read as the
/// ordinary way to answer a call, which is the one thing it is not; the tests
/// below keep it because composing the three pieces by hand in each of them
/// tests the same three calls with more room to get them wrong.
fn writeResponse(conn: *h2.Connection, out: *h2.OutBuf, idx: usize, body: []const u8) !void {
    const stream_id = conn.streams[idx].id;
    const total = prefix_len + body.len;
    const frames = (total + conn.peer.max_frame_size - 1) / conn.peer.max_frame_size;
    if (out.available() < total + h2.OutBuf.reserve + (frames + 2) * h2.frame_header_len + 64) return Error.OutputFull;
    const win: i64 = @min(conn.send_window, conn.streams[idx].window);
    if (win < total) return Error.WindowExhausted;

    try writeResponseHeaders(out, stream_id);
    var off: usize = 0;
    while (off < total) {
        const next = writeData(conn, out, idx, body, off);
        // Ruled out by the checks above; guards the loop rather than the
        // arithmetic.
        if (next == off) return Error.OutputFull;
        off = next;
    }
    try writeTrailers(out, stream_id, .ok, "");
}

/// Write as much of a response body as may go out right now, in DATA frames
/// no larger than the peer's SETTINGS_MAX_FRAME_SIZE and within both send
/// windows (RFC 9113 §6.9), charging the windows for what was written.
///
/// `off` counts over the 5-byte gRPC prefix followed by `body`, so a caller
/// resumes by passing back the returned offset; `prefix_len + body.len` means
/// the body is complete and the trailers may follow. A return equal to `off`
/// means nothing could be written: no window, or the output buffer is down to
/// its `reserve`, and the caller must park the stream until a WINDOW_UPDATE
/// arrives or the buffer is flushed.
///
/// The prefix is regenerated from `body.len` rather than stored: the body
/// lives in a per-stream arena a handler filled from its start, so there is
/// no room in front of it, and threading a five-byte copy through the parked
/// state is more machinery than recomputing it.
pub fn writeData(conn: *h2.Connection, out: *h2.OutBuf, idx: usize, body: []const u8, off: usize) usize {
    const total = prefix_len + body.len;
    var pos = off;
    while (pos < total) {
        const n = @min(total - pos, conn.dataBudget(out, idx));
        if (n == 0) break;
        // `beginFrame` cannot fail: `dataBudget` already kept a header's worth
        // of room, and the assert makes a regression loud rather than a
        // silently dropped frame.
        const cursor = out.beginFrame(.data, 0, conn.streams[idx].id) catch unreachable;
        const dst = out.body()[0..n];
        var w: usize = 0;
        if (pos < prefix_len) {
            var pfx: [prefix_len]u8 = undefined;
            writePrefix(&pfx, @intCast(body.len));
            w = @min(prefix_len - pos, n);
            @memcpy(dst[0..w], pfx[pos..][0..w]);
        }
        if (w < n) @memcpy(dst[w..n], body[pos + w - prefix_len ..][0 .. n - w]);
        out.endFrame(cursor, n) catch unreachable;
        conn.chargeSend(idx, n);
        pos += n;
    }
    return pos;
}

/// Write just the response HEADERS frame, for callers that stream the body
/// into the output buffer themselves.
pub fn writeResponseHeaders(out: *h2.OutBuf, stream_id: u31) !void {
    var hb: [128]u8 = undefined;
    var n: usize = 0;
    n += try hpack.Encoder.writeHeader(hb[n..], ":status", "200");
    n += try hpack.Encoder.writeHeader(hb[n..], "content-type", "application/grpc");
    try out.frame(.headers, h2.Flags.end_headers, stream_id, hb[0..n]);
}

/// Write the trailing HEADERS frame carrying the status.
///
/// END_STREAM lives here, not on the DATA frame, that is what makes it a
/// trailer rather than a second header block.
pub fn writeTrailers(out: *h2.OutBuf, stream_id: u31, status: Status, message: []const u8) !void {
    var hb: [512]u8 = undefined;
    var n: usize = 0;
    n += try hpack.Encoder.writeHeader(hb[n..], "grpc-status", status.text());
    if (message.len > 0) {
        // grpc-message is percent-encoded per the gRPC spec. Our messages are
        // ASCII identifiers and short English phrases, so the encoder below
        // only has to handle the characters that actually appear; anything
        // outside the safe set is escaped rather than passed through.
        var enc: [256]u8 = undefined;
        const encoded = percentEncode(&enc, message);
        n += try hpack.Encoder.writeHeader(hb[n..], "grpc-message", encoded);
    }
    try out.frame(.headers, h2.Flags.end_headers | h2.Flags.end_stream, stream_id, hb[0..n]);
}

/// Write a trailers-only error response.
///
/// A failed unary call carries `:status: 200` with the gRPC status in the same
/// header block, HTTP is the transport and succeeded, so an HTTP error status
/// would be wrong and tonic would surface it as a transport failure instead of
/// the gRPC error we meant.
pub fn writeError(out: *h2.OutBuf, stream_id: u31, status: Status, message: []const u8) !void {
    var hb: [512]u8 = undefined;
    var n: usize = 0;
    n += try hpack.Encoder.writeHeader(hb[n..], ":status", "200");
    n += try hpack.Encoder.writeHeader(hb[n..], "content-type", "application/grpc");
    n += try hpack.Encoder.writeHeader(hb[n..], "grpc-status", status.text());
    if (message.len > 0) {
        var enc: [256]u8 = undefined;
        const encoded = percentEncode(&enc, message);
        n += try hpack.Encoder.writeHeader(hb[n..], "grpc-message", encoded);
    }
    try out.frame(.headers, h2.Flags.end_headers | h2.Flags.end_stream, stream_id, hb[0..n]);
}

/// Reverse `percentEncode`.
///
/// Any conformant gRPC client does this, and ours has to as well or every
/// message carrying a non-ASCII byte reads back mangled. The spec's messages
/// cite sections as `§n`, and `§` is U+00A7, two bytes, both outside the
/// unreserved range, so *every* `UNIMPLEMENTED` message strawmann sends is
/// percent-encoded on the wire. That is correct, and it means a test asserting
/// on the raw header value is asserting on `%C2%A7` rather than on what a real
/// client sees.
///
/// Returns the decoded slice within `out`. Malformed escapes are passed through
/// verbatim rather than rejected: this is diagnostic text, and losing the whole
/// message because one byte was mis-encoded is the wrong trade.
pub fn percentDecode(out: []u8, s: []const u8) []const u8 {
    var n: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] == '%' and i + 2 < s.len) {
            const hi = std.fmt.charToDigit(s[i + 1], 16) catch {
                out[n] = s[i];
                n += 1;
                i += 1;
                continue;
            };
            const lo = std.fmt.charToDigit(s[i + 2], 16) catch {
                out[n] = s[i];
                n += 1;
                i += 1;
                continue;
            };
            out[n] = hi * 16 + lo;
            n += 1;
            i += 3;
        } else {
            out[n] = s[i];
            n += 1;
            i += 1;
        }
    }
    return out[0..n];
}

/// Percent-encode per the gRPC `Status-Message` rule: bytes outside
/// `0x20..0x7E` minus `%` are escaped as `%XX`.
///
/// The output is truncated to `out.len` (256 bytes from `writeError`, which
/// bounds the trailers-only frame). Truncation is deliberate and always falls
/// on an escape boundary: an escape is written only when all three bytes fit,
/// so the client never sees a dangling `%C` and `percentDecode` never has to
/// guess. A message with many non-ASCII bytes therefore loses its tail, which
/// is the right trade for diagnostic text carried in a header.
fn percentEncode(out: []u8, s: []const u8) []const u8 {
    const hex = "0123456789ABCDEF";
    var n: usize = 0;
    for (s) |c| {
        const safe = c >= 0x20 and c <= 0x7e and c != '%';
        if (safe) {
            if (n + 1 > out.len) break;
            out[n] = c;
            n += 1;
        } else {
            // Whole escape or nothing.
            if (n + 3 > out.len) break;
            out[n] = '%';
            out[n + 1] = hex[c >> 4];
            out[n + 2] = hex[c & 0xf];
            n += 3;
        }
    }
    return out[0..n];
}

// =========================================================================
// Tests
// =========================================================================

const testing = std.testing;

test "message prefix round-trips" {
    var buf: [prefix_len + 16]u8 = undefined;
    const msg = "protobuf-bytes!!";
    writePrefix(buf[0..prefix_len], msg.len);
    @memcpy(buf[prefix_len..], msg);

    const got = try decodeMessage(&buf);
    try testing.expectEqualStrings(msg, got);
}

test "an empty message is valid" {
    // HealthCheckRequest is an empty message, so this is the very first thing
    // any client sends (§2: HealthCheck is on the critical path of every
    // connection).
    var buf: [prefix_len]u8 = undefined;
    writePrefix(&buf, 0);
    const got = try decodeMessage(&buf);
    try testing.expectEqual(@as(usize, 0), got.len);
}

test "a compressed message is refused rather than mis-parsed" {
    var buf: [prefix_len + 4]u8 = undefined;
    writePrefix(buf[0..prefix_len], 4);
    buf[0] = 1; // compressed flag
    try testing.expectError(Error.CompressionUnsupported, decodeMessage(&buf));
}

test "a truncated message is refused" {
    var buf: [prefix_len + 2]u8 = undefined;
    writePrefix(buf[0..prefix_len], 100);
    try testing.expectError(Error.IncompleteMessage, decodeMessage(&buf));
    try testing.expectError(Error.IncompleteMessage, decodeMessage(buf[0..3]));
}

test "path parsing handles the paths bfb issues" {
    const cases = [_]struct { []const u8, []const u8, []const u8 }{
        .{ "/qdrant.Qdrant/HealthCheck", "qdrant.Qdrant", "HealthCheck" },
        .{ "/qdrant.Collections/Create", "qdrant.Collections", "Create" },
        .{ "/qdrant.Points/Upsert", "qdrant.Points", "Upsert" },
        .{ "/qdrant.Points/QueryBatch", "qdrant.Points", "QueryBatch" },
    };
    for (cases) |c| {
        const p = Path.parse(c[0]).?;
        try testing.expectEqualStrings(c[1], p.service);
        try testing.expectEqualStrings(c[2], p.method);
        try testing.expect(p.eql(c[1], c[2]));
    }

    try testing.expectEqual(@as(?Path, null), Path.parse(""));
    try testing.expectEqual(@as(?Path, null), Path.parse("no-leading-slash"));
    try testing.expectEqual(@as(?Path, null), Path.parse("/only-one-segment"));
    try testing.expectEqual(@as(?Path, null), Path.parse("//empty-service"));
    try testing.expectEqual(@as(?Path, null), Path.parse("/service/"));
}

fn testBuffers(alloc: std.mem.Allocator, n: usize, size: usize) ![]([]u8) {
    const bufs = try alloc.alloc([]u8, n);
    for (bufs) |*b| b.* = try alloc.alloc(u8, size);
    return bufs;
}

fn freeBuffers(alloc: std.mem.Allocator, bufs: []([]u8)) void {
    for (bufs) |b| alloc.free(b);
    alloc.free(bufs);
}

fn testConn(bufs: []([]u8)) h2.Connection {
    var c = h2.Connection.init(bufs);
    // A generous peer, like hyper's: 2 MiB stream windows, 5 MiB connection.
    c.peer.initial_window_size = 2 * 1024 * 1024;
    c.send_window = 5 * 1024 * 1024;
    return c;
}

test "a success response is three frames ending in trailers with END_STREAM" {
    var ob: [4096]u8 = undefined;
    var out = h2.OutBuf.init(&ob);
    var c = testConn(&.{});
    // The pool has no request buffers, so open slot 0 by hand.
    c.streams[0].id = 1;
    c.streams[0].state = .half_closed_remote;
    const body = "\x08\x01\x10\x02";
    try writeResponse(&c, &out, 0, body);

    var pos: usize = 0;
    var frames: usize = 0;
    var saw_data = false;
    var last_had_end_stream = false;

    while (pos + h2.frame_header_len <= out.len) {
        const h = h2.FrameHeader.parse(out.buf[pos..][0..h2.frame_header_len]);
        const payload = out.buf[pos + h2.frame_header_len ..][0..h.length];
        frames += 1;
        last_had_end_stream = h.hasFlag(h2.Flags.end_stream);

        if (h.frame_type == .data) {
            saw_data = true;
            // Prefix then message.
            try testing.expectEqual(@as(u8, 0), payload[0]);
            try testing.expectEqual(@as(u32, body.len), std.mem.readInt(u32, payload[1..5], .big));
            try testing.expectEqualStrings(body, payload[5..]);
        }
        pos += h2.frame_header_len + h.length;
    }

    try testing.expectEqual(@as(usize, 3), frames);
    try testing.expect(saw_data);
    // Tonic waits forever without this.
    try testing.expect(last_had_end_stream);
}

test "trailers carry grpc-status and decode back through HPACK" {
    var ob: [4096]u8 = undefined;
    var out = h2.OutBuf.init(&ob);
    try writeTrailers(&out, 3, .ok, "");

    const h = h2.FrameHeader.parse(out.buf[0..h2.frame_header_len]);
    try testing.expectEqual(h2.FrameType.headers, h.frame_type);
    try testing.expect(h.hasFlag(h2.Flags.end_stream));
    try testing.expect(h.hasFlag(h2.Flags.end_headers));

    var dec = hpack.Decoder.init();
    const hdrs = try dec.decode(out.buf[h2.frame_header_len..][0..h.length]);
    try testing.expectEqual(@as(usize, 1), hdrs.len);
    try testing.expectEqualStrings("grpc-status", hdrs[0].name);
    try testing.expectEqualStrings("0", hdrs[0].value);
}

test "an error response is trailers-only with HTTP 200" {
    var ob: [4096]u8 = undefined;
    var out = h2.OutBuf.init(&ob);
    try writeError(&out, 5, .unimplemented, "sparse vectors");

    // Exactly one frame.
    const h = h2.FrameHeader.parse(out.buf[0..h2.frame_header_len]);
    try testing.expectEqual(h2.FrameType.headers, h.frame_type);
    try testing.expect(h.hasFlag(h2.Flags.end_stream));
    try testing.expectEqual(out.len, h2.frame_header_len + h.length);

    var dec = hpack.Decoder.init();
    const hdrs = try dec.decode(out.buf[h2.frame_header_len..][0..h.length]);

    var status_text: []const u8 = "";
    var msg: []const u8 = "";
    var http_status: []const u8 = "";
    for (hdrs) |x| {
        if (std.mem.eql(u8, x.name, "grpc-status")) status_text = x.value;
        if (std.mem.eql(u8, x.name, "grpc-message")) msg = x.value;
        if (std.mem.eql(u8, x.name, ":status")) http_status = x.value;
    }
    // HTTP succeeded; only gRPC failed.
    try testing.expectEqualStrings("200", http_status);
    try testing.expectEqualStrings("12", status_text);
    try testing.expectEqualStrings("sparse vectors", msg);
}

test "grpc-message percent-encodes unsafe bytes" {
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("hello", percentEncode(&buf, "hello"));
    // The gRPC spec's Percent-Byte-Unencoded is %x20-%x24 / %x26-%x7E, so a
    // space passes through and only '%' is carved out of the printable range.
    try testing.expectEqualStrings("a b", percentEncode(&buf, "a b"));
    try testing.expectEqualStrings("a%1Fb", percentEncode(&buf, "a\x1fb"));
    try testing.expectEqualStrings("a%7Fb", percentEncode(&buf, "a\x7fb"));
    try testing.expectEqualStrings("100%25", percentEncode(&buf, "100%"));
    try testing.expectEqualStrings("a%0Ab", percentEncode(&buf, "a\nb"));
}

test "a truncated grpc-message never ends in a partial escape" {
    // A message of section signs (two non-ASCII bytes each, six encoded
    // bytes) overflows the 256-byte grpc-message budget. The cut must fall
    // between escapes, whatever the buffer size, or a conformant client
    // decoding the header sees a malformed `%XX`.
    const msg = "§" ** 100;
    var sizes = [_]usize{ 256, 255, 254, 253, 100, 7, 4, 3, 2, 1, 0 };
    for (&sizes) |size| {
        var buf: [256]u8 = undefined;
        const enc = percentEncode(buf[0..size], msg);
        try testing.expect(enc.len <= size);
        // Every escape is complete: the encoding is a whole number of `%XX`.
        try testing.expectEqual(@as(usize, 0), enc.len % 3);
        var i: usize = 0;
        while (i < enc.len) : (i += 3) {
            try testing.expectEqual(@as(u8, '%'), enc[i]);
            _ = try std.fmt.charToDigit(enc[i + 1], 16);
            _ = try std.fmt.charToDigit(enc[i + 2], 16);
        }
        // And decoding gives back a prefix of the message.
        var dec: [256]u8 = undefined;
        const d = percentDecode(&dec, enc);
        try testing.expectEqualStrings(msg[0..d.len], d);
    }
    // Mixed content: the escape that does not fit is dropped whole, and
    // nothing after it is written, so the output is a prefix of the encoding.
    var small: [5]u8 = undefined;
    try testing.expectEqualStrings("a%C2", percentEncode(&small, "a§b"));
}

test "every status code has distinct text matching its number" {
    const all = [_]Status{
        .ok,                 .cancelled,           .unknown,        .invalid_argument,
        .deadline_exceeded,  .not_found,           .already_exists, .permission_denied,
        .resource_exhausted, .failed_precondition, .aborted,        .out_of_range,
        .unimplemented,      .internal,            .unavailable,    .data_loss,
        .unauthenticated,
    };
    for (all) |s| {
        const parsed = try std.fmt.parseInt(u32, s.text(), 10);
        try testing.expectEqual(@intFromEnum(s), parsed);
    }
}

test "percent encode/decode round-trips the spec's section signs" {
    // Every UNIMPLEMENTED message cites a section, and `§` is U+00A7, two
    // bytes, both escaped. A client that does not decode reads `%C2%A7`.
    const cases = [_][]const u8{
        "/qdrant.Points/Scroll is not implemented by strawmann (listed in spec §2 as a later phase)",
        "payload filtering (phase 3)",
        "sharding and replication (§1 non-goal); only value 1 is supported",
        "100% of the time",
        "",
        "\x00\x01\x7f\xff",
    };
    var enc_buf: [1024]u8 = undefined;
    var dec_buf: [1024]u8 = undefined;
    for (cases) |c| {
        const enc = percentEncode(&enc_buf, c);
        // The encoded form is header-safe: printable ASCII only.
        for (enc) |b| try std.testing.expect(b >= 0x20 and b <= 0x7e);
        try std.testing.expectEqualStrings(c, percentDecode(&dec_buf, enc));
    }
}

test "percentDecode passes malformed escapes through rather than dropping the message" {
    var buf: [64]u8 = undefined;
    // A stray `%` at the end, and a non-hex escape. Diagnostic text is worth
    // more mangled than absent.
    try std.testing.expectEqualStrings("50%", percentDecode(&buf, "50%"));
    try std.testing.expectEqualStrings("%zz ok", percentDecode(&buf, "%zz ok"));
    try std.testing.expectEqualStrings("a%", percentDecode(&buf, "a%"));
}

/// Walk the frames in `out` for `stream_id`, returning the DATA payload
/// concatenated into `body_out` and the number of DATA frames, checking each
/// against `max_frame`.
fn collectData(out: *const h2.OutBuf, stream_id: u31, max_frame: usize, body_out: []u8) !struct { len: usize, frames: usize, saw_trailers: bool } {
    var pos: usize = 0;
    var len: usize = 0;
    var frames: usize = 0;
    var saw_trailers = false;
    while (pos + h2.frame_header_len <= out.len) {
        const h = h2.FrameHeader.parse(out.buf[pos..][0..h2.frame_header_len]);
        const payload = out.buf[pos + h2.frame_header_len ..][0..h.length];
        pos += h2.frame_header_len + h.length;
        if (h.stream_id != stream_id) continue;
        if (h.frame_type == .data) {
            try testing.expect(h.length <= max_frame);
            @memcpy(body_out[len..][0..payload.len], payload);
            len += payload.len;
            frames += 1;
        } else if (h.frame_type == .headers and h.hasFlag(h2.Flags.end_stream)) {
            saw_trailers = true;
        }
    }
    return .{ .len = len, .frames = frames, .saw_trailers = saw_trailers };
}

test "a large body is split into DATA frames no larger than the peer's max frame size" {
    // hyper's h2 answers a DATA frame over its SETTINGS_MAX_FRAME_SIZE (16 384
    // by default) with FRAME_SIZE_ERROR and a GOAWAY. A QueryBatch of 16 x
    // limit 100 encodes to more than that, so every such response killed the
    // connection.
    const bufs = try testBuffers(testing.allocator, 2, 64);
    defer freeBuffers(testing.allocator, bufs);
    var c = testConn(bufs);
    const idx = try c.openStream(1);

    const body = try testing.allocator.alloc(u8, 100 * 1024);
    defer testing.allocator.free(body);
    for (body, 0..) |*b, i| b.* = @truncate(i *% 31);

    const ob = try testing.allocator.alloc(u8, 256 * 1024);
    defer testing.allocator.free(ob);
    var out = h2.OutBuf.init(ob);
    try writeResponse(&c, &out, idx, body);

    const got = try testing.allocator.alloc(u8, body.len + prefix_len);
    defer testing.allocator.free(got);
    const r = try collectData(&out, 1, 16384, got);
    try testing.expectEqual(prefix_len + body.len, r.len);
    try testing.expectEqual(@as(usize, 7), r.frames); // ceil(102405 / 16384)
    try testing.expect(r.saw_trailers);
    try testing.expectEqualSlices(u8, body, try decodeMessage(got[0..r.len]));
    // And both windows were charged for exactly the bytes sent.
    try testing.expectEqual(@as(i32, 5 * 1024 * 1024 - @as(i32, @intCast(r.len))), c.send_window);
    try testing.expectEqual(@as(i32, 2 * 1024 * 1024 - @as(i32, @intCast(r.len))), c.streams[idx].window);
}

test "a body larger than the stream window is parked and completes after WINDOW_UPDATE" {
    const bufs = try testBuffers(testing.allocator, 2, 64);
    defer freeBuffers(testing.allocator, bufs);
    // A peer at the RFC defaults: 65535-byte windows and 16384-byte frames.
    var c = h2.Connection.init(bufs);
    const idx = try c.openStream(1);
    const sid = c.streams[idx].id;

    const body = try testing.allocator.alloc(u8, 100 * 1024);
    defer testing.allocator.free(body);
    @memset(body, 0xab);
    const total = prefix_len + body.len;

    const ob = try testing.allocator.alloc(u8, 256 * 1024);
    defer testing.allocator.free(ob);
    var out = h2.OutBuf.init(ob);

    // The one-shot writer refuses rather than overrun the window.
    try testing.expectError(Error.WindowExhausted, writeResponse(&c, &out, idx, body));
    try testing.expectEqual(@as(usize, 0), out.len);

    // The incremental writer sends exactly the window and stops.
    try writeResponseHeaders(&out, sid);
    var off = writeData(&c, &out, idx, body, 0);
    try testing.expectEqual(@as(usize, 65535), off);
    try testing.expectEqual(@as(i32, 0), c.send_window);
    try testing.expectEqual(@as(i32, 0), c.streams[idx].window);
    // Nothing more can go out: parked.
    try testing.expectEqual(off, writeData(&c, &out, idx, body, off));

    // Only the stream window opens: still parked on the connection window.
    var wu: [h2.frame_header_len + 4]u8 = undefined;
    (h2.FrameHeader{ .length = 4, .frame_type = .window_update, .flags = 0, .stream_id = sid }).write(wu[0..h2.frame_header_len]);
    std.mem.writeInt(u32, wu[h2.frame_header_len..][0..4], 1 << 20, .big);
    _ = try c.readFrame(&wu, &out);
    try testing.expectEqual(off, writeData(&c, &out, idx, body, off));

    // Then the connection window: the rest goes out.
    (h2.FrameHeader{ .length = 4, .frame_type = .window_update, .flags = 0, .stream_id = 0 }).write(wu[0..h2.frame_header_len]);
    _ = try c.readFrame(&wu, &out);
    off = writeData(&c, &out, idx, body, off);
    try testing.expectEqual(total, off);
    try writeTrailers(&out, sid, .ok, "");

    const got = try testing.allocator.alloc(u8, total);
    defer testing.allocator.free(got);
    const r = try collectData(&out, sid, 16384, got);
    try testing.expectEqual(total, r.len);
    try testing.expect(r.saw_trailers);
    try testing.expectEqualSlices(u8, body, try decodeMessage(got[0..r.len]));
}

test "writeData stops at the output buffer's reserve and resumes after a flush" {
    const bufs = try testBuffers(testing.allocator, 2, 64);
    defer freeBuffers(testing.allocator, bufs);
    var c = testConn(bufs);
    const idx = try c.openStream(1);

    const body = try testing.allocator.alloc(u8, 40 * 1024);
    defer testing.allocator.free(body);
    for (body, 0..) |*b, i| b.* = @truncate(i);
    const total = prefix_len + body.len;

    // Room for one full frame and a bit, then the reserve.
    var ob: [16384 + h2.frame_header_len + 100 + h2.OutBuf.reserve]u8 = undefined;
    var out = h2.OutBuf.init(&ob);
    var got: [prefix_len + 40 * 1024]u8 = undefined;
    var got_len: usize = 0;

    var off: usize = 0;
    var flushes: usize = 0;
    while (off < total) {
        const next = writeData(&c, &out, idx, body, off);
        try testing.expect(next > off); // an empty buffer always makes progress
        try testing.expect(out.available() >= h2.OutBuf.reserve);
        off = next;
        const r = try collectData(&out, 1, 16384, got[got_len..]);
        got_len += r.len;
        out.reset(); // the flush
        flushes += 1;
    }
    try testing.expectEqual(total, got_len);
    try testing.expectEqualSlices(u8, body, try decodeMessage(got[0..got_len]));
    // 40 KiB in ~16 KiB pieces: at least three flushes were needed.
    try testing.expect(flushes >= 3);
}
