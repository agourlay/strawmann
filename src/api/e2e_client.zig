//! The client, harness and message builders `e2e.zig`'s tests are written in.
//!
//! Split out of `e2e.zig`, which at 4,429 lines was the largest file in the
//! tree: sixty tests over one hand-written gRPC client. The client is not a
//! test, and keeping it in the same file meant the tests could not be divided
//! by subject without dividing it too.
//!
//! Six declarations are `pub` for that reason and no other. This module is
//! reached only by `api.zig`'s test import; nothing ships it.
//! End-to-end tests: a real socket, a real HTTP/2 handshake, real gRPC frames.
//!
//! M1's exit criterion is "`bfb -n 100k -d 768 --search` completes end-to-end".
//! `bfb` is a Rust binary that is not part of this repo's build, so these tests
//! stand in for it by speaking the same wire protocol from the same side: a
//! client connects over TCP, sends the connection preface, and issues the exact
//! RPC sequence §2 says `qdrant-client` issues.
//!
//! They exist because every layer below is unit-tested in isolation and unit
//! tests cannot catch the interesting failures here, a HEADERS frame the
//! decoder accepts but the handler routes wrongly, a response that is three
//! frames instead of two, trailers that never arrive. §8.5's T0 tier calls this
//! out directly: "This catches 'right numbers, wrong shape,' which no amount of
//! score comparison will find."

const std = @import("std");
const linux = std.os.linux;

const strawmann_net = @import("../net/net.zig");
const proto = @import("../proto/proto.zig");
const api = @import("handlers.zig");

const h2 = strawmann_net.h2;
const hpack = strawmann_net.hpack;
const grpc = strawmann_net.grpc;
const server = strawmann_net.server;
const wire = proto.wire;

// =========================================================================
// A minimal gRPC client, sufficient to drive the server the way tonic does.
// =========================================================================

pub const Client = struct {
    /// Accumulated unparsed bytes: room for one max-size gRPC message.
    pub const buf_len = 1 << 20;

    // `Client` is returned by value from `connect` and copied freely by the
    // tests (`var e = extra.*`, an ArrayList of them). With `buf` inline that
    // made it a 1 MiB value, and Debug codegen materialises every intermediate
    // -- the literal, the local, the error-union return, the errdefer copy --
    // so `connectWith` carried a 10.6 MiB frame and the connection-slot test
    // a 9.5 MiB one, twenty megabytes of stack to send a twelve-byte SETTINGS
    // frame. Whether that fit depended on the invoking shell's soft stack
    // limit (8 MiB: SIGSEGV in the stack probe at test 147; 32 MiB: green),
    // which is not a property a test suite may have.
    //
    // What is left is 59,760 bytes, 59,440 of them `hpack.Decoder`'s dynamic
    // table, so the bound is that plus headroom rather than a round number:
    // it exists to refuse the next megabyte, not to argue about the decoder.
    comptime {
        std.debug.assert(@sizeOf(Client) <= 64 * 1024);
    }

    fd: linux.fd_t,
    next_stream: u31 = 1,
    dec: hpack.Decoder = hpack.Decoder.init(),
    /// Heap-allocated for the reason above; owned by this client, freed by
    /// `close`. Copies share it, as they shared the fd before.
    buf: *[buf_len]u8,
    len: usize = 0,
    /// What this client advertised, and enforces the way hyper's h2 does: a
    /// DATA frame longer than `max_frame_size` is a FRAME_SIZE_ERROR there
    /// (a GOAWAY, and every in-flight call fails with a transport error), so
    /// it is a test failure here. Likewise more DATA than the connection
    /// window allows is a FLOW_CONTROL_ERROR.
    max_frame_size: u32 = hyper_max_frame_size,
    conn_window: i64 = hyper_conn_window,
    /// Streams this client has RST_STREAM'd. Frames still arriving for them
    /// are dropped, their DATA handed back to the connection window as h2
    /// does, and counted in `late_cancelled`.
    reset_sids: [64]u31 = undefined,
    reset_n: usize = 0,
    late_cancelled: usize = 0,
    /// PING ACKs seen, to check none was lost while the server's output was
    /// backed up.
    ping_acks: usize = 0,
    /// Set by a test that reads what a server wrote before closing the
    /// connection: the WINDOW_UPDATEs and SETTINGS acks sent while parsing
    /// then fail with EPIPE, which is not what such a test is checking.
    peer_may_be_gone: bool = false,

    /// hyper 1.10 client defaults (`proto/h2/client.rs`): 16 KiB frames, a
    /// 2 MiB stream window and a 5 MiB connection window.
    const hyper_max_frame_size: u32 = 16 * 1024;
    const hyper_stream_window: u32 = 2 * 1024 * 1024;
    const hyper_conn_window: u32 = 5 * 1024 * 1024;

    pub fn connect(port: u16) !Client {
        return connectWith(port, .{});
    }

    const Settings = struct {
        stream_window: u32 = hyper_stream_window,
        conn_window: u32 = hyper_conn_window,
        /// SO_RCVBUF, to make a client that stops reading push back on the
        /// server's socket quickly.
        rcvbuf: ?c_int = null,
    };

    /// Connect and advertise the given windows. The defaults are hyper's;
    /// smaller ones make the server park bodies and wait for WINDOW_UPDATE.
    pub fn connectWith(port: u16, settings: Settings) !Client {
        const fd = try server.sys.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, linux.IPPROTO.TCP);
        errdefer server.sys.close(fd);

        if (settings.rcvbuf) |n| try server.sys.setsockoptInt(fd, linux.SOL.SOCKET, linux.SO.RCVBUF, n);
        // As tonic does. Without it two small writes in a row (a RST_STREAM
        // and the next request, say) wait on the delayed ACK, ~40 ms.
        server.setNoDelay(fd);
        const addr = server.Address.init(server.Address.loopback, port);
        const sa = addr.sockaddrIn();
        const rc = linux.connect(fd, @ptrCast(&sa), @sizeOf(linux.sockaddr.in));
        if (linux.errno(rc) != .SUCCESS) return error.ConnectFailed;

        const buf = try std.heap.page_allocator.create([buf_len]u8);
        errdefer std.heap.page_allocator.destroy(buf);
        var c = Client{ .fd = fd, .buf = buf, .conn_window = settings.conn_window };
        // §6.1: h2c with prior knowledge, preface then SETTINGS, no upgrade.
        try c.writeAll(h2.client_preface);
        var ob: [64]u8 = undefined;
        var out = h2.OutBuf.init(&ob);
        // The SETTINGS and stream-0 WINDOW_UPDATE hyper's h2 sends first.
        var st: [12]u8 = undefined;
        std.mem.writeInt(u16, st[0..2], @intFromEnum(h2.SettingId.initial_window_size), .big);
        std.mem.writeInt(u32, st[2..6], settings.stream_window, .big);
        std.mem.writeInt(u16, st[6..8], @intFromEnum(h2.SettingId.max_frame_size), .big);
        std.mem.writeInt(u32, st[8..12], hyper_max_frame_size, .big);
        try out.frame(.settings, 0, 0, &st);
        if (settings.conn_window > 65535) {
            var wu: [4]u8 = undefined;
            std.mem.writeInt(u32, &wu, settings.conn_window - 65535, .big);
            try out.frame(.window_update, 0, 0, &wu);
        }
        try c.writeAll(out.written());
        return c;
    }

    /// Hand back the window a DATA frame consumed, on both the connection and
    /// the stream, as h2 does when the body is read.
    pub fn releaseCapacity(self: *Client, sid: u31, n: u32) !void {
        if (n == 0) return;
        var wb: [32]u8 = undefined;
        var wo = h2.OutBuf.init(&wb);
        var wu: [4]u8 = undefined;
        std.mem.writeInt(u32, &wu, n, .big);
        try wo.frame(.window_update, 0, 0, &wu);
        if (sid != 0) try wo.frame(.window_update, 0, sid, &wu);
        self.writeAll(wo.written()) catch |e| if (!self.peer_may_be_gone) return e;
        self.conn_window += n;
    }

    pub fn close(self: *Client) void {
        server.sys.close(self.fd);
        std.heap.page_allocator.destroy(self.buf);
    }

    pub fn writeAll(self: *Client, bytes: []const u8) !void {
        var sent: usize = 0;
        while (sent < bytes.len) {
            const n = try server.sys.write(self.fd, bytes[sent..]) orelse continue;
            if (n == 0) return error.Closed;
            sent += n;
        }
    }

    /// Issue a unary RPC and return the decoded response message.
    ///
    /// `out` receives the response body; the returned slice points into it.
    pub fn call(self: *Client, path: []const u8, body: []const u8, out: []u8) !Response {
        const sid = try self.send(path, body);
        return self.readResponse(sid, out);
    }

    /// Send a unary request without waiting for the answer, returning its
    /// stream id for a later `readResponse`. Several may be in flight.
    pub fn send(self: *Client, path: []const u8, body: []const u8) !u31 {
        const sid = self.next_stream;
        self.next_stream += 2; // client streams are odd

        var frame_buf: [1 << 20]u8 = undefined;
        var ob = h2.OutBuf.init(&frame_buf);

        // HEADERS, exactly the set tonic sends.
        var hb: [512]u8 = undefined;
        var hn: usize = 0;
        hn += try hpack.Encoder.writeHeader(hb[hn..], ":method", "POST");
        hn += try hpack.Encoder.writeHeader(hb[hn..], ":scheme", "http");
        hn += try hpack.Encoder.writeHeader(hb[hn..], ":path", path);
        hn += try hpack.Encoder.writeHeader(hb[hn..], ":authority", "localhost");
        hn += try hpack.Encoder.writeHeader(hb[hn..], "content-type", "application/grpc");
        // §6.1: "Tonic sends `te: trailers` and expects trailers even on success."
        hn += try hpack.Encoder.writeHeader(hb[hn..], "te", "trailers");
        try ob.frame(.headers, h2.Flags.end_headers, sid, hb[0..hn]);

        // DATA: 5-byte gRPC prefix + message, END_STREAM.
        var payload: [1 << 20]u8 = undefined;
        grpc.writePrefix(payload[0..grpc.prefix_len], @intCast(body.len));
        @memcpy(payload[grpc.prefix_len..][0..body.len], body);
        try ob.frame(.data, h2.Flags.end_stream, sid, payload[0 .. grpc.prefix_len + body.len]);

        try self.writeAll(ob.written());
        return sid;
    }

    /// RST_STREAM(CANCEL) a request in flight, as tonic does when a call's
    /// future is dropped or its deadline passes.
    pub fn reset(self: *Client, sid: u31) !void {
        var fb: [32]u8 = undefined;
        var fo = h2.OutBuf.init(&fb);
        var code: [4]u8 = undefined;
        std.mem.writeInt(u32, &code, @intFromEnum(h2.ErrorCode.cancel), .big);
        try fo.frame(.rst_stream, 0, sid, &code);
        try self.writeAll(fo.written());
        if (self.reset_n < self.reset_sids.len) {
            self.reset_sids[self.reset_n] = sid;
            self.reset_n += 1;
        }
    }

    pub fn wasReset(self: *const Client, sid: u31) bool {
        for (self.reset_sids[0..self.reset_n]) |r| if (r == sid) return true;
        return false;
    }

    pub fn ping(self: *Client) !void {
        var fb: [32]u8 = undefined;
        var fo = h2.OutBuf.init(&fb);
        try fo.frame(.ping, 0, 0, "abcdefgh");
        try self.writeAll(fo.written());
    }

    /// Buffer (without consuming) until at least `min_bytes` of DATA for
    /// `sid` have arrived, or the read deadline passes. Lets a test act on a
    /// server-side state, a body parked on an exhausted window, say, from a
    /// client-side observable instead of a sleep.
    pub fn awaitData(self: *Client, sid: u31, min_bytes: usize) !void {
        var deadline: usize = 0;
        while (true) {
            var pos: usize = 0;
            var seen: usize = 0;
            while (pos + h2.frame_header_len <= self.len) {
                const fh = h2.FrameHeader.parse(self.buf[pos..][0..h2.frame_header_len]);
                const total = h2.frame_header_len + fh.length;
                if (pos + total > self.len) break;
                if (fh.frame_type == .goaway) return error.Goaway;
                if (fh.stream_id == sid and fh.frame_type == .data) seen += fh.length;
                pos += total;
            }
            if (seen >= min_bytes) return;
            const got = try server.sys.read(self.fd, self.buf[self.len..]) orelse {
                deadline += 1;
                if (deadline > 100_000) return error.Timeout;
                continue;
            };
            if (got == 0) return error.Closed;
            self.len += got;
        }
    }

    /// Drop buffered frames for streams this client has reset, releasing
    /// their DATA to the connection window as `readResponse` does, so a test
    /// that resets without reading anything else does not starve the
    /// server's connection window.
    pub fn dropResetFrames(self: *Client) !void {
        var pos: usize = 0;
        var keep: usize = 0;
        while (pos + h2.frame_header_len <= self.len) {
            const fh = h2.FrameHeader.parse(self.buf[pos..][0..h2.frame_header_len]);
            const total = h2.frame_header_len + fh.length;
            if (pos + total > self.len) break;
            const start = pos;
            pos += total;
            if (fh.stream_id != 0 and self.wasReset(fh.stream_id)) {
                self.late_cancelled += 1;
                if (fh.frame_type == .data) {
                    self.conn_window -= fh.length;
                    try self.releaseCapacity(0, fh.length);
                }
                continue;
            }
            std.mem.copyForwards(u8, self.buf[keep..][0..total], self.buf[start..pos]);
            keep += total;
        }
        const rest = self.len - pos;
        std.mem.copyForwards(u8, self.buf[keep..][0..rest], self.buf[pos..self.len]);
        self.len = keep + rest;
    }

    /// The last stream id named by a GOAWAY buffered on this client, if any.
    pub fn goawayLastStream(self: *const Client) ?u32 {
        var pos: usize = 0;
        while (pos + h2.frame_header_len <= self.len) {
            const fh = h2.FrameHeader.parse(self.buf[pos..][0..h2.frame_header_len]);
            const total = h2.frame_header_len + fh.length;
            if (pos + total > self.len) break;
            if (fh.frame_type == .goaway) {
                return std.mem.readInt(u32, self.buf[pos + h2.frame_header_len ..][0..4], .big);
            }
            pos += total;
        }
        return null;
    }

    /// Complete frames buffered for streams other than 0: responses to
    /// requests nobody has read yet. Once every expected response has been
    /// read, anything left here was sent for a stream that should have had
    /// nothing, a cancelled one, say.
    pub fn retainedFrames(self: *const Client) usize {
        var pos: usize = 0;
        var n: usize = 0;
        while (pos + h2.frame_header_len <= self.len) {
            const fh = h2.FrameHeader.parse(self.buf[pos..][0..h2.frame_header_len]);
            const total = h2.frame_header_len + fh.length;
            if (pos + total > self.len) break;
            if (fh.stream_id != 0) n += 1;
            pos += total;
        }
        return n;
    }

    pub const Response = struct {
        status: grpc.Status,
        message: []const u8,
        body: []const u8,
        /// Frames observed for this stream, so a test can assert the shape
        /// (§8.5 T0) and not merely the values.
        headers_frames: usize,
        data_frames: usize,
        saw_end_stream: bool,
    };

    pub fn readResponse(self: *Client, sid: u31, out: []u8) !Response {
        var r = Response{
            .status = .unknown,
            .message = "",
            .body = out[0..0],
            .headers_frames = 0,
            .data_frames = 0,
            .saw_end_stream = false,
        };
        var body_len: usize = 0;
        var msg_buf: [256]u8 = undefined;
        var msg_len: usize = 0;

        var deadline: usize = 0;
        while (!r.saw_end_stream) {
            // Parse whatever is already buffered before reading more. Frames
            // for other streams (responses to requests still in flight) are
            // kept, compacted to the front, for the `readResponse` that wants
            // them; frames on stream 0 are handled and dropped.
            var pos: usize = 0;
            var keep: usize = 0;
            while (pos + h2.frame_header_len <= self.len) {
                const fh = h2.FrameHeader.parse(self.buf[pos..][0..h2.frame_header_len]);
                const total = h2.frame_header_len + fh.length;
                if (pos + total > self.len) break;
                const payload = self.buf[pos + h2.frame_header_len ..][0..fh.length];
                const frame_start = pos;
                pos += total;

                if (fh.length > self.max_frame_size) return error.FrameSizeError;
                if (fh.frame_type == .goaway) return error.Goaway;

                if (fh.stream_id != 0 and fh.stream_id != sid) {
                    if (self.wasReset(fh.stream_id)) {
                        // A response that beat our RST_STREAM to the server:
                        // legal, dropped, and accounted for.
                        self.late_cancelled += 1;
                        if (fh.frame_type == .data) {
                            self.conn_window -= fh.length;
                            try self.releaseCapacity(0, fh.length);
                        }
                        continue;
                    }
                    std.mem.copyForwards(u8, self.buf[keep..][0..total], self.buf[frame_start..pos]);
                    keep += total;
                    continue;
                }
                if (fh.frame_type == .rst_stream and fh.stream_id == sid) return error.StreamReset;
                if (fh.frame_type == .data) {
                    self.conn_window -= fh.length;
                    if (self.conn_window < 0) return error.FlowControlError;
                }

                if (fh.stream_id != sid) {
                    // SETTINGS / WINDOW_UPDATE / PING on stream 0. Ack SETTINGS
                    // so the server does not consider us unresponsive.
                    if (fh.frame_type == .ping and fh.hasFlag(h2.Flags.ack)) self.ping_acks += 1;
                    if (fh.frame_type == .settings and !fh.hasFlag(h2.Flags.ack)) {
                        var ab: [16]u8 = undefined;
                        var ao = h2.OutBuf.init(&ab);
                        try ao.frame(.settings, h2.Flags.ack, 0, &.{});
                        self.writeAll(ao.written()) catch |e| if (!self.peer_may_be_gone) return e;
                    }
                    continue;
                }

                switch (fh.frame_type) {
                    .headers => {
                        r.headers_frames += 1;
                        const hdrs = try self.dec.decode(payload);
                        for (hdrs) |hd| {
                            if (std.mem.eql(u8, hd.name, "grpc-status")) {
                                r.status = @enumFromInt(try std.fmt.parseInt(u32, hd.value, 10));
                            } else if (std.mem.eql(u8, hd.name, "grpc-message")) {
                                // Decode, as a real client does. `§` is two
                                // non-ASCII bytes, so every message citing a
                                // spec section arrives percent-encoded.
                                const dec = grpc.percentDecode(&msg_buf, hd.value);
                                msg_len = dec.len;
                            }
                        }
                    },
                    .data => {
                        r.data_frames += 1;
                        // The message may span frames: the server splits DATA
                        // at the advertised max frame size.
                        if (body_len + payload.len > out.len) return error.ResponseTooLarge;
                        @memcpy(out[body_len..][0..payload.len], payload);
                        body_len += payload.len;
                        try self.releaseCapacity(sid, fh.length);
                    },
                    else => {},
                }
                if (fh.hasFlag(h2.Flags.end_stream)) {
                    // Stop here: a GOAWAY buffered behind a complete response
                    // must not turn that response into an error. Whatever
                    // follows stays buffered for the next call.
                    r.saw_end_stream = true;
                    break;
                }
            }
            if (pos > 0) {
                const rest = self.len - pos;
                std.mem.copyForwards(u8, self.buf[keep..][0..rest], self.buf[pos..self.len]);
                self.len = keep + rest;
            }
            if (r.saw_end_stream) break;

            const got = try server.sys.read(self.fd, self.buf[self.len..]) orelse {
                deadline += 1;
                if (deadline > 100_000) return error.Timeout;
                continue;
            };
            if (got == 0) return error.Closed;
            self.len += got;
        }

        // Strip the gRPC prefix now that the whole body is assembled.
        if (body_len > 0) {
            const inner = try grpc.decodeMessage(out[0..body_len]);
            std.mem.copyForwards(u8, out[0..inner.len], inner);
            body_len = inner.len;
        }
        r.body = out[0..body_len];
        // Copy the message into the response's own storage. The HPACK arena is
        // reused by the next call.
        const stable = out[body_len..][0..msg_len];
        @memcpy(stable, msg_buf[0..msg_len]);
        r.message = stable;
        return r;
    }
};

// =========================================================================
// Harness
// =========================================================================

pub const Harness = struct {
    alloc: std.mem.Allocator,
    engine: *api.Engine,
    contexts: []api.Context,
    srv: *server.Server,
    port: u16,

    /// The production default. Most tests run at this size.
    pub const request_buffer: usize = 1 << 20;

    pub fn start(alloc: std.mem.Allocator) !Harness {
        return startWith(alloc, request_buffer);
    }

    /// A harness whose server can map arena files, for the placement tests.
    /// `data_dir` must outlive the harness.
    pub fn startWithDataDir(alloc: std.mem.Allocator, data_dir: []const u8) !Harness {
        var h = try startWith(alloc, request_buffer);
        h.engine.data_dir = data_dir;
        return h;
    }

    /// A harness with a chosen per-stream request buffer, so the oversize path
    /// can be exercised without the test client having to build a body larger
    /// than its own frame buffer.
    pub fn startWith(alloc: std.mem.Allocator, req_buf: usize) !Harness {
        const engine = try alloc.create(api.Engine);
        engine.* = api.Engine.init(alloc);
        engine.default_capacity = 4096;

        const workers = 2;
        const max_dim = 1536;
        engine.max_dim = max_dim;
        const contexts = try alloc.alloc(api.Context, workers);
        for (contexts) |*c| {
            c.* = .{ .engine = engine, .workspace = try api.Workspace.init(alloc, max_dim) };
        }
        ctx_slots = contexts;
        next_slot = .init(0);

        const srv = try server.Server.init(alloc, .{
            .addr = server.Address.init(server.Address.loopback, 0),
            .io_threads = 1,
            .worker_threads = workers,
            .streams_per_conn = 4,
            .connections_per_io = 2,
            .request_buffer = req_buf,
            .response_buffer = 1 << 20,
        }, dispatch, @ptrCast(contexts.ptr));

        const port = try srv.bind();
        try srv.start();
        return .{ .alloc = alloc, .engine = engine, .contexts = contexts, .srv = srv, .port = port };
    }

    pub fn stop(self: *Harness) void {
        self.srv.stop();
        self.srv.deinit();
        for (self.contexts) |*c| c.workspace.deinit(self.alloc);
        self.alloc.free(self.contexts);
        self.engine.deinit();
        self.alloc.destroy(self.engine);
    }
};

var ctx_slots: []api.Context = &.{};
threadlocal var my_slot: ?usize = null;
var next_slot: std.atomic.Value(usize) = .init(0);

fn dispatch(_: *anyopaque, req: *const server.Request, out: *server.ResponseBuf) server.Completion {
    const slot = my_slot orelse blk: {
        const s = next_slot.fetchAdd(1, .monotonic) % @max(1, ctx_slots.len);
        my_slot = s;
        break :blk s;
    };
    return api.handle(@ptrCast(&ctx_slots[slot]), req, out);
}

// =========================================================================
// Message builders
// =========================================================================

pub fn buildCreateCollection(buf: []u8, name: []const u8, dim: u64, distance: i32) ![]const u8 {
    return buildCreateCollectionDt(buf, name, dim, distance, 0);
}

/// `datatype` is `VectorParams.datatype`, field 6: Default=0, Float32=1,
/// Uint8=2, Float16=3.
pub fn buildCreateCollectionDt(buf: []u8, name: []const u8, dim: u64, distance: i32, datatype: i32) ![]const u8 {
    var w = wire.Writer.init(buf);
    try w.writeStringField(1, name);
    {
        // `hnsw_config { full_scan_threshold = 0 }`. Every collection in
        // these tests is a few hundred points, well under Qdrant's default
        // threshold of 10 000 KB, so with the default every query here would
        // be answered by the plain scan (`handlers.fullScanPreferred`) and
        // the tests that exercise the graph path would test nothing. Asking
        // for the graph explicitly is what a bfb run on a small collection
        // has to do too. The threshold tests build their own create message.
        const hc = try w.beginNested(4, 2);
        try w.writeVarintFieldAlways(3, 0);
        try w.endNested(hc);
    }
    const vc = try w.beginNested(10, 2); // vectors_config (VERIFIED: field 10)
    const p = try w.beginNested(1, 2); // params
    try w.writeVarintField(1, dim);
    try w.writeVarintField(2, @intCast(distance));
    if (datatype != 0) try w.writeVarintField(6, @intCast(datatype));
    try w.endNested(p);
    try w.endNested(vc);
    return w.written();
}

pub fn buildUpsert(buf: []u8, name: []const u8, wait: bool, ids: []const u64, vecs: []const []const f32) ![]const u8 {
    var w = wire.Writer.init(buf);
    try w.writeStringField(1, name);
    try w.writeBoolField(2, wait);
    for (ids, vecs) |id, v| {
        const pt = try w.beginNested(3, 3);
        {
            const idn = try w.beginNested(1, 2);
            try w.writeVarintFieldAlways(1, id);
            try w.endNested(idn);
        }
        {
            // Vectors.vector = 1, Vector.dense = 101 (VERIFIED against the
            // generated client), DenseVector.data = 1.
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

pub fn buildQueryBatch(buf: []u8, name: []const u8, queries: []const []const f32, limit: u64) ![]const u8 {
    var w = wire.Writer.init(buf);
    try w.writeStringField(1, name);
    for (queries) |q| {
        const qp = try w.beginNested(2, 3);
        try w.writeStringField(1, name);
        {
            // Query.nearest = 1, VectorInput.dense = 2 (VERIFIED: `id` is 1).
            const query = try w.beginNested(3, 3);
            const nearest = try w.beginNested(1, 3);
            const dv = try w.beginNested(2, 3);
            try w.writePackedFloats(1, q);
            try w.endNested(dv);
            try w.endNested(nearest);
            try w.endNested(query);
        }
        // QueryPoints.limit = 8 (VERIFIED).
        try w.writeVarintFieldAlways(8, limit);
        try w.endNested(qp);
    }
    return w.written();
}
