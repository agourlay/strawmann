//! HTTP/2 (RFC 9113), the server-side subset needed for h2c with prior
//! knowledge.
//!
//! §6.1 defines the scope exactly:
//!
//!   "**h2c with prior knowledge, no TLS, no compression, server-side only.**
//!    No ALPN, no upgrade dance, no push, no priorities. This is a small enough
//!    subset that a hand-written HTTP/2 implementation is realistic, roughly:
//!    connection preface, SETTINGS, HEADERS/CONTINUATION, DATA, WINDOW_UPDATE,
//!    RST_STREAM, GOAWAY, PING, plus HPACK."
//!
//! §2 explains why the subset is this small: "`bfb` links `qdrant-client 1.18`,
//! which speaks gRPC over h2c with prior knowledge (the default
//! `--uri http://localhost:6334` implies no TLS, no ALPN). Compression is not
//! enabled. This collapses the transport problem enormously."
//!
//! ## Flow control is the part that bites
//!
//! §6.1: "A 100-point batch of 768-dim fp32 is ~307 KB. The default 64 KiB
//! initial window will stall it into five round trips. Advertise a large
//! `SETTINGS_INITIAL_WINDOW_SIZE` (e.g. 8 MiB) and keep the connection-level
//! window topped up aggressively."
//!
//! Note this is two separate windows. Raising only the stream window via
//! SETTINGS leaves the *connection* window at its fixed 64 KiB initial value -
//! SETTINGS cannot change it, and only a WINDOW_UPDATE on stream 0 can. A
//! server that raises one and not the other still stalls, just less obviously.
//! `sendInitialFrames` does both.
//!
//! The *send* side is the peer's business and cannot be raised from here:
//! hyper's h2 grants 2 MiB per stream and 5 MiB per connection, and answers a
//! DATA frame over its 16 384-byte SETTINGS_MAX_FRAME_SIZE with a GOAWAY.
//! `dataBudget` is the one place both limits are consulted, so a response
//! goes out in frames the peer accepts and stops, parked, when a window runs
//! dry until the peer's WINDOW_UPDATE arrives.

const std = @import("std");
const hpack = @import("hpack.zig");

/// RFC 9113 §3.4. Sent by the client immediately on connect.
pub const client_preface = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

pub const frame_header_len = 9;

pub const FrameType = enum(u8) {
    data = 0x0,
    headers = 0x1,
    priority = 0x2,
    rst_stream = 0x3,
    settings = 0x4,
    push_promise = 0x5,
    ping = 0x6,
    goaway = 0x7,
    window_update = 0x8,
    continuation = 0x9,
    _,
};

pub const Flags = struct {
    pub const end_stream: u8 = 0x1;
    pub const ack: u8 = 0x1; // SETTINGS and PING reuse bit 0
    pub const end_headers: u8 = 0x4;
    pub const padded: u8 = 0x8;
    pub const priority: u8 = 0x20;
};

/// RFC 9113 §7.
pub const ErrorCode = enum(u32) {
    no_error = 0x0,
    protocol_error = 0x1,
    internal_error = 0x2,
    flow_control_error = 0x3,
    settings_timeout = 0x4,
    stream_closed = 0x5,
    frame_size_error = 0x6,
    refused_stream = 0x7,
    cancel = 0x8,
    compression_error = 0x9,
    connect_error = 0xa,
    enhance_your_calm = 0xb,
    inadequate_security = 0xc,
    http_1_1_required = 0xd,
    _,
};

pub const SettingId = enum(u16) {
    header_table_size = 0x1,
    enable_push = 0x2,
    max_concurrent_streams = 0x3,
    initial_window_size = 0x4,
    max_frame_size = 0x5,
    max_header_list_size = 0x6,
    _,
};

pub const Error = error{
    /// A frame violated the protocol in a way that must close the connection.
    ProtocolError,
    /// A frame's length is illegal for its type.
    FrameSizeError,
    /// Flow-control accounting went negative or a window overflowed 2^31-1.
    FlowControlError,
    /// HPACK failed. RFC 9113 requires this be treated as a connection error.
    CompressionError,
    /// More concurrent streams than the pool allows.
    RefusedStream,
    /// The output buffer could not hold a frame we needed to write.
    OutputFull,
};

pub const FrameHeader = struct {
    length: u32,
    frame_type: FrameType,
    flags: u8,
    stream_id: u31,

    pub fn parse(buf: *const [frame_header_len]u8) FrameHeader {
        return .{
            .length = (@as(u32, buf[0]) << 16) | (@as(u32, buf[1]) << 8) | buf[2],
            .frame_type = @enumFromInt(buf[3]),
            .flags = buf[4],
            // The high bit is reserved and RFC 9113 §4.1 says a receiver must
            // ignore it rather than reject the frame.
            .stream_id = @truncate(std.mem.readInt(u32, buf[5..9], .big)),
        };
    }

    pub fn write(self: FrameHeader, buf: *[frame_header_len]u8) void {
        buf[0] = @truncate(self.length >> 16);
        buf[1] = @truncate(self.length >> 8);
        buf[2] = @truncate(self.length);
        buf[3] = @intFromEnum(self.frame_type);
        buf[4] = self.flags;
        std.mem.writeInt(u32, buf[5..9], self.stream_id, .big);
    }

    pub fn hasFlag(self: FrameHeader, f: u8) bool {
        return self.flags & f != 0;
    }
};

/// Settings we advertise, and the ones the peer has told us about.
pub const Settings = struct {
    /// One source of truth with the decoder: `hpack.Decoder` refuses a
    /// dynamic table size update above this, so advertising anything else
    /// would break a peer that took us at our word.
    header_table_size: u32 = hpack.default_table_size,
    enable_push: bool = false,
    max_concurrent_streams: u32 = default_max_streams,
    initial_window_size: u32 = default_window,
    max_frame_size: u32 = default_max_frame,
    /// Advertised by `sendInitialFrames`, because it *is* enforced: the
    /// decoder's arena refuses a block past this size (`hpack.zig`), and a
    /// block that cannot be decoded in full desynchronises the dynamic table
    /// for every later request, so the refusal is a COMPRESSION_ERROR for the
    /// connection. Advisory in the RFC (§6.5.2), but a limit a peer is not
    /// told about is one it cannot stay under: a tonic client attaching a
    /// large auth token or tracing baggage used to lose every in-flight
    /// stream to a limit it had no way to know.
    ///
    /// The peer's value is recorded and nothing reads it: what we send is
    /// `:status`, `content-type`, `grpc-status` and a `grpc-message` the
    /// encoder already bounds at 512 bytes of header block (`grpc.zig`), so
    /// there is no size decision here for a peer's limit to inform. This
    /// comment used to say the value sized what we send, which described a
    /// use the field has never had.
    max_header_list_size: u32 = 32 * 1024,

    /// §6.1: "Advertise a large `SETTINGS_INITIAL_WINDOW_SIZE` (e.g. 8 MiB)".
    /// The RFC's default is 65535, which stalls a 307 KB upsert batch into five
    /// round trips.
    pub const default_window: u32 = 8 * 1024 * 1024;

    /// The largest DATA payload we accept in one frame. The RFC permits
    /// 16384..16777215; 1 MiB keeps per-frame overhead negligible for large
    /// upserts without letting a peer force a huge single allocation.
    pub const default_max_frame: u32 = 1024 * 1024;

    pub const default_max_streams: u32 = 128;
};

/// Maximum concurrent streams, and therefore the size of the stream pool.
///
/// bfb's `-p` sets request concurrency and `-c 1` is its default, so all of it
/// multiplexes onto one connection (§6.1). 128 covers `-p 64` with headroom;
/// beyond it we return REFUSED_STREAM, which is a legal, retryable response
/// rather than a stall.
pub const max_streams = Settings.default_max_streams;

pub const StreamState = enum {
    idle,
    open,
    /// Client sent END_STREAM; we owe a response.
    half_closed_remote,
    /// Reset by the peer while a worker still owns the request. The slot is
    /// held, invisible to `findStream`, until the completion drains and drops
    /// the response; see `Event.reset`.
    closed,
};

pub const Stream = struct {
    id: u31 = 0,
    state: StreamState = .idle,
    /// Send window for this stream, from the peer's SETTINGS.
    window: i32 = 65535,

    /// The gRPC request body, accumulated across DATA frames.
    body: []u8 = &.{},
    body_len: usize = 0,
    /// Set when the body outgrew `body`, so the remaining DATA is drained and
    /// the stream is failed on `END_STREAM` instead of the connection dying.
    too_large: bool = false,
    /// A `.request` was produced for this slot, so a worker may hold pointers
    /// into `body`. Such a slot is only ever freed by `closeStream`, never by
    /// the peer: an RST_STREAM on it marks the stream `.closed` and the server
    /// frees it when the completion drains. Freeing it on RST let the next
    /// HEADERS reuse the slot under the worker, and the old completion's
    /// `closeStream` then reset the *new* stream, whose response was never
    /// sent.
    dispatched: bool = false,

    /// `:path`, copied out of the HPACK arena because that arena is reused by
    /// the next header block while this stream is still pending.
    path_buf: [128]u8 = undefined,
    path_len: usize = 0,

    pub fn path(self: *const Stream) []const u8 {
        return self.path_buf[0..self.path_len];
    }

    pub fn bodyBytes(self: *const Stream) []const u8 {
        return self.body[0..self.body_len];
    }

    fn reset(self: *Stream) void {
        self.state = .idle;
        self.body_len = 0;
        self.path_len = 0;
        self.id = 0;
        // Must be cleared with the rest: a slot reused after an over-sized
        // request would fail every subsequent request on that slot, and slots
        // are recycled constantly.
        self.too_large = false;
        self.dispatched = false;
    }
};

/// What `readFrame` produced.
pub const Event = union(enum) {
    /// Nothing complete yet; feed more bytes.
    need_more,
    /// A frame was consumed but produced no request.
    progress,
    /// A complete request is ready on this stream index.
    request: usize,
    /// The request body exceeded the per-stream buffer. Stream-level: the
    /// connection stays up and other streams are unaffected.
    request_too_large: usize,
    /// A live stream in this slot was taken away from the peer: it sent
    /// RST_STREAM, or we did (STREAM_CLOSED). A dispatched slot is held
    /// as `.closed` until its completion drains and calls `closeStream`; any
    /// other slot has already been freed. Either way the caller must drop
    /// whatever response it had in progress for the slot.
    reset: usize,
    /// The peer is going away.
    goaway,
};

/// A single client connection.
///
/// Owns its HPACK decoder (which must persist across requests, HPACK is
/// stateful per connection), its stream pool, and its flow-control windows.
pub const Connection = struct {
    hpack_dec: hpack.Decoder,
    streams: [max_streams]Stream,
    /// Peer's settings, governing what we may send.
    peer: Settings,
    /// Our settings, governing what the peer may send.
    local: Settings,

    /// Connection-level send window (what we may write).
    send_window: i32 = 65535,
    /// Connection-level receive window (what the peer may write). Topped up
    /// aggressively, see `maybeTopUpWindow`.
    recv_window: i32 = 65535,

    /// Number of stream slots backed by a request buffer. Slots beyond this
    /// exist in the array but can never be opened, see `openStream`.
    usable_streams: usize = 0,

    preface_seen: bool = false,
    goaway_sent: bool = false,
    /// The peer sent GOAWAY. RFC 9113 §6.8: it still expects responses on
    /// the streams already open, but is going away, so any stream it opens
    /// after this is refused (RST_STREAM REFUSED_STREAM, retryable) rather
    /// than dispatched to a worker whose answer would land on a closing
    /// connection. The server closes once the in-flight streams have drained.
    goaway_received: bool = false,
    /// Highest client stream id seen in a HEADERS frame, refused or not.
    /// RFC 9113 §5.1.1: ids must increase, so a new one at or below this is
    /// a stream the peer already closed (explicitly, or implicitly by using
    /// a higher id), and a DATA frame above it names a stream that was never
    /// opened. Also what GOAWAY reports as last-stream-id.
    last_stream_id: u31 = 0,

    /// Header block reassembly across CONTINUATION frames.
    header_accum: [64 * 1024]u8 = undefined,
    header_accum_len: usize = 0,
    header_stream: u31 = 0,
    expecting_continuation: bool = false,
    /// END_STREAM from the HEADERS frame that opened a CONTINUATION sequence.
    ///
    /// END_STREAM is carried on HEADERS, never on CONTINUATION, so a header
    /// block that spans frames must remember it. Dropping it left a
    /// bodyless request (HEADERS with END_STREAM but no END_HEADERS, any
    /// request whose header block exceeds one frame) permanently `.open`,
    /// holding a stream slot while the client waited for a response that could
    /// never be produced.
    header_end_stream: bool = false,
    /// The header block being accumulated belongs to a stream that was
    /// refused (pool exhausted, or opened after the peer's GOAWAY) or that is
    /// no longer accepting HEADERS (half-closed by the peer, or already
    /// closed). RFC 9113 §4.3 still requires the block to be decoded, because
    /// it may modify the HPACK dynamic table and skipping it desynchronises
    /// every later header block on the connection. The block is decoded and
    /// discarded, then RST_STREAM with this code is sent: REFUSED_STREAM for
    /// the former, STREAM_CLOSED for the latter (§5.1: a stream error, not a
    /// connection error, so the other streams live on).
    header_rst: ?ErrorCode = null,

    pub fn init(buffers: []([]u8)) Connection {
        var c: Connection = .{
            .hpack_dec = hpack.Decoder.init(),
            .streams = undefined,
            .peer = .{
                // RFC defaults until the peer's SETTINGS arrives. Assuming our
                // own generous values here would let us overrun a conforming
                // client on the very first response.
                .initial_window_size = 65535,
                .max_frame_size = 16384,
            },
            .local = .{},
        };
        for (&c.streams, 0..) |*s, i| {
            s.* = .{};
            s.body = if (i < buffers.len) buffers[i] else &.{};
        }
        c.usable_streams = @min(buffers.len, max_streams);
        // Advertise exactly what we can serve. A conforming client then never
        // exceeds it, and REFUSED_STREAM stays an edge case rather than the
        // normal path.
        c.local.max_concurrent_streams = @intCast(c.usable_streams);
        return c;
    }

    fn findStream(self: *Connection, id: u31) ?usize {
        for (&self.streams, 0..) |*s, i| {
            // A `.closed` slot is owned by a worker and no longer belongs to
            // the peer: frames for its id are treated as arriving on a stream
            // we already forgot, which RFC 9113 §5.1 lets us ignore.
            if (s.id == id and s.state != .idle and s.state != .closed) return i;
        }
        return null;
    }

    pub fn openStream(self: *Connection, id: u31) Error!usize {
        // A stream id already in use must reuse its slot. The ordinary gRPC
        // request-with-trailers shape (HEADERS, DATA, HEADERS+END_STREAM)
        // otherwise allocates a second slot for the same id on the trailing
        // HEADERS and leaks it for the life of the connection.
        if (self.findStream(id)) |existing| return existing;
        if (self.goaway_received) return Error.RefusedStream;

        // Bounded by `usable_streams`, not by the array length: a slot without
        // a request buffer would accept HEADERS and then fail on the first DATA
        // frame, which is a much worse failure than refusing the stream up
        // front.
        for (self.streams[0..self.usable_streams], 0..) |*s, i| {
            if (s.state == .idle) {
                s.reset();
                s.id = id;
                s.state = .open;
                s.window = @intCast(self.peer.initial_window_size);
                return i;
            }
        }
        return Error.RefusedStream;
    }

    pub fn closeStream(self: *Connection, idx: usize) void {
        self.streams[idx].reset();
    }

    /// Take a live stream away from the peer, whether it reset the stream or
    /// we did: the caller must drop whatever response it had in progress.
    fn dropStream(self: *Connection, i: usize) Event {
        const s = &self.streams[i];
        if (s.dispatched) {
            // A worker holds this slot's request buffer. Park it as
            // `.closed` and let the completion free it.
            s.state = .closed;
        } else {
            self.closeStream(i);
        }
        return .{ .reset = i };
    }

    /// RFC 9113 §5.1: a frame on a stream that is half-closed (remote) or
    /// closed is a *stream* error of type STREAM_CLOSED. Answer with
    /// RST_STREAM on that stream alone; the connection, and every other
    /// stream on it, carries on. It used to be mapped to a GOAWAY.
    fn streamClosed(self: *Connection, out: *OutBuf, id: u31) Error!Event {
        try self.sendRstStream(out, id, .stream_closed);
        const i = self.findStream(id) orelse return .progress;
        return self.dropStream(i);
    }

    // ---------------------------------------------------------------------
    // Output
    // ---------------------------------------------------------------------

    /// Write the frames a server must send before anything else.
    ///
    /// Both windows are raised here. SETTINGS covers the *stream* initial
    /// window for streams opened afterwards; the *connection* window is fixed
    /// at 65535 by the RFC and can only be raised by a WINDOW_UPDATE on stream
    /// 0. Doing only the first is the subtle version of this bug.
    pub fn sendInitialFrames(self: *Connection, out: *OutBuf) Error!void {
        var payload: [6 * 6]u8 = undefined;
        var n: usize = 0;
        n += writeSetting(payload[n..], .max_concurrent_streams, self.local.max_concurrent_streams);
        n += writeSetting(payload[n..], .initial_window_size, self.local.initial_window_size);
        n += writeSetting(payload[n..], .max_frame_size, self.local.max_frame_size);
        n += writeSetting(payload[n..], .header_table_size, self.local.header_table_size);
        n += writeSetting(payload[n..], .enable_push, 0);
        n += writeSetting(payload[n..], .max_header_list_size, self.local.max_header_list_size);
        try out.frame(.settings, 0, 0, payload[0..n]);

        const bump: u32 = self.local.initial_window_size - 65535;
        if (bump > 0) {
            var wu: [4]u8 = undefined;
            std.mem.writeInt(u32, &wu, bump, .big);
            try out.frame(.window_update, 0, 0, &wu);
            self.recv_window += @intCast(bump);
        }
    }

    fn writeSetting(buf: []u8, id: SettingId, value: u32) usize {
        std.mem.writeInt(u16, buf[0..2], @intFromEnum(id), .big);
        std.mem.writeInt(u32, buf[2..6], value, .big);
        return 6;
    }

    /// Replenish the connection receive window once it has drained past half.
    ///
    /// §61: "keep the connection-level window topped up aggressively." Waiting
    /// until the window is exhausted would serialise the upload; topping up at
    /// the halfway mark keeps the client writing continuously.
    pub fn maybeTopUpWindow(self: *Connection, out: *OutBuf) Error!void {
        const target: i32 = @intCast(self.local.initial_window_size);
        if (self.recv_window * 2 > target) return;
        const bump: u32 = @intCast(target - self.recv_window);
        if (bump == 0) return;
        var wu: [4]u8 = undefined;
        std.mem.writeInt(u32, &wu, bump, .big);
        try out.frame(.window_update, 0, 0, &wu);
        self.recv_window = target;
    }

    pub fn sendGoaway(self: *Connection, out: *OutBuf, code: ErrorCode) Error!void {
        if (self.goaway_sent) return;
        self.goaway_sent = true;
        var payload: [8]u8 = undefined;
        std.mem.writeInt(u32, payload[0..4], self.last_stream_id, .big);
        std.mem.writeInt(u32, payload[4..8], @intFromEnum(code), .big);
        try out.frame(.goaway, 0, 0, &payload);
    }

    pub fn sendRstStream(self: *Connection, out: *OutBuf, id: u31, code: ErrorCode) Error!void {
        _ = self;
        var payload: [4]u8 = undefined;
        std.mem.writeInt(u32, &payload, @intFromEnum(code), .big);
        try out.frame(.rst_stream, 0, id, &payload);
    }

    // ---------------------------------------------------------------------
    // Input
    // ---------------------------------------------------------------------

    /// Consume the client preface. Returns bytes consumed, or null if the
    /// buffer does not yet hold all 24.
    pub fn consumePreface(self: *Connection, data: []const u8) Error!?usize {
        if (self.preface_seen) return 0;
        if (data.len < client_preface.len) {
            // A short buffer that already disagrees is a hard error rather than
            // a wait, otherwise a plaintext HTTP/1.1 client hangs instead of
            // being told.
            if (!std.mem.startsWith(u8, client_preface, data)) return Error.ProtocolError;
            return null;
        }
        if (!std.mem.eql(u8, data[0..client_preface.len], client_preface)) return Error.ProtocolError;
        self.preface_seen = true;
        return client_preface.len;
    }

    /// Parse one frame from the front of `data`.
    ///
    /// Returns the number of bytes consumed and the resulting event. Consumes
    /// nothing and returns `need_more` when the buffer does not hold a full
    /// frame, so the caller can compact and read again.
    pub fn readFrame(self: *Connection, data: []const u8, out: *OutBuf) Error!struct { consumed: usize, event: Event } {
        if (data.len < frame_header_len) return .{ .consumed = 0, .event = .need_more };
        const h = FrameHeader.parse(data[0..frame_header_len]);

        if (h.length > self.local.max_frame_size) return Error.FrameSizeError;
        const total = frame_header_len + h.length;
        if (data.len < total) return .{ .consumed = 0, .event = .need_more };
        const payload = data[frame_header_len..total];

        // RFC 9113 §6.10: a CONTINUATION must immediately follow HEADERS or
        // another CONTINUATION on the same stream. Anything else interleaved is
        // a connection error, and is a known attack shape, since unbounded
        // interleaving lets a peer make the server buffer without limit.
        if (self.expecting_continuation and h.frame_type != .continuation) {
            return Error.ProtocolError;
        }

        const event: Event = switch (h.frame_type) {
            .settings => blk: {
                try self.handleSettings(h, payload, out);
                break :blk .progress;
            },
            .headers => try self.handleHeaders(h, payload, out),
            .continuation => try self.handleContinuation(h, payload, out),
            .data => try self.handleData(h, payload, out),
            .window_update => blk: {
                try self.handleWindowUpdate(h, payload);
                break :blk .progress;
            },
            .rst_stream => blk: {
                if (h.length != 4) return Error.FrameSizeError;
                if (h.stream_id == 0) return Error.ProtocolError;
                const i = self.findStream(h.stream_id) orelse break :blk .progress;
                break :blk self.dropStream(i);
            },
            .ping => blk: {
                if (h.length != 8) return Error.FrameSizeError;
                // RFC 9113 §6.7: PING is a connection-level frame.
                if (h.stream_id != 0) return Error.ProtocolError;
                if (!h.hasFlag(Flags.ack)) try out.frame(.ping, Flags.ack, 0, payload);
                break :blk .progress;
            },
            .goaway => blk: {
                // RFC 9113 §6.8: stream 0 only, and at least last-stream-id
                // plus an error code.
                if (h.stream_id != 0) return Error.ProtocolError;
                if (h.length < 8) return Error.FrameSizeError;
                self.goaway_received = true;
                break :blk .goaway;
            },
            // §6.1: "no push, no priorities". PRIORITY is deprecated by RFC
            // 9113 and must be ignored rather than rejected, but §6.3 still
            // makes one on stream 0 a PROTOCOL_ERROR; PUSH_PROMISE from a
            // client is always a protocol error.
            .priority => blk: {
                if (h.length != 5) return Error.FrameSizeError;
                if (h.stream_id == 0) return Error.ProtocolError;
                break :blk .progress;
            },
            .push_promise => return Error.ProtocolError,
            // Unknown frame types must be ignored (RFC 9113 §4.1), which is how
            // extensions stay deployable.
            else => .progress,
        };

        return .{ .consumed = total, .event = event };
    }

    fn handleSettings(self: *Connection, h: FrameHeader, payload: []const u8, out: *OutBuf) Error!void {
        if (h.stream_id != 0) return Error.ProtocolError;
        if (h.hasFlag(Flags.ack)) {
            if (h.length != 0) return Error.FrameSizeError;
            return;
        }
        if (h.length % 6 != 0) return Error.FrameSizeError;

        var i: usize = 0;
        while (i + 6 <= payload.len) : (i += 6) {
            const id: SettingId = @enumFromInt(std.mem.readInt(u16, payload[i..][0..2], .big));
            const value = std.mem.readInt(u32, payload[i + 2 ..][0..4], .big);
            switch (id) {
                .header_table_size => self.peer.header_table_size = value,
                .enable_push => {
                    if (value > 1) return Error.ProtocolError;
                    self.peer.enable_push = value == 1;
                },
                .max_concurrent_streams => self.peer.max_concurrent_streams = value,
                .initial_window_size => {
                    if (value > 0x7fffffff) return Error.FlowControlError;
                    // RFC 9113 §6.9.2: a change to this setting adjusts the
                    // window of every *existing* stream by the delta. Applying
                    // it only to new streams is a slow, rare stall that is
                    // extremely hard to reproduce.
                    const old: i64 = @intCast(self.peer.initial_window_size);
                    const delta: i64 = @as(i64, value) - old;
                    for (&self.streams) |*s| {
                        if (s.state == .idle) continue;
                        const nw: i64 = @as(i64, s.window) + delta;
                        if (nw > 0x7fffffff) return Error.FlowControlError;
                        s.window = @intCast(nw);
                    }
                    self.peer.initial_window_size = value;
                },
                .max_frame_size => {
                    if (value < 16384 or value > 16777215) return Error.ProtocolError;
                    self.peer.max_frame_size = value;
                },
                .max_header_list_size => self.peer.max_header_list_size = value,
                else => {}, // unknown settings are ignored
            }
        }
        try out.frame(.settings, Flags.ack, 0, &.{});
    }

    fn handleWindowUpdate(self: *Connection, h: FrameHeader, payload: []const u8) Error!void {
        if (h.length != 4) return Error.FrameSizeError;
        const inc = std.mem.readInt(u32, payload[0..4], .big) & 0x7fffffff;
        // RFC 9113 §6.9: a zero increment is a protocol error on stream 0 and a
        // stream error otherwise; treating both as connection errors is simpler
        // and no client sends one.
        if (inc == 0) return Error.ProtocolError;

        if (h.stream_id == 0) {
            const nw: i64 = @as(i64, self.send_window) + inc;
            if (nw > 0x7fffffff) return Error.FlowControlError;
            self.send_window = @intCast(nw);
        } else if (self.findStream(h.stream_id)) |i| {
            const nw: i64 = @as(i64, self.streams[i].window) + inc;
            if (nw > 0x7fffffff) return Error.FlowControlError;
            self.streams[i].window = @intCast(nw);
        } else if (h.stream_id > self.last_stream_id) {
            // RFC 9113 §5.1: WINDOW_UPDATE on an *idle* stream, one no HEADERS
            // has ever opened, is a connection error. Anything at or below the
            // highest id seen is a stream we already closed, and §6.9 requires
            // that one to be ignored rather than treated as an error: a peer
            // may legally credit a stream we answered and forgot. Same rule as
            // DATA in `handleData`.
            return Error.ProtocolError;
        }
    }

    fn handleHeaders(self: *Connection, h: FrameHeader, payload: []const u8, out: *OutBuf) Error!Event {
        if (h.stream_id == 0) return Error.ProtocolError;

        var block = payload;
        if (h.hasFlag(Flags.padded)) {
            if (block.len < 1) return Error.FrameSizeError;
            const pad = block[0];
            if (@as(usize, pad) + 1 > block.len) return Error.ProtocolError;
            block = block[1 .. block.len - pad];
        }
        if (h.hasFlag(Flags.priority)) {
            if (block.len < 5) return Error.FrameSizeError;
            block = block[5..]; // stream dependency + weight, ignored
        }

        // Which stream, and what to do with the block once decoded.
        var idx: ?usize = null;
        self.header_rst = null;
        if (self.findStream(h.stream_id)) |i| {
            // RFC 9113 §5.1: HEADERS on a half-closed (remote) stream is a
            // STREAM_CLOSED error, exactly as DATA is. Accepting it re-ran
            // `finishHeaders` on a slot a worker already owns and emitted a
            // second `.request` for it. Stream-level: the block is decoded
            // and dropped, then the stream is reset from `finishHeaders`.
            if (self.streams[i].state != .open) {
                self.header_rst = .stream_closed;
            } else {
                idx = i;
            }
        } else if (h.stream_id % 2 == 0) {
            // RFC 9113 §5.1.1: clients open odd-numbered streams only.
            return Error.ProtocolError;
        } else if (h.stream_id <= self.last_stream_id) {
            // RFC 9113 §5.1.1: ids increase, so this stream is closed, either
            // explicitly (reset by the peer while a worker held it, or already
            // answered, and a client that got a complete response early is
            // allowed its trailers) or implicitly by the use of a higher id.
            // §5.1 makes a frame on a closed stream a STREAM_CLOSED *stream*
            // error; the block is still decoded for the dynamic table.
            self.header_rst = .stream_closed;
        } else {
            self.last_stream_id = h.stream_id;
            idx = self.openStream(h.stream_id) catch |e| switch (e) {
                // Legal and retryable, much better than stalling the client.
                // The block is still decoded below (RFC 9113 §4.3), then the
                // RST goes out from `finishHeaders`.
                Error.RefusedStream => blk: {
                    self.header_rst = .refused_stream;
                    break :blk null;
                },
                else => return e,
            };
        }

        if (block.len > self.header_accum.len) return Error.CompressionError;
        @memcpy(self.header_accum[0..block.len], block);
        self.header_accum_len = block.len;
        self.header_stream = h.stream_id;

        if (!h.hasFlag(Flags.end_headers)) {
            self.expecting_continuation = true;
            self.header_end_stream = h.hasFlag(Flags.end_stream);
            return .progress;
        }
        return try self.finishHeaders(idx, h.hasFlag(Flags.end_stream), out);
    }

    fn handleContinuation(self: *Connection, h: FrameHeader, payload: []const u8, out: *OutBuf) Error!Event {
        if (!self.expecting_continuation or h.stream_id != self.header_stream) {
            return Error.ProtocolError;
        }
        if (self.header_accum_len + payload.len > self.header_accum.len) {
            return Error.CompressionError;
        }
        @memcpy(self.header_accum[self.header_accum_len..][0..payload.len], payload);
        self.header_accum_len += payload.len;

        if (!h.hasFlag(Flags.end_headers)) return .progress;
        self.expecting_continuation = false;

        const idx: ?usize = if (self.header_rst != null)
            null
        else
            self.findStream(h.stream_id) orelse return Error.ProtocolError;
        // END_STREAM belongs to the HEADERS frame that started this block, not
        // to the CONTINUATION that ends it, so it is replayed from there.
        const end_stream = self.header_end_stream;
        self.header_end_stream = false;
        return try self.finishHeaders(idx, end_stream, out);
    }

    fn finishHeaders(self: *Connection, idx: ?usize, end_stream: bool, out: *OutBuf) Error!Event {
        const headers = self.hpack_dec.decode(self.header_accum[0..self.header_accum_len]) catch {
            // RFC 9113 §4.3: HPACK failure is always a connection error,
            // because the decoder state is now unrecoverable.
            return Error.CompressionError;
        };
        self.header_accum_len = 0;

        const i = idx orelse {
            // Refused or closed: the block has done its work on the dynamic
            // table and is otherwise discarded.
            const code = self.header_rst orelse return Error.ProtocolError;
            self.header_rst = null;
            if (code == .stream_closed) return self.streamClosed(out, self.header_stream);
            try self.sendRstStream(out, self.header_stream, code);
            return .progress;
        };
        const s = &self.streams[i];
        for (headers) |hdr| {
            if (std.mem.eql(u8, hdr.name, ":path")) {
                // A `:path` longer than the buffer is an unknown method, not a
                // connection error: it is truncated and flows through the
                // ordinary UNIMPLEMENTED response for a path nothing matches.
                // Every real method name is far shorter, so a truncated path
                // can never equal one. It used to be a PROTOCOL_ERROR GOAWAY,
                // taking every other stream down with it.
                const n = @min(hdr.value.len, s.path_buf.len);
                @memcpy(s.path_buf[0..n], hdr.value[0..n]);
                s.path_len = n;
            }
        }

        // Request trailers on a body that overflowed: the stream was already
        // answered with RESOURCE_EXHAUSTED on the first frame that overflowed,
        // and the rest of the body is being drained. Dispatching here would
        // hand a worker the truncated body and produce a second response.
        if (s.too_large) {
            if (end_stream) s.state = .half_closed_remote;
            return .progress;
        }

        if (end_stream) {
            s.state = .half_closed_remote;
            s.dispatched = true;
            return .{ .request = i };
        }
        return .progress;
    }

    fn handleData(self: *Connection, h: FrameHeader, payload: []const u8, out: *OutBuf) Error!Event {
        if (h.stream_id == 0) return Error.ProtocolError;

        // Flow-control accounting is on the *padded* length, per RFC 9113 §6.9:
        // padding counts against the window even though it is not data.
        self.recv_window -= @intCast(h.length);
        if (self.recv_window < 0) return Error.FlowControlError;
        try self.maybeTopUpWindow(out);

        var body = payload;
        if (h.hasFlag(Flags.padded)) {
            if (body.len < 1) return Error.FrameSizeError;
            const pad = body[0];
            if (@as(usize, pad) + 1 > body.len) return Error.ProtocolError;
            body = body[1 .. body.len - pad];
        }

        const idx = self.findStream(h.stream_id) orelse {
            // RFC 9113 §5.1: DATA on an idle stream, one no HEADERS has ever
            // opened, is a connection error. Anything at or below the highest
            // id seen is a stream we already closed: a legal race, dropped.
            if (h.stream_id > self.last_stream_id) return Error.ProtocolError;
            return .progress;
        };
        const s = &self.streams[idx];
        // RFC 9113 §5.1: DATA on a half-closed (remote) stream is STREAM_CLOSED.
        // Accepting it would append to a body already dispatched and emit a
        // *second* `.request` for the same slot, so two workers would write the
        // same response buffer concurrently and two responses would go out on
        // one stream id. A stream error: RST_STREAM this stream, keep the rest.
        if (s.state != .open) return self.streamClosed(out, h.stream_id);

        // An over-sized body is a **stream** error, not a connection error.
        //
        // Returning `Error.RequestTooLarge` here made the caller send GOAWAY and
        // tear down the whole connection, taking every other in-flight stream
        // with it and giving the client nothing but "transport error" to work
        // from. A 1 MiB request buffer is a legitimate limit, it is sized for
        // bfb's `-b 100` at d=1536, but exceeding it is an ordinary client
        // mistake that deserves an ordinary `RESOURCE_EXHAUSTED`, the way gRPC
        // reports a message over `max_receive_message_length`.
        //
        // Found by uploading d=1536 vectors 512 to a batch: 3.1 MB per request,
        // and the connection simply died mid-upload.
        //
        // It is reported on the *first* frame that overflows, not at
        // END_STREAM. Waiting meant a body larger than the 8 MiB stream window
        // never got its error at all: no stream-level WINDOW_UPDATE is sent for
        // a body we are discarding, so the client stalled forever, holding the
        // slot. Reporting immediately lets the server answer while the client
        // is still uploading (RFC 9113 §8.1 allows a complete response before
        // the request has ended), and the remaining DATA is drained here.
        if (s.too_large) {
            // Draining. Credit the stream window back so a client that keeps
            // uploading is never stalled on a body nobody will read; the
            // connection window is topped up above as for any DATA.
            try self.creditStreamWindow(out, h.stream_id, h.length);
            if (h.hasFlag(Flags.end_stream)) s.state = .half_closed_remote;
            return .progress;
        }
        if (s.body_len + body.len > s.body.len) {
            s.too_large = true;
            try self.creditStreamWindow(out, h.stream_id, h.length);
            if (h.hasFlag(Flags.end_stream)) s.state = .half_closed_remote;
            return .{ .request_too_large = idx };
        }
        @memcpy(s.body[s.body_len..][0..body.len], body);
        s.body_len += body.len;

        if (h.hasFlag(Flags.end_stream)) {
            s.state = .half_closed_remote;
            s.dispatched = true;
            return .{ .request = idx };
        }
        return .progress;
    }

    /// Hand `n` bytes of stream-level window back to the peer. Only needed for
    /// a body being discarded: an accepted body fits the request buffer, which
    /// is smaller than the initial stream window, so it never runs dry. Once
    /// the error response is out the slot is freed and later DATA for the id
    /// is dropped uncredited; a real client has reset the stream by then, and
    /// one that ignores the response can still push a further window's worth.
    fn creditStreamWindow(self: *Connection, out: *OutBuf, id: u31, n: u32) Error!void {
        _ = self;
        if (n == 0) return;
        var wu: [4]u8 = undefined;
        std.mem.writeInt(u32, &wu, n, .big);
        try out.frame(.window_update, 0, id, &wu);
    }

    // ---------------------------------------------------------------------
    // Sending DATA
    // ---------------------------------------------------------------------

    /// The largest DATA payload that may go out on stream `idx` right now:
    /// bounded by the peer's SETTINGS_MAX_FRAME_SIZE (hyper's h2 answers a
    /// larger frame with FRAME_SIZE_ERROR and a GOAWAY, at its 16 384 default
    /// that is any response over ~16 KB), by both send windows (RFC 9113
    /// §6.9), and by the room left in `out` after `OutBuf.reserve`.
    pub fn dataBudget(self: *const Connection, out: *const OutBuf, idx: usize) usize {
        const room = out.available() -| (OutBuf.reserve + frame_header_len);
        const win: i32 = @min(self.send_window, self.streams[idx].window);
        if (win <= 0) return 0;
        return @min(room, @min(@as(usize, self.peer.max_frame_size), @as(usize, @intCast(win))));
    }

    /// Charge one DATA frame of `n` bytes to both send windows.
    pub fn chargeSend(self: *Connection, idx: usize, n: usize) void {
        self.send_window -= @intCast(n);
        self.streams[idx].window -= @intCast(n);
        std.debug.assert(self.send_window >= 0 and self.streams[idx].window >= 0);
    }
};

/// A fixed output buffer with frame-writing helpers.
///
/// §6.1: "Writev batching: coalesce all completed responses for a connection
/// into a single `writev` per epoll wakeup." Accumulating every frame into one
/// contiguous buffer is the simpler form of that: one `write` per wakeup rather
/// than one per frame, with no iovec bookkeeping.
pub const OutBuf = struct {
    buf: []u8,
    len: usize = 0,

    /// Headroom that response bodies leave free. Control frames the reader
    /// must emit while parsing (SETTINGS ack, PING ack, WINDOW_UPDATE top-ups,
    /// RST_STREAM for a refused stream, GOAWAY) and a response's trailers are
    /// all far smaller than this, so as long as bodies stop at `reserve` and
    /// the reader pauses input below `control_reserve`, `OutputFull` cannot
    /// reach a place that would have to drop a frame or tear the connection
    /// down.
    pub const reserve: usize = 1024;
    /// The most output a single inbound frame can generate: at most two
    /// control frames of 13 bytes each (a DATA frame being drained after an
    /// over-sized body credits both the connection and the stream window),
    /// or one PING ack of 17. The reader stops parsing input while less than
    /// this is free and resumes once the buffer has been flushed.
    pub const control_reserve: usize = 64;

    pub fn init(buf: []u8) OutBuf {
        return .{ .buf = buf };
    }

    pub fn reset(self: *OutBuf) void {
        self.len = 0;
    }

    pub fn written(self: *const OutBuf) []const u8 {
        return self.buf[0..self.len];
    }

    pub fn available(self: *const OutBuf) usize {
        return self.buf.len - self.len;
    }

    pub fn frame(self: *OutBuf, t: FrameType, flags: u8, stream_id: u31, payload: []const u8) Error!void {
        if (self.available() < frame_header_len + payload.len) return Error.OutputFull;
        const h = FrameHeader{
            .length = @intCast(payload.len),
            .frame_type = t,
            .flags = flags,
            .stream_id = stream_id,
        };
        h.write(self.buf[self.len..][0..frame_header_len]);
        self.len += frame_header_len;
        @memcpy(self.buf[self.len..][0..payload.len], payload);
        self.len += payload.len;
    }

    /// Reserve space for a frame header and return where the body starts, so a
    /// payload can be built in place and the length backfilled. Avoids staging
    /// a response in a scratch buffer and then copying it.
    pub fn beginFrame(self: *OutBuf, t: FrameType, flags: u8, stream_id: u31) Error!FrameCursor {
        if (self.available() < frame_header_len) return Error.OutputFull;
        const at = self.len;
        self.len += frame_header_len;
        return .{ .header_at = at, .frame_type = t, .flags = flags, .stream_id = stream_id };
    }

    pub const FrameCursor = struct {
        header_at: usize,
        frame_type: FrameType,
        flags: u8,
        stream_id: u31,
    };

    /// The writable region after a `beginFrame`.
    pub fn body(self: *OutBuf) []u8 {
        return self.buf[self.len..];
    }

    pub fn endFrame(self: *OutBuf, c: FrameCursor, body_len: usize) Error!void {
        if (self.available() < body_len) return Error.OutputFull;
        const h = FrameHeader{
            .length = @intCast(body_len),
            .frame_type = c.frame_type,
            .flags = c.flags,
            .stream_id = c.stream_id,
        };
        h.write(self.buf[c.header_at..][0..frame_header_len]);
        self.len += body_len;
    }
};

// =========================================================================
// Tests
// =========================================================================

const testing = std.testing;

fn testBuffers(alloc: std.mem.Allocator, n: usize, size: usize) ![]([]u8) {
    const bufs = try alloc.alloc([]u8, n);
    for (bufs) |*b| b.* = try alloc.alloc(u8, size);
    return bufs;
}

fn freeBuffers(alloc: std.mem.Allocator, bufs: []([]u8)) void {
    for (bufs) |b| alloc.free(b);
    alloc.free(bufs);
}

test "frame header round-trips, including the reserved bit" {
    var buf: [frame_header_len]u8 = undefined;
    const h = FrameHeader{ .length = 0x123456, .frame_type = .data, .flags = 0x5, .stream_id = 0x7fffffff };
    h.write(&buf);
    const got = FrameHeader.parse(&buf);
    try testing.expectEqual(h.length, got.length);
    try testing.expectEqual(h.frame_type, got.frame_type);
    try testing.expectEqual(h.flags, got.flags);
    try testing.expectEqual(h.stream_id, got.stream_id);

    // RFC 9113 §4.1: the reserved high bit must be ignored, not rejected.
    buf[5] |= 0x80;
    const masked = FrameHeader.parse(&buf);
    try testing.expectEqual(h.stream_id, masked.stream_id);
}

test "preface is validated and partial prefaces wait" {
    const bufs = try testBuffers(testing.allocator, 4, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);

    try testing.expectEqual(@as(?usize, null), try c.consumePreface(client_preface[0..10]));
    try testing.expectEqual(@as(?usize, client_preface.len), try c.consumePreface(client_preface));
    try testing.expect(c.preface_seen);

    // An HTTP/1.1 request must fail immediately rather than hang.
    var c2 = Connection.init(bufs);
    try testing.expectError(Error.ProtocolError, c2.consumePreface("GET / HTTP/1.1\r\n"));
}

test "initial frames raise BOTH the stream and connection windows" {
    // The bug this guards: SETTINGS_INITIAL_WINDOW_SIZE governs streams only.
    // Without the stream-0 WINDOW_UPDATE the connection window stays at 65535
    // and a 307 KB upsert still stalls (§6.1).
    const bufs = try testBuffers(testing.allocator, 4, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);

    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);
    try c.sendInitialFrames(&out);

    var pos: usize = 0;
    var saw_settings = false;
    var saw_conn_window_update = false;
    while (pos + frame_header_len <= out.len) {
        const h = FrameHeader.parse(out.buf[pos..][0..frame_header_len]);
        const payload = out.buf[pos + frame_header_len ..][0..h.length];
        switch (h.frame_type) {
            .settings => {
                saw_settings = true;
                var i: usize = 0;
                var found_window = false;
                var found_header_list = false;
                while (i + 6 <= payload.len) : (i += 6) {
                    const id = std.mem.readInt(u16, payload[i..][0..2], .big);
                    const v = std.mem.readInt(u32, payload[i + 2 ..][0..4], .big);
                    if (id == @intFromEnum(SettingId.initial_window_size)) {
                        try testing.expectEqual(Settings.default_window, v);
                        found_window = true;
                    }
                    // The limit the decoder enforces is the one the peer is told.
                    if (id == @intFromEnum(SettingId.max_header_list_size)) {
                        try testing.expectEqual(c.local.max_header_list_size, v);
                        found_header_list = true;
                    }
                }
                try testing.expect(found_window);
                try testing.expect(found_header_list);
            },
            .window_update => {
                try testing.expectEqual(@as(u31, 0), h.stream_id);
                const inc = std.mem.readInt(u32, payload[0..4], .big);
                try testing.expectEqual(Settings.default_window - 65535, inc);
                saw_conn_window_update = true;
            },
            else => {},
        }
        pos += frame_header_len + h.length;
    }
    try testing.expect(saw_settings);
    try testing.expect(saw_conn_window_update);
    try testing.expectEqual(@as(i32, @intCast(Settings.default_window)), c.recv_window);
}

test "SETTINGS is acked and a peer initial_window_size change adjusts open streams" {
    const bufs = try testBuffers(testing.allocator, 4, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    c.preface_seen = true;

    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);

    // Open a stream so there is something to adjust.
    const idx = try c.openStream(1);
    try testing.expectEqual(@as(i32, 65535), c.streams[idx].window);

    // Peer raises initial window to 1 MiB.
    var payload: [6]u8 = undefined;
    std.mem.writeInt(u16, payload[0..2], @intFromEnum(SettingId.initial_window_size), .big);
    std.mem.writeInt(u32, payload[2..6], 1024 * 1024, .big);
    var frame_buf: [frame_header_len + 6]u8 = undefined;
    (FrameHeader{ .length = 6, .frame_type = .settings, .flags = 0, .stream_id = 0 }).write(frame_buf[0..frame_header_len]);
    @memcpy(frame_buf[frame_header_len..], &payload);

    const r = try c.readFrame(&frame_buf, &out);
    try testing.expectEqual(frame_buf.len, r.consumed);
    // RFC 9113 §6.9.2: existing streams get the delta.
    try testing.expectEqual(@as(i32, 1024 * 1024), c.streams[idx].window);

    // And we acked.
    const ack = FrameHeader.parse(out.buf[0..frame_header_len]);
    try testing.expectEqual(FrameType.settings, ack.frame_type);
    try testing.expect(ack.hasFlag(Flags.ack));
}

test "PING is echoed with ACK and the payload preserved" {
    const bufs = try testBuffers(testing.allocator, 4, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [256]u8 = undefined;
    var out = OutBuf.init(&ob);

    var frame_buf: [frame_header_len + 8]u8 = undefined;
    (FrameHeader{ .length = 8, .frame_type = .ping, .flags = 0, .stream_id = 0 }).write(frame_buf[0..frame_header_len]);
    const data = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    @memcpy(frame_buf[frame_header_len..], &data);

    _ = try c.readFrame(&frame_buf, &out);
    const h = FrameHeader.parse(out.buf[0..frame_header_len]);
    try testing.expectEqual(FrameType.ping, h.frame_type);
    try testing.expect(h.hasFlag(Flags.ack));
    try testing.expectEqualSlices(u8, &data, out.buf[frame_header_len..][0..8]);
}

test "a partial frame consumes nothing and asks for more" {
    const bufs = try testBuffers(testing.allocator, 4, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [256]u8 = undefined;
    var out = OutBuf.init(&ob);

    var frame_buf: [frame_header_len + 8]u8 = undefined;
    (FrameHeader{ .length = 8, .frame_type = .ping, .flags = 0, .stream_id = 0 }).write(frame_buf[0..frame_header_len]);

    // Header only.
    var r = try c.readFrame(frame_buf[0..frame_header_len], &out);
    try testing.expectEqual(@as(usize, 0), r.consumed);
    try testing.expectEqual(Event.need_more, r.event);

    // Fewer than 9 bytes.
    r = try c.readFrame(frame_buf[0..4], &out);
    try testing.expectEqual(@as(usize, 0), r.consumed);
    try testing.expectEqual(Event.need_more, r.event);
}

test "HEADERS + DATA with END_STREAM produces a request" {
    const bufs = try testBuffers(testing.allocator, 4, 4096);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);

    // HEADERS: :path /qdrant.Points/Upsert, no END_STREAM.
    var hb: [256]u8 = undefined;
    var hn: usize = 0;
    hn += hpack.encodeInteger(hb[hn..], 6, 0x40, 4); // literal w/ indexing, name = :path
    const path = "/qdrant.Points/Upsert";
    hn += hpack.encodeInteger(hb[hn..], 7, 0x00, path.len);
    @memcpy(hb[hn..][0..path.len], path);
    hn += path.len;

    var frame_buf: [512]u8 = undefined;
    (FrameHeader{ .length = @intCast(hn), .frame_type = .headers, .flags = Flags.end_headers, .stream_id = 1 })
        .write(frame_buf[0..frame_header_len]);
    @memcpy(frame_buf[frame_header_len..][0..hn], hb[0..hn]);

    var r = try c.readFrame(frame_buf[0 .. frame_header_len + hn], &out);
    try testing.expectEqual(Event.progress, r.event);

    // DATA with END_STREAM.
    const body = "hello grpc body";
    (FrameHeader{ .length = body.len, .frame_type = .data, .flags = Flags.end_stream, .stream_id = 1 })
        .write(frame_buf[0..frame_header_len]);
    @memcpy(frame_buf[frame_header_len..][0..body.len], body);

    r = try c.readFrame(frame_buf[0 .. frame_header_len + body.len], &out);
    switch (r.event) {
        .request => |idx| {
            try testing.expectEqualStrings(path, c.streams[idx].path());
            try testing.expectEqualStrings(body, c.streams[idx].bodyBytes());
            try testing.expectEqual(StreamState.half_closed_remote, c.streams[idx].state);
        },
        else => return error.ExpectedRequest,
    }
}

test "header block split across CONTINUATION frames reassembles" {
    const bufs = try testBuffers(testing.allocator, 4, 4096);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);

    var hb: [256]u8 = undefined;
    var hn: usize = 0;
    hn += hpack.encodeInteger(hb[hn..], 6, 0x40, 4);
    const path = "/qdrant.Points/QueryBatch";
    hn += hpack.encodeInteger(hb[hn..], 7, 0x00, path.len);
    @memcpy(hb[hn..][0..path.len], path);
    hn += path.len;

    const split = hn / 2;
    var frame_buf: [512]u8 = undefined;

    // HEADERS without END_HEADERS.
    (FrameHeader{ .length = @intCast(split), .frame_type = .headers, .flags = 0, .stream_id = 1 })
        .write(frame_buf[0..frame_header_len]);
    @memcpy(frame_buf[frame_header_len..][0..split], hb[0..split]);
    var r = try c.readFrame(frame_buf[0 .. frame_header_len + split], &out);
    try testing.expectEqual(Event.progress, r.event);
    try testing.expect(c.expecting_continuation);

    // An interleaved DATA frame here is a protocol error.
    var bad: [frame_header_len]u8 = undefined;
    (FrameHeader{ .length = 0, .frame_type = .data, .flags = 0, .stream_id = 1 }).write(&bad);
    try testing.expectError(Error.ProtocolError, c.readFrame(&bad, &out));

    // CONTINUATION with END_HEADERS + the stream ends via a later DATA.
    const rest = hn - split;
    (FrameHeader{ .length = @intCast(rest), .frame_type = .continuation, .flags = Flags.end_headers, .stream_id = 1 })
        .write(frame_buf[0..frame_header_len]);
    @memcpy(frame_buf[frame_header_len..][0..rest], hb[split..hn]);
    r = try c.readFrame(frame_buf[0 .. frame_header_len + rest], &out);
    try testing.expectEqual(Event.progress, r.event);
    try testing.expect(!c.expecting_continuation);

    const idx = c.findStream(1).?;
    try testing.expectEqualStrings(path, c.streams[idx].path());
}

test "connection receive window is topped up before it drains" {
    const bufs = try testBuffers(testing.allocator, 4, 8 * 1024 * 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [64 * 1024]u8 = undefined;
    var out = OutBuf.init(&ob);
    try c.sendInitialFrames(&out);
    out.reset();

    _ = try c.openStream(1);

    // Send just over half the window as DATA and confirm a WINDOW_UPDATE comes
    // back, waiting for exhaustion would serialise the upload (§6.1).
    const chunk = 512 * 1024;
    const payload = try testing.allocator.alloc(u8, chunk);
    defer testing.allocator.free(payload);
    @memset(payload, 0xab);

    const frame_buf = try testing.allocator.alloc(u8, frame_header_len + chunk);
    defer testing.allocator.free(frame_buf);

    var sent: usize = 0;
    var saw_update = false;
    while (sent < Settings.default_window / 2 + chunk) : (sent += chunk) {
        (FrameHeader{ .length = chunk, .frame_type = .data, .flags = 0, .stream_id = 1 })
            .write(frame_buf[0..frame_header_len]);
        @memcpy(frame_buf[frame_header_len..], payload);
        _ = try c.readFrame(frame_buf, &out);
        if (out.len > 0) {
            const h = FrameHeader.parse(out.buf[0..frame_header_len]);
            if (h.frame_type == .window_update and h.stream_id == 0) saw_update = true;
            out.reset();
        }
    }
    try testing.expect(saw_update);
    // And the window never went negative.
    try testing.expect(c.recv_window > 0);
}

test "exceeding the stream pool yields REFUSED_STREAM, not a stall" {
    const bufs = try testBuffers(testing.allocator, 2, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    // Shrink the pool to the two buffers we allocated by marking the rest used.
    for (c.streams[2..]) |*s| s.state = .open;

    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);

    var hb: [64]u8 = undefined;
    const hn = hpack.encodeInteger(&hb, 7, 0x80, 2); // :method GET
    var frame_buf: [128]u8 = undefined;

    // Two succeed.
    for (1..3) |i| {
        (FrameHeader{ .length = @intCast(hn), .frame_type = .headers, .flags = Flags.end_headers, .stream_id = @intCast(i * 2 - 1) })
            .write(frame_buf[0..frame_header_len]);
        @memcpy(frame_buf[frame_header_len..][0..hn], hb[0..hn]);
        _ = try c.readFrame(frame_buf[0 .. frame_header_len + hn], &out);
    }
    out.reset();

    // The third is refused.
    (FrameHeader{ .length = @intCast(hn), .frame_type = .headers, .flags = Flags.end_headers, .stream_id = 5 })
        .write(frame_buf[0..frame_header_len]);
    @memcpy(frame_buf[frame_header_len..][0..hn], hb[0..hn]);
    _ = try c.readFrame(frame_buf[0 .. frame_header_len + hn], &out);

    const h = FrameHeader.parse(out.buf[0..frame_header_len]);
    try testing.expectEqual(FrameType.rst_stream, h.frame_type);
    try testing.expectEqual(@as(u31, 5), h.stream_id);
    const code = std.mem.readInt(u32, out.buf[frame_header_len..][0..4], .big);
    try testing.expectEqual(@intFromEnum(ErrorCode.refused_stream), code);
}

test "a body larger than the stream buffer fails the stream, not the connection" {
    // An over-sized body used to return `Error.RequestTooLarge`, which made the
    // caller send GOAWAY and tear down the whole connection, taking every
    // other in-flight stream with it and telling the client only "transport
    // error". Uploading d=1536 vectors 512 to a batch (3.1 MB) hit it.
    const bufs = try testBuffers(testing.allocator, 2, 128);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    for (c.streams[2..]) |*s| s.state = .open;
    const idx = try c.openStream(1);

    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);
    try c.sendInitialFrames(&out);
    out.reset();

    // The first frame that overflows the 128-byte body buffer reports the
    // failure *immediately*, it does not wait for END_STREAM: a client
    // uploading more than the stream window would otherwise never learn why
    // it stalled. The stream window it consumed is credited straight back so
    // the upload can drain.
    var frame_buf: [frame_header_len + 256]u8 = undefined;
    (FrameHeader{ .length = 256, .frame_type = .data, .flags = 0, .stream_id = 1 })
        .write(frame_buf[0..frame_header_len]);
    @memset(frame_buf[frame_header_len..], 0);
    const r1 = try c.readFrame(&frame_buf, &out);
    try testing.expectEqual(@as(usize, idx), r1.event.request_too_large);
    try testing.expect(c.streams[idx].too_large);
    try testing.expect(!c.streams[idx].dispatched);
    {
        const wu = FrameHeader.parse(out.buf[0..frame_header_len]);
        try testing.expectEqual(FrameType.window_update, wu.frame_type);
        try testing.expectEqual(@as(u31, 1), wu.stream_id);
        try testing.expectEqual(@as(u32, 256), std.mem.readInt(u32, out.buf[frame_header_len..][0..4], .big));
        out.reset();
    }

    // Further DATA is drained, credited, and reported exactly once: a second
    // `.request_too_large` would make the server answer the stream twice.
    const r_more = try c.readFrame(&frame_buf, &out);
    try testing.expectEqual(Event.progress, r_more.event);
    try testing.expectEqual(FrameType.window_update, FrameHeader.parse(out.buf[0..frame_header_len]).frame_type);
    out.reset();

    // END_STREAM is absorbed too.
    var tail: [frame_header_len]u8 = undefined;
    (FrameHeader{ .length = 0, .frame_type = .data, .flags = Flags.end_stream, .stream_id = 1 })
        .write(&tail);
    const r2 = try c.readFrame(&tail, &out);
    try testing.expectEqual(Event.progress, r2.event);
    try testing.expectEqual(StreamState.half_closed_remote, c.streams[idx].state);

    // The slot is reusable afterwards: `too_large` must not persist, or every
    // later request landing on this slot fails too.
    c.closeStream(idx);
    const again = try c.openStream(3);
    try testing.expect(!c.streams[again].too_large);
}

test "oversized frames and bad frame sizes are refused" {
    const bufs = try testBuffers(testing.allocator, 2, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [256]u8 = undefined;
    var out = OutBuf.init(&ob);

    // Longer than our advertised max_frame_size.
    var h: [frame_header_len]u8 = undefined;
    (FrameHeader{ .length = Settings.default_max_frame + 1, .frame_type = .data, .flags = 0, .stream_id = 1 }).write(&h);
    try testing.expectError(Error.FrameSizeError, c.readFrame(&h, &out));

    // RST_STREAM must be exactly 4 bytes.
    var rst: [frame_header_len + 3]u8 = undefined;
    (FrameHeader{ .length = 3, .frame_type = .rst_stream, .flags = 0, .stream_id = 1 }).write(rst[0..frame_header_len]);
    try testing.expectError(Error.FrameSizeError, c.readFrame(&rst, &out));

    // SETTINGS length must be a multiple of 6.
    var st: [frame_header_len + 5]u8 = undefined;
    (FrameHeader{ .length = 5, .frame_type = .settings, .flags = 0, .stream_id = 0 }).write(st[0..frame_header_len]);
    try testing.expectError(Error.FrameSizeError, c.readFrame(&st, &out));
}

test "PUSH_PROMISE from a client is a protocol error" {
    const bufs = try testBuffers(testing.allocator, 2, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [256]u8 = undefined;
    var out = OutBuf.init(&ob);

    var h: [frame_header_len]u8 = undefined;
    (FrameHeader{ .length = 0, .frame_type = .push_promise, .flags = 0, .stream_id = 1 }).write(&h);
    try testing.expectError(Error.ProtocolError, c.readFrame(&h, &out));
}

test "unknown frame types are ignored, per RFC 9113 §4.1" {
    const bufs = try testBuffers(testing.allocator, 2, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [256]u8 = undefined;
    var out = OutBuf.init(&ob);

    var f: [frame_header_len + 4]u8 = undefined;
    (FrameHeader{ .length = 4, .frame_type = @enumFromInt(0xfa), .flags = 0, .stream_id = 0 }).write(f[0..frame_header_len]);
    const r = try c.readFrame(&f, &out);
    try testing.expectEqual(f.len, r.consumed);
    try testing.expectEqual(Event.progress, r.event);
}

test "padded DATA strips padding and still charges it to the window" {
    const bufs = try testBuffers(testing.allocator, 2, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    for (c.streams[2..]) |*s| s.state = .open;
    _ = try c.openStream(1);

    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);
    // Send the initial frames first so `recv_window` is already at target;
    // otherwise the top-up fires on the first DATA frame and masks the
    // accounting this test is checking.
    try c.sendInitialFrames(&out);
    out.reset();
    const before = c.recv_window;

    // pad_len=3, body "hi", 3 pad bytes -> payload length 6.
    var f: [frame_header_len + 6]u8 = undefined;
    (FrameHeader{ .length = 6, .frame_type = .data, .flags = Flags.padded | Flags.end_stream, .stream_id = 1 })
        .write(f[0..frame_header_len]);
    f[frame_header_len] = 3;
    f[frame_header_len + 1] = 'h';
    f[frame_header_len + 2] = 'i';
    @memset(f[frame_header_len + 3 ..], 0);

    const r = try c.readFrame(&f, &out);
    switch (r.event) {
        .request => |idx| try testing.expectEqualStrings("hi", c.streams[idx].bodyBytes()),
        else => return error.ExpectedRequest,
    }
    // The full padded length is charged, not just the 2 data bytes.
    try testing.expectEqual(before - 6, c.recv_window);
}

test "window update overflow is a flow-control error" {
    const bufs = try testBuffers(testing.allocator, 2, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [256]u8 = undefined;
    var out = OutBuf.init(&ob);

    var f: [frame_header_len + 4]u8 = undefined;
    (FrameHeader{ .length = 4, .frame_type = .window_update, .flags = 0, .stream_id = 0 }).write(f[0..frame_header_len]);
    std.mem.writeInt(u32, f[frame_header_len..][0..4], 0x7fffffff, .big);
    // Window is already 65535, so +2^31-1 overflows.
    try testing.expectError(Error.FlowControlError, c.readFrame(&f, &out));

    // A zero increment is a protocol error.
    std.mem.writeInt(u32, f[frame_header_len..][0..4], 0, .big);
    try testing.expectError(Error.ProtocolError, c.readFrame(&f, &out));
}

test "OutBuf refuses to overflow and beginFrame backfills the length" {
    var small: [16]u8 = undefined;
    var out = OutBuf.init(&small);
    try testing.expectError(Error.OutputFull, out.frame(.data, 0, 1, &[_]u8{0} ** 32));

    var big: [256]u8 = undefined;
    var out2 = OutBuf.init(&big);
    const c = try out2.beginFrame(.data, Flags.end_stream, 3);
    const b = out2.body();
    @memcpy(b[0..5], "hello");
    try out2.endFrame(c, 5);

    const h = FrameHeader.parse(out2.buf[0..frame_header_len]);
    try testing.expectEqual(@as(u32, 5), h.length);
    try testing.expectEqual(@as(u31, 3), h.stream_id);
    try testing.expect(h.hasFlag(Flags.end_stream));
    try testing.expectEqualStrings("hello", out2.buf[frame_header_len..][0..5]);
    try testing.expectEqual(@as(usize, frame_header_len + 5), out2.len);
}

test "advertised max_frame_size must fit the read buffer, or a legal frame stalls forever" {
    // A frame larger than the connection's read buffer can never be assembled:
    // the read loop fills the buffer, `readFrame` returns `need_more`, and
    // compaction has nothing to remove. The connection then sits idle with
    // bytes pending and no way to make progress, a silent hang from a
    // perfectly legal client.
    //
    // The server therefore lowers `local.max_frame_size` to what its read
    // buffer can hold. This test pins the invariant the two settings must
    // satisfy, and the RFC floor that bounds it.
    // The server's own clamp, not a restatement of its arithmetic: deleting
    // the clamp used to leave this test green.
    const server = @import("server.zig");
    const read_buffer: usize = 256 * 1024;
    const usable = read_buffer - frame_header_len;
    const advertised = server.advertisedMaxFrame(read_buffer);

    try testing.expect(advertised <= usable);
    // RFC 9113 §6.5.2 floors SETTINGS_MAX_FRAME_SIZE at 16384.
    try testing.expect(advertised >= 16384);
    try testing.expectEqual(@as(u32, 16384), server.advertisedMaxFrame(16384 + frame_header_len));

    // The default protocol maximum does *not* fit the default read buffer,
    // which is exactly the mismatch that made this necessary.
    try testing.expect(Settings.default_max_frame > usable);
}

test "DATA on a half-closed stream is refused, not dispatched twice" {
    // Without the state check this appended to a body already handed to a
    // worker and returned a *second* `.request` for the same slot: two workers
    // writing one response buffer, two responses on one stream id.
    const bufs = try testBuffers(testing.allocator, 4, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);
    try c.sendInitialFrames(&out);
    out.reset();

    _ = try c.openStream(1);

    var f: [frame_header_len + 4]u8 = undefined;
    (FrameHeader{ .length = 4, .frame_type = .data, .flags = Flags.end_stream, .stream_id = 1 })
        .write(f[0..frame_header_len]);
    @memcpy(f[frame_header_len..], "abcd");

    const first = try c.readFrame(&f, &out);
    switch (first.event) {
        .request => {},
        else => return error.ExpectedRequest,
    }
    // The stream is now half-closed; a second DATA is a *stream* error (RFC
    // 9113 §5.1): RST_STREAM(STREAM_CLOSED) on stream 1, the connection
    // survives, and the dispatched slot is parked as `.closed` exactly as a
    // peer RST would leave it, so the worker's response is dropped. It used
    // to be a GOAWAY.
    const idx = first.event.request;
    const second = try c.readFrame(&f, &out);
    try testing.expectEqual(@as(usize, idx), second.event.reset);
    try testing.expectEqual(StreamState.closed, c.streams[idx].state);
    try testing.expectEqualSlices(u8, &expectRst(1, .stream_closed), out.written()[0 .. frame_header_len + 4]);
    try testing.expect(!c.goaway_sent);

    // And an unrelated stream still works.
    out.reset();
    var fb: [512]u8 = undefined;
    const r3 = try c.readFrame(testHeadersFrame(&fb, 3, Flags.end_headers | Flags.end_stream, "/b"), &out);
    try testing.expect(r3.event.request != idx);
}

/// The wire bytes of RST_STREAM(`code`) on `sid`.
fn expectRst(sid: u31, code: ErrorCode) [frame_header_len + 4]u8 {
    var b: [frame_header_len + 4]u8 = undefined;
    (FrameHeader{ .length = 4, .frame_type = .rst_stream, .flags = 0, .stream_id = sid }).write(b[0..frame_header_len]);
    std.mem.writeInt(u32, b[frame_header_len..][0..4], @intFromEnum(code), .big);
    return b;
}

test "END_STREAM survives a header block split across CONTINUATION" {
    // A bodyless request whose header block exceeds one frame carries
    // END_STREAM on the HEADERS and END_HEADERS on the CONTINUATION. Dropping
    // the flag left the stream `.open` forever, holding a slot while the client
    // waited for a response that could never come.
    const bufs = try testBuffers(testing.allocator, 4, 4096);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);

    var hb: [256]u8 = undefined;
    var hn: usize = 0;
    hn += hpack.encodeInteger(hb[hn..], 6, 0x40, 4); // literal w/ indexing:path
    const path = "/qdrant.Qdrant/HealthCheck";
    hn += hpack.encodeInteger(hb[hn..], 7, 0x00, path.len);
    @memcpy(hb[hn..][0..path.len], path);
    hn += path.len;

    const split = hn / 2;
    var frame_buf: [512]u8 = undefined;

    // HEADERS with END_STREAM but *not* END_HEADERS.
    (FrameHeader{ .length = @intCast(split), .frame_type = .headers, .flags = Flags.end_stream, .stream_id = 1 })
        .write(frame_buf[0..frame_header_len]);
    @memcpy(frame_buf[frame_header_len..][0..split], hb[0..split]);
    var r = try c.readFrame(frame_buf[0 .. frame_header_len + split], &out);
    try testing.expectEqual(Event.progress, r.event);

    // CONTINUATION with END_HEADERS completes the block, and the request must
    // be dispatched because the earlier END_STREAM applies.
    const rest = hn - split;
    (FrameHeader{ .length = @intCast(rest), .frame_type = .continuation, .flags = Flags.end_headers, .stream_id = 1 })
        .write(frame_buf[0..frame_header_len]);
    @memcpy(frame_buf[frame_header_len..][0..rest], hb[split..hn]);
    r = try c.readFrame(frame_buf[0 .. frame_header_len + rest], &out);
    switch (r.event) {
        .request => |idx| {
            try testing.expectEqualStrings(path, c.streams[idx].path());
            try testing.expectEqual(StreamState.half_closed_remote, c.streams[idx].state);
        },
        else => return error.ExpectedRequest,
    }
}

test "a repeated stream id reuses its slot rather than leaking a new one" {
    // The ordinary gRPC shape ends with a trailing HEADERS on the same stream.
    // Allocating a second slot for it leaks one per request.
    const bufs = try testBuffers(testing.allocator, 4, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);

    const a = try c.openStream(7);
    const b = try c.openStream(7);
    try testing.expectEqual(a, b);

    var live: usize = 0;
    for (c.streams[0..c.usable_streams]) |s| {
        if (s.state != .idle) live += 1;
    }
    try testing.expectEqual(@as(usize, 1), live);
}

/// A HEADERS frame carrying `:path` as a literal with incremental indexing,
/// which inserts the path into the peer's dynamic table at index 62.
fn testHeadersFrame(buf: []u8, stream_id: u31, flags: u8, path: []const u8) []const u8 {
    var hb: [256]u8 = undefined;
    var hn: usize = 0;
    hn += hpack.encodeInteger(hb[hn..], 6, 0x40, 4); // literal w/ indexing, name = :path
    hn += hpack.encodeInteger(hb[hn..], 7, 0x00, @intCast(path.len));
    @memcpy(hb[hn..][0..path.len], path);
    hn += path.len;
    (FrameHeader{ .length = @intCast(hn), .frame_type = .headers, .flags = flags, .stream_id = stream_id })
        .write(buf[0..frame_header_len]);
    @memcpy(buf[frame_header_len..][0..hn], hb[0..hn]);
    return buf[0 .. frame_header_len + hn];
}

fn testRstFrame(buf: *[frame_header_len + 4]u8, stream_id: u31, code: ErrorCode) []const u8 {
    (FrameHeader{ .length = 4, .frame_type = .rst_stream, .flags = 0, .stream_id = stream_id })
        .write(buf[0..frame_header_len]);
    std.mem.writeInt(u32, buf[frame_header_len..][0..4], @intFromEnum(code), .big);
    return buf;
}

test "RST_STREAM on a dispatched stream holds the slot until the completion frees it" {
    // The slot used to go `.idle` on RST while a worker still owned the
    // request. The next HEADERS reused it, DATA landed in the buffer under
    // the worker, and the old completion's `closeStream` then reset the *new*
    // stream, whose response was never sent.
    const bufs = try testBuffers(testing.allocator, 2, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);

    var fb: [512]u8 = undefined;
    const r1 = try c.readFrame(testHeadersFrame(&fb, 1, Flags.end_headers | Flags.end_stream, "/a"), &out);
    const owned = r1.event.request;
    try testing.expect(c.streams[owned].dispatched);

    // The peer resets it while the worker runs.
    var rb: [frame_header_len + 4]u8 = undefined;
    const r2 = try c.readFrame(testRstFrame(&rb, 1, .cancel), &out);
    try testing.expectEqual(@as(usize, owned), r2.event.reset);
    try testing.expectEqual(StreamState.closed, c.streams[owned].state);
    try testing.expectEqual(@as(u31, 1), c.streams[owned].id);

    // A new stream must land in the *other* slot; the pool is two wide, so a
    // third distinct stream is refused rather than handed the held slot.
    const r3 = try c.readFrame(testHeadersFrame(&fb, 3, Flags.end_headers | Flags.end_stream, "/b"), &out);
    try testing.expect(r3.event.request != owned);
    out.reset();
    const r4 = try c.readFrame(testHeadersFrame(&fb, 5, Flags.end_headers | Flags.end_stream, "/c"), &out);
    try testing.expectEqual(Event.progress, r4.event);
    try testing.expectEqual(FrameType.rst_stream, FrameHeader.parse(out.buf[0..frame_header_len]).frame_type);
    try testing.expectEqual(StreamState.closed, c.streams[owned].state);

    // Frames for the reset id are ignored, not applied to the held slot.
    var df: [frame_header_len + 2]u8 = undefined;
    (FrameHeader{ .length = 2, .frame_type = .data, .flags = 0, .stream_id = 1 }).write(df[0..frame_header_len]);
    @memcpy(df[frame_header_len..], "xx");
    try testing.expectEqual(Event.progress, (try c.readFrame(&df, &out)).event);
    try testing.expectEqual(@as(usize, 0), c.streams[owned].body_len);

    // The completion frees it, and only then is it reusable.
    c.closeStream(owned);
    out.reset();
    const r5 = try c.readFrame(testHeadersFrame(&fb, 7, Flags.end_headers | Flags.end_stream, "/d"), &out);
    try testing.expectEqual(@as(usize, owned), r5.event.request);
    try testing.expectEqualStrings("/d", c.streams[owned].path());
}

test "RST_STREAM on an undispatched stream frees it and still reports the reset" {
    // No worker owns an `.open` stream, so the slot is freed at once; the
    // caller is told anyway so any error response it had queued for the
    // slot (an over-sized body, say) is dropped.
    const bufs = try testBuffers(testing.allocator, 2, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);

    var fb: [512]u8 = undefined;
    _ = try c.readFrame(testHeadersFrame(&fb, 1, Flags.end_headers, "/a"), &out);
    const idx = c.findStream(1).?;
    try testing.expect(!c.streams[idx].dispatched);

    var rb: [frame_header_len + 4]u8 = undefined;
    const r = try c.readFrame(testRstFrame(&rb, 1, .cancel), &out);
    try testing.expectEqual(@as(usize, idx), r.event.reset);
    try testing.expectEqual(StreamState.idle, c.streams[idx].state);

    // RST_STREAM on stream 0 is a connection error, and on an unknown stream
    // it is nothing at all.
    try testing.expectError(Error.ProtocolError, c.readFrame(testRstFrame(&rb, 0, .cancel), &out));
    try testing.expectEqual(Event.progress, (try c.readFrame(testRstFrame(&rb, 99, .cancel), &out)).event);
}

test "a refused HEADERS still decodes its block, so the dynamic table stays in sync" {
    // RFC 9113 §4.3: every header block must be decoded, refused or not,
    // because it may insert into the HPACK dynamic table. Skipping the block
    // of a REFUSED_STREAM left the table one entry short, and the client's
    // very next request, indexing the entry it believes it added, failed
    // with COMPRESSION_ERROR, a connection error, for a stream that was
    // legitimately refused.
    const bufs = try testBuffers(testing.allocator, 1, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);
    var fb: [512]u8 = undefined;

    // Fills the single slot. `:path /first` becomes dynamic index 62.
    const r1 = try c.readFrame(testHeadersFrame(&fb, 1, Flags.end_headers | Flags.end_stream, "/first"), &out);
    const held = r1.event.request;

    // Refused, but its `:path /refused` must still become index 62 (pushing
    // /first to 63). Split across a CONTINUATION to cover that path too.
    const whole = testHeadersFrame(&fb, 3, Flags.end_stream, "/refused");
    const split = frame_header_len + 5;
    var head: [64]u8 = undefined;
    @memcpy(head[0..split], whole[0..split]);
    (FrameHeader{ .length = 5, .frame_type = .headers, .flags = Flags.end_stream, .stream_id = 3 }).write(head[0..frame_header_len]);
    out.reset();
    var r = try c.readFrame(head[0..split], &out);
    try testing.expectEqual(Event.progress, r.event);
    try testing.expect(c.expecting_continuation);
    try testing.expectEqual(@as(usize, 0), out.len); // no RST until the block is complete

    var cont: [64]u8 = undefined;
    const rest = whole.len - split;
    (FrameHeader{ .length = @intCast(rest), .frame_type = .continuation, .flags = Flags.end_headers, .stream_id = 3 }).write(cont[0..frame_header_len]);
    @memcpy(cont[frame_header_len..][0..rest], whole[split..]);
    r = try c.readFrame(cont[0 .. frame_header_len + rest], &out);
    try testing.expectEqual(Event.progress, r.event);
    try testing.expect(!c.expecting_continuation);
    const rst = FrameHeader.parse(out.buf[0..frame_header_len]);
    try testing.expectEqual(FrameType.rst_stream, rst.frame_type);
    try testing.expectEqual(@as(u31, 3), rst.stream_id);
    try testing.expectEqual(@intFromEnum(ErrorCode.refused_stream), std.mem.readInt(u32, out.buf[frame_header_len..][0..4], .big));

    // Free the slot, then a request naming the refused block's entry by
    // index (62 = most recent insertion) must decode to it.
    c.closeStream(held);
    var hb: [8]u8 = undefined;
    const hn = hpack.encodeInteger(&hb, 7, 0x80, 62);
    var f: [64]u8 = undefined;
    (FrameHeader{ .length = @intCast(hn), .frame_type = .headers, .flags = Flags.end_headers | Flags.end_stream, .stream_id = 5 }).write(f[0..frame_header_len]);
    @memcpy(f[frame_header_len..][0..hn], hb[0..hn]);
    r = try c.readFrame(f[0 .. frame_header_len + hn], &out);
    try testing.expectEqualStrings("/refused", c.streams[r.event.request].path());
}

test "HEADERS with END_STREAM on a half-closed stream is refused, not dispatched twice" {
    // The DATA path already guarded this; a trailing HEADERS after END_STREAM
    // re-ran `finishHeaders` on a slot a worker owned and produced a second
    // `.request` for it.
    const bufs = try testBuffers(testing.allocator, 2, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);
    var fb: [512]u8 = undefined;

    const first = try c.readFrame(testHeadersFrame(&fb, 1, Flags.end_headers | Flags.end_stream, "/a"), &out);
    const idx = first.event.request;
    try testing.expectEqual(StreamState.half_closed_remote, c.streams[idx].state);
    // A stream error, not a connection error: RST_STREAM(STREAM_CLOSED) and
    // the slot is parked `.closed` for the worker's completion to free.
    out.reset();
    const second = try c.readFrame(testHeadersFrame(&fb, 1, Flags.end_headers | Flags.end_stream, "/b"), &out);
    try testing.expectEqual(@as(usize, idx), second.event.reset);
    try testing.expectEqual(StreamState.closed, c.streams[idx].state);
    try testing.expectEqualSlices(u8, &expectRst(1, .stream_closed), out.written()[0 .. frame_header_len + 4]);
    // Also without END_STREAM: nothing may follow END_STREAM from the peer.
    // The slot is now `.closed`, so this is a HEADERS on a closed id: the
    // same RST, and no second reset event for a slot already taken away.
    out.reset();
    const third = try c.readFrame(testHeadersFrame(&fb, 1, Flags.end_headers, "/c"), &out);
    try testing.expectEqual(Event.progress, third.event);
    try testing.expectEqualSlices(u8, &expectRst(1, .stream_closed), out.written()[0 .. frame_header_len + 4]);
    // The blocks were still decoded: `/c` is the newest dynamic entry (62),
    // so a request naming it by index gets it. Skipping the block would have
    // desynchronised the table.
    c.closeStream(idx);
    var hb: [8]u8 = undefined;
    const hn = hpack.encodeInteger(&hb, 7, 0x80, 62);
    var f: [64]u8 = undefined;
    (FrameHeader{ .length = @intCast(hn), .frame_type = .headers, .flags = Flags.end_headers | Flags.end_stream, .stream_id = 3 }).write(f[0..frame_header_len]);
    @memcpy(f[frame_header_len..][0..hn], hb[0..hn]);
    const r = try c.readFrame(f[0 .. frame_header_len + hn], &out);
    try testing.expectEqualStrings("/c", c.streams[r.event.request].path());
    try testing.expect(!c.goaway_sent);
}

test "request trailers on a body that overflowed do not dispatch the truncated body" {
    // After `.request_too_large` the stream stays `.open` while the rest of
    // the body drains. A trailing HEADERS+END_STREAM used to reach
    // `finishHeaders` and dispatch the partial body as a request, a second
    // response for a stream already answered RESOURCE_EXHAUSTED.
    const bufs = try testBuffers(testing.allocator, 2, 128);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);
    try c.sendInitialFrames(&out);
    out.reset();
    var fb: [512]u8 = undefined;

    _ = try c.readFrame(testHeadersFrame(&fb, 1, Flags.end_headers, "/a"), &out);
    const idx = c.findStream(1).?;

    var df: [frame_header_len + 100]u8 = undefined;
    (FrameHeader{ .length = 100, .frame_type = .data, .flags = 0, .stream_id = 1 }).write(df[0..frame_header_len]);
    @memset(df[frame_header_len..], 1);
    try testing.expectEqual(Event.progress, (try c.readFrame(&df, &out)).event); // fits
    try testing.expectEqual(@as(usize, idx), (try c.readFrame(&df, &out)).event.request_too_large); // overflows
    try testing.expectEqual(StreamState.open, c.streams[idx].state);

    const r = try c.readFrame(testHeadersFrame(&fb, 1, Flags.end_headers | Flags.end_stream, "/a"), &out);
    try testing.expectEqual(Event.progress, r.event);
    try testing.expect(!c.streams[idx].dispatched);
    try testing.expectEqual(StreamState.half_closed_remote, c.streams[idx].state);
}

test "stream ids are validated: PING and GOAWAY on 0, PRIORITY off 0, HEADERS odd and increasing" {
    const bufs = try testBuffers(testing.allocator, 4, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var ob: [4096]u8 = undefined;
    var fb: [512]u8 = undefined;

    // PING must be on stream 0 (RFC 9113 §6.7).
    {
        var c = Connection.init(bufs);
        var out = OutBuf.init(&ob);
        var f: [frame_header_len + 8]u8 = undefined;
        (FrameHeader{ .length = 8, .frame_type = .ping, .flags = 0, .stream_id = 1 }).write(f[0..frame_header_len]);
        @memset(f[frame_header_len..], 0);
        try testing.expectError(Error.ProtocolError, c.readFrame(&f, &out));
    }
    // GOAWAY must be on stream 0 and carry at least 8 bytes (§6.8).
    {
        var c = Connection.init(bufs);
        var out = OutBuf.init(&ob);
        var f: [frame_header_len + 8]u8 = undefined;
        (FrameHeader{ .length = 8, .frame_type = .goaway, .flags = 0, .stream_id = 1 }).write(f[0..frame_header_len]);
        @memset(f[frame_header_len..], 0);
        try testing.expectError(Error.ProtocolError, c.readFrame(&f, &out));
        (FrameHeader{ .length = 4, .frame_type = .goaway, .flags = 0, .stream_id = 0 }).write(f[0..frame_header_len]);
        try testing.expectError(Error.FrameSizeError, c.readFrame(f[0 .. frame_header_len + 4], &out));
        (FrameHeader{ .length = 8, .frame_type = .goaway, .flags = 0, .stream_id = 0 }).write(f[0..frame_header_len]);
        try testing.expectEqual(Event.goaway, (try c.readFrame(&f, &out)).event);
        try testing.expect(c.goaway_received);
    }
    // PRIORITY must not be on stream 0 (§6.3).
    {
        var c = Connection.init(bufs);
        var out = OutBuf.init(&ob);
        var f: [frame_header_len + 5]u8 = undefined;
        (FrameHeader{ .length = 5, .frame_type = .priority, .flags = 0, .stream_id = 0 }).write(f[0..frame_header_len]);
        @memset(f[frame_header_len..], 0);
        try testing.expectError(Error.ProtocolError, c.readFrame(&f, &out));
        (FrameHeader{ .length = 5, .frame_type = .priority, .flags = 0, .stream_id = 1 }).write(f[0..frame_header_len]);
        try testing.expectEqual(Event.progress, (try c.readFrame(&f, &out)).event);
    }
    // A client may only open odd-numbered streams (§5.1.1).
    {
        var c = Connection.init(bufs);
        var out = OutBuf.init(&ob);
        try testing.expectError(Error.ProtocolError, c.readFrame(testHeadersFrame(&fb, 2, Flags.end_headers, "/a"), &out));
    }
    // A new stream at or below the highest id seen is a closed stream (§5.1.1
    // closes lower idle ids implicitly): STREAM_CLOSED on that stream, block
    // still decoded, connection intact.
    {
        var c = Connection.init(bufs);
        var out = OutBuf.init(&ob);
        _ = try c.readFrame(testHeadersFrame(&fb, 5, Flags.end_headers | Flags.end_stream, "/a"), &out);
        try testing.expectEqual(@as(u31, 5), c.last_stream_id);
        out.reset();
        const r = try c.readFrame(testHeadersFrame(&fb, 3, Flags.end_headers | Flags.end_stream, "/b"), &out);
        try testing.expectEqual(Event.progress, r.event);
        try testing.expectEqualSlices(u8, &expectRst(3, .stream_closed), out.written()[0 .. frame_header_len + 4]);
        try testing.expectEqual(@as(?usize, null), c.findStream(3));
        // The refused-then-reused shape: 7 is refused only if the pool is
        // full, so here it simply opens; the id was recorded either way.
        out.reset();
        _ = try c.readFrame(testHeadersFrame(&fb, 7, Flags.end_headers | Flags.end_stream, "/c"), &out);
        try testing.expectEqual(@as(u31, 7), c.last_stream_id);
    }
    // A refused stream still advances the high-water mark, so its trailing
    // DATA is a closed-stream drop rather than an idle-stream error.
    {
        const one = try testBuffers(testing.allocator, 1, 1024);
        defer freeBuffers(testing.allocator, one);
        var c = Connection.init(one);
        var out = OutBuf.init(&ob);
        _ = try c.readFrame(testHeadersFrame(&fb, 1, Flags.end_headers | Flags.end_stream, "/a"), &out);
        _ = try c.readFrame(testHeadersFrame(&fb, 3, Flags.end_headers, "/b"), &out); // refused
        try testing.expectEqual(@as(u31, 3), c.last_stream_id);
        var df: [frame_header_len + 2]u8 = undefined;
        (FrameHeader{ .length = 2, .frame_type = .data, .flags = Flags.end_stream, .stream_id = 3 }).write(df[0..frame_header_len]);
        @memcpy(df[frame_header_len..], "xx");
        try testing.expectEqual(Event.progress, (try c.readFrame(&df, &out)).event);
    }
}

test "DATA on a stream no HEADERS ever opened is a protocol error, DATA on a closed one is dropped" {
    const bufs = try testBuffers(testing.allocator, 4, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);
    var fb: [512]u8 = undefined;

    var df: [frame_header_len + 2]u8 = undefined;
    @memcpy(df[frame_header_len..], "xx");

    // Stream 1 answered and freed; a late DATA for it is the legal race.
    const r = try c.readFrame(testHeadersFrame(&fb, 1, Flags.end_headers | Flags.end_stream, "/a"), &out);
    c.closeStream(r.event.request);
    (FrameHeader{ .length = 2, .frame_type = .data, .flags = 0, .stream_id = 1 }).write(df[0..frame_header_len]);
    try testing.expectEqual(Event.progress, (try c.readFrame(&df, &out)).event);

    // Stream 3 was never opened: RFC 9113 §5.1, idle, PROTOCOL_ERROR. It used
    // to be silently dropped, and hyper's h2 rejects it, so a peer relying on
    // the drop was already broken elsewhere.
    (FrameHeader{ .length = 2, .frame_type = .data, .flags = 0, .stream_id = 3 }).write(df[0..frame_header_len]);
    try testing.expectError(Error.ProtocolError, c.readFrame(&df, &out));
}

test "WINDOW_UPDATE on an idle stream is a protocol error, on a closed one it is credited or dropped" {
    const bufs = try testBuffers(testing.allocator, 4, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);
    var fb: [512]u8 = undefined;

    var wf: [frame_header_len + 4]u8 = undefined;
    std.mem.writeInt(u32, wf[frame_header_len..][0..4], 1024, .big);

    // Stream 1 is open: the credit lands on its send window.
    _ = try c.readFrame(testHeadersFrame(&fb, 1, Flags.end_headers, "/a"), &out);
    const idx = c.findStream(1).?;
    const before = c.streams[idx].window;
    (FrameHeader{ .length = 4, .frame_type = .window_update, .flags = 0, .stream_id = 1 }).write(wf[0..frame_header_len]);
    try testing.expectEqual(Event.progress, (try c.readFrame(&wf, &out)).event);
    try testing.expectEqual(before + 1024, c.streams[idx].window);

    // Stream 1 answered and freed: a late credit for it is the legal race
    // §6.9 requires us to ignore, not to error on.
    c.closeStream(idx);
    try testing.expectEqual(Event.progress, (try c.readFrame(&wf, &out)).event);

    // Stream 3 was never opened: RFC 9113 §5.1, idle, PROTOCOL_ERROR. It used
    // to be silently dropped, which lost the idle/closed distinction entirely.
    (FrameHeader{ .length = 4, .frame_type = .window_update, .flags = 0, .stream_id = 3 }).write(wf[0..frame_header_len]);
    try testing.expectError(Error.ProtocolError, c.readFrame(&wf, &out));
}

test "an over-long :path is an unknown method on that stream, not a connection error" {
    // A `:path` longer than `Stream.path_buf` used to be a PROTOCOL_ERROR
    // GOAWAY, tearing down every other stream for one bad request. It is now
    // truncated, which no real method name can match, and dispatched so the
    // server answers UNIMPLEMENTED on that stream alone.
    const bufs = try testBuffers(testing.allocator, 2, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);
    var fb: [512]u8 = undefined;

    const long = "/" ++ "x" ** 200;
    const r = try c.readFrame(testHeadersFrame(&fb, 1, Flags.end_headers | Flags.end_stream, long), &out);
    const idx = r.event.request;
    try testing.expectEqual(c.streams[idx].path_buf.len, c.streams[idx].path().len);
    try testing.expectEqualStrings(long[0..c.streams[idx].path_buf.len], c.streams[idx].path());
    try testing.expect(!c.goaway_sent);
}

test "streams opened after the peer's GOAWAY are refused, in-flight ones are kept" {
    // RFC 9113 §6.8: a GOAWAY sender still expects responses on the streams
    // it already opened; the server must not drop them, and has no business
    // dispatching new ones to a connection that is going away.
    const bufs = try testBuffers(testing.allocator, 2, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [4096]u8 = undefined;
    var out = OutBuf.init(&ob);
    var fb: [512]u8 = undefined;

    const r1 = try c.readFrame(testHeadersFrame(&fb, 1, Flags.end_headers | Flags.end_stream, "/a"), &out);
    const held = r1.event.request;

    var g: [frame_header_len + 8]u8 = undefined;
    (FrameHeader{ .length = 8, .frame_type = .goaway, .flags = 0, .stream_id = 0 }).write(g[0..frame_header_len]);
    @memset(g[frame_header_len..], 0);
    try testing.expectEqual(Event.goaway, (try c.readFrame(&g, &out)).event);

    // The in-flight stream is untouched.
    try testing.expectEqual(StreamState.half_closed_remote, c.streams[held].state);
    try testing.expectEqual(@as(u31, 1), c.streams[held].id);

    // A new one is refused (retryable), its block still decoded.
    out.reset();
    const r2 = try c.readFrame(testHeadersFrame(&fb, 3, Flags.end_headers | Flags.end_stream, "/b"), &out);
    try testing.expectEqual(Event.progress, r2.event);
    try testing.expectEqualSlices(u8, &expectRst(3, .refused_stream), out.written()[0 .. frame_header_len + 4]);
    try testing.expectEqual(@as(?usize, null), c.findStream(3));
}

test "peer SETTINGS_MAX_FRAME_SIZE and both send windows bound what dataBudget allows" {
    const bufs = try testBuffers(testing.allocator, 2, 1024);
    defer freeBuffers(testing.allocator, bufs);
    var c = Connection.init(bufs);
    var ob: [128 * 1024]u8 = undefined;
    var out = OutBuf.init(&ob);
    const idx = try c.openStream(1);

    // RFC defaults until the peer says otherwise: 16384-byte frames, 65535
    // windows. hyper enforces exactly these.
    try testing.expectEqual(@as(usize, 16384), c.dataBudget(&out, idx));

    // The stream window is the tighter bound once it drops below a frame.
    c.streams[idx].window = 100;
    try testing.expectEqual(@as(usize, 100), c.dataBudget(&out, idx));
    c.streams[idx].window = 65535;
    // Then the connection window.
    c.send_window = 7;
    try testing.expectEqual(@as(usize, 7), c.dataBudget(&out, idx));
    c.send_window = 0;
    try testing.expectEqual(@as(usize, 0), c.dataBudget(&out, idx));
    c.send_window = 65535;

    // The output buffer keeps `reserve` plus a header free.
    out.len = ob.len - OutBuf.reserve - frame_header_len - 10;
    try testing.expectEqual(@as(usize, 10), c.dataBudget(&out, idx));
    out.len = ob.len - OutBuf.reserve;
    try testing.expectEqual(@as(usize, 0), c.dataBudget(&out, idx));
    out.len = 0;

    // A larger peer frame size is honoured, and charging is on both windows.
    c.peer.max_frame_size = 1 << 20;
    try testing.expectEqual(@as(usize, 65535), c.dataBudget(&out, idx));
    c.chargeSend(idx, 1000);
    try testing.expectEqual(@as(i32, 64535), c.send_window);
    try testing.expectEqual(@as(i32, 64535), c.streams[idx].window);
}
