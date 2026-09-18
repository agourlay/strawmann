//! §6.1, the transport.
//!
//! h2c with prior knowledge, no TLS, no compression, server-side only.
//! Hand-written rather than nghttp2-backed: the build host has no nghttp2, so
//! §6.1's "M6: hand-rolled h2" is the only available option rather than an
//! optimisation deferred until profiling justifies it.

const std = @import("std");

pub const hpack = @import("hpack.zig");
pub const h2 = @import("h2.zig");
pub const grpc = @import("grpc.zig");
pub const server = @import("server.zig");
pub const fuzz = @import("fuzz.zig");

pub const Server = server.Server;
pub const Config = server.Config;
pub const Request = server.Request;
pub const Completion = server.Completion;
pub const ResponseBuf = server.ResponseBuf;
pub const Status = grpc.Status;
pub const Address = server.Address;

test {
    std.testing.refAllDecls(@This());
    _ = hpack;
    _ = h2;
    _ = grpc;
    _ = server;
    _ = fuzz;
}
