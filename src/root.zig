//! strawmann, a Qdrant-wire-compatible vector search engine in Zig.
//!
//! This is not a product. It is a performance oracle: the output is a set of
//! validated cost models and a demonstration of how close a real server gets to
//! them. See `docs/spec.md`.
//!
//! Everything is exported from here so that the server, the benchmarks and the
//! tests all compile the same code.

const std = @import("std");

pub const lock = @import("lock.zig");
pub const dist = @import("dist/dist.zig");
pub const sysinfo = @import("sysinfo.zig");
pub const proto = @import("proto/proto.zig");
pub const net = @import("net/net.zig");
pub const core = @import("core/core.zig");
pub const index = @import("index/index.zig");
pub const api = @import("api/api.zig");
pub const quant = @import("quant/quant.zig");

test {
    // Zig 0.16 dropped `refAllDeclsRecursive`; the module tree is walked
    // explicitly so that adding a file without wiring it in here is a visible
    // omission rather than a silently untested module.
    std.testing.refAllDecls(@This());
    _ = lock;
    _ = dist;
    _ = proto;
    _ = net;
    _ = core;
    _ = index;
    _ = api;
    _ = quant;
}
