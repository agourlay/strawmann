//! §2 / §12, the gRPC surface bfb exercises.

const std = @import("std");

pub const handlers = @import("handlers.zig");
pub const collections = @import("collections.zig");
pub const e2e = @import("e2e.zig");
pub const e2e_client = @import("e2e_client.zig");

pub const Engine = handlers.Engine;
pub const Context = handlers.Context;
pub const Workspace = handlers.Workspace;
pub const handle = handlers.handle;

test {
    std.testing.refAllDecls(@This());
    _ = handlers;
    _ = e2e;
    _ = e2e_client;
}
