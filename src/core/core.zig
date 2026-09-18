//! §3 / §6.4, the data model and storage.

const std = @import("std");

pub const ids = @import("ids.zig");
pub const collection = @import("collection.zig");
pub const scroll = @import("scroll.zig");
pub const quantized_search = @import("quantized_search.zig");
pub const persist = @import("persist.zig");
pub const quantized = @import("quantized.zig");
pub const storage = @import("storage.zig");
pub const payload = @import("payload.zig");

pub const ExternalId = ids.ExternalId;
pub const IdSpace = ids.IdSpace;
pub const Collection = collection.Collection;
pub const Config = collection.Config;
pub const Status = collection.Status;

test {
    std.testing.refAllDecls(@This());
    _ = ids;
    _ = collection;
    _ = quantized_search;
    _ = persist;
    _ = quantized;
    _ = storage;
    _ = payload;
}
