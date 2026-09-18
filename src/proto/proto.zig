//! §6.2 / §12, the hand-written protobuf layer.
//!
//! Two files: `wire.zig` is format primitives with no Qdrant knowledge, and
//! `messages.zig` is the §12 message subset with no wire-format knowledge
//! beyond what `wire.zig` exposes.
//!
//! There is deliberately no code generator and no `.proto` compilation step.
//! §11 lists "Zig ecosystem gaps (no mature gRPC, no protobuf codegen worth
//! using...)" as high-certainty but low-severity, with the mitigation "hand-
//! write it; the narrow scope is what makes this tractable". The vendored
//! `.proto` files under `proto/` exist as a reference for humans and for the
//! conformance harness, and are never compiled.

const std = @import("std");

pub const wire = @import("wire.zig");
pub const messages = @import("messages.zig");

pub const Reader = wire.Reader;
pub const Writer = wire.Writer;
pub const WireType = wire.WireType;
pub const DecodeError = messages.DecodeError;

test {
    std.testing.refAllDecls(@This());
    _ = wire;
    _ = messages;
}
