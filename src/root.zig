//! llama.zig library root — re-exports core modules for `zig build test`.
const std = @import("std");

pub const gguf = @import("gguf.zig");
pub const tensor = @import("tensor.zig");
pub const model = @import("model.zig");
pub const weights = @import("weights.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const compute_graph = @import("compute_graph.zig");
pub const vulkan_backend = @import("vulkan_backend.zig");

test {
    _ = @import("weights.zig");
    _ = @import("sampler.zig");
    _ = @import("tokenizer.zig");
}

test "bf16 roundtrip" {
    const w = @import("weights.zig");
    const val: f32 = 3.14159;
    const bits: u16 = @truncate(@as(u32, @bitCast(val)) >> 16);
    const back = w.bf16ToF32(bits);
    try std.testing.expectApproxEqAbs(val, back, 0.01);
}
