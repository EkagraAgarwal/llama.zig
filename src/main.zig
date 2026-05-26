const std = @import("std");
const tensor = @import("tensor.zig");
const vulkan = @import("vulkan_backend.zig");
const gguf = @import("gguf.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("llama.zig: llama.cpp engine port (Vulkan backend)\n", .{});
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try stdout.print("Usage: {s} --model <path.gguf>\n", .{args[0]});
        return;
    }

    try stdout.print("Initializing Vulkan backend...\n", .{});
    var vk_ctx = try vulkan.Context.init(allocator);
    defer vk_ctx.deinit();

    try stdout.print("Vulkan backend initialized.\n", .{});
    
    // Placeholder for GGUF loading
    // const model = try gguf.loadModel(allocator, args[2]);
    // defer model.deinit();
}

test "basic test" {
    try std.testing.expectEqual(10, 5 + 5);
}
