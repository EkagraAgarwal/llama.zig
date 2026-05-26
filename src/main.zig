const std = @import("std");
const tensor = @import("tensor.zig");
const vulkan = @import("vulkan_backend.zig");
const gguf = @import("gguf.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const stdout_file = std.Io.File.stdout();
    var buffer: [4096]u8 = undefined;
    var writer = stdout_file.writerStreaming(init.io, &buffer);

    try writer.interface.print("llama.zig: llama.cpp engine port (Vulkan backend)\n", .{});
    
    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_it.deinit();
    
    _ = args_it.next(); // Skip executable name

    var model_path: ?[]const u8 = null;

    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            model_path = args_it.next();
        }
    }

    if (model_path == null) {
        try writer.interface.print("Usage: llama.zig --model <path.gguf>\n", .{});
        try writer.interface.flush();
        return;
    }

    try writer.interface.print("Loading model: {s}\n", .{model_path.?});
    try writer.interface.flush();
    var ctx = try gguf.loadModel(allocator, model_path.?);
    defer ctx.deinit();

    try writer.interface.print("Successfully loaded model: {s}\n", .{model_path.?});
    try writer.interface.print("GGUF Version: {}\n", .{ctx.version});
    try writer.interface.print("Tensor Count: {}\n", .{ctx.tensor_count});
    try writer.interface.print("KV Count: {}\n", .{ctx.kv_count});
    
    try writer.interface.print("Initializing Vulkan backend...\n", .{});
    try writer.interface.flush();
    
    // Skip Vulkan for now to show parser success
    // var vk_ctx = try vulkan.Context.init(allocator);
    // defer vk_ctx.deinit();
    // try writer.interface.print("Vulkan backend initialized.\n", .{});
    
    try writer.interface.flush();
}

test "basic test" {
    try std.testing.expectEqual(10, 5 + 5);
}
