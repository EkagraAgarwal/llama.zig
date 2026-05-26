const std = @import("std");
const tensor = @import("tensor.zig");
const vulkan = @import("vulkan_backend.zig");
const gguf = @import("gguf.zig");
const kernels_data = @import("kernels_data");
const vk = @import("vulkan");

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
    
    try writer.interface.print("Initializing Vulkan backend...\n", .{});
    try writer.interface.flush();
    
    var vk_ctx = try vulkan.Context.init(allocator);
    defer vk_ctx.deinit();
    
    try writer.interface.print("Vulkan backend initialized successfully.\n", .{});
    
    // --- GPU TEST CASE: Simple Addition ---
    try writer.interface.print("Running GPU 'Add' test (using Descriptor Sets)...\n", .{});
    try writer.interface.flush();

    const n: u32 = 1024;
    const size = n * @sizeOf(f32);

    // 1. Create Staging Buffer (Host Visible)
    var staging = try vulkan.Buffer.init(vk_ctx, size, .{ .transfer_src_bit = true, .transfer_dst_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer staging.deinit(vk_ctx);

    // 2. Create GPU Buffers (Device Local)
    var buf_a = try vulkan.Buffer.init(vk_ctx, size, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
    defer buf_a.deinit(vk_ctx);
    var buf_b = try vulkan.Buffer.init(vk_ctx, size, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true });
    defer buf_b.deinit(vk_ctx);
    var buf_c = try vulkan.Buffer.init(vk_ctx, size, .{ .storage_buffer_bit = true, .transfer_src_bit = true }, .{ .device_local_bit = true });
    defer buf_c.deinit(vk_ctx);

    // 3. Populate Staging and Upload
    const mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, staging.memory, 0, size, .{});
    const data: [*]f32 = @ptrCast(@alignCast(mapped));
    for (0..n) |i| {
        data[i] = @as(f32, @floatFromInt(i));
    }
    vk_ctx.vkd.unmapMemory(vk_ctx.device, staging.memory);
    try vk_ctx.copyBuffer(staging, buf_a, size);
    
    const mapped_b = try vk_ctx.vkd.mapMemory(vk_ctx.device, staging.memory, 0, size, .{});
    const data_b: [*]f32 = @ptrCast(@alignCast(mapped_b));
    for (0..n) |i| {
        data_b[i] = 10.0;
    }
    vk_ctx.vkd.unmapMemory(vk_ctx.device, staging.memory);
    try vk_ctx.copyBuffer(staging, buf_b, size);

    // 4. Create Pipeline
    const shader_module = try vk_ctx.createShaderModule(kernels_data.data);
    defer vk_ctx.vkd.destroyShaderModule(vk_ctx.device, shader_module, null);

    const pipeline = try vulkan.Pipeline.init(vk_ctx, shader_module, "add", 3);
    defer pipeline.deinit(vk_ctx);

    // 5. Descriptor Set Allocation and Update
    var desc_set: vk.DescriptorSet = undefined;
    try vk_ctx.vkd.allocateDescriptorSets(vk_ctx.device, &.{
        .descriptor_pool = vk_ctx.descriptor_pool,
        .descriptor_set_count = 1,
        .p_set_layouts = (&pipeline.set_layout)[0..1],
    }, (&desc_set)[0..1]);

    const buf_infos = [_]vk.DescriptorBufferInfo{
        .{ .buffer = buf_a.buffer, .offset = 0, .range = size },
        .{ .buffer = buf_b.buffer, .offset = 0, .range = size },
        .{ .buffer = buf_c.buffer, .offset = 0, .range = size },
    };

    const write_desc = vk.WriteDescriptorSet{
        .dst_set = desc_set,
        .dst_binding = 0,
        .dst_array_element = 0,
        .descriptor_count = 3,
        .descriptor_type = .storage_buffer,
        .p_image_info = undefined,
        .p_buffer_info = &buf_infos,
        .p_texel_buffer_view = undefined,
    };
    vk_ctx.vkd.updateDescriptorSets(vk_ctx.device, (&write_desc)[0..1], null);

    // 6. Dispatch
    var cmd_buf: vk.CommandBuffer = undefined;
    try vk_ctx.vkd.allocateCommandBuffers(vk_ctx.device, &.{
        .command_pool = vk_ctx.cmd_pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, (&cmd_buf)[0..1]);

    try vk_ctx.vkd.beginCommandBuffer(cmd_buf, &.{
        .flags = .{ .one_time_submit_bit = true },
    });

    vk_ctx.vkd.cmdBindPipeline(cmd_buf, .compute, pipeline.pipeline);
    vk_ctx.vkd.cmdBindDescriptorSets(cmd_buf, .compute, pipeline.layout, 0, (&desc_set)[0..1], &[_]u32{});

    vk_ctx.vkd.cmdDispatch(cmd_buf, (n + 63) / 64, 1, 1);

    try vk_ctx.vkd.endCommandBuffer(cmd_buf);

    const submit_info = vk.SubmitInfo{
        .command_buffer_count = 1,
        .p_command_buffers = (&cmd_buf)[0..1],
    };
    try vk_ctx.vkd.queueSubmit(vk_ctx.compute_queue, (&submit_info)[0..1], .null_handle);
    try vk_ctx.vkd.queueWaitIdle(vk_ctx.compute_queue);

    // 7. Download and Verify
    try vk_ctx.copyBuffer(buf_c, staging, size);
    const result_mapped = try vk_ctx.vkd.mapMemory(vk_ctx.device, staging.memory, 0, size, .{});
    const result_data: [*]f32 = @ptrCast(@alignCast(result_mapped));
    
    try writer.interface.print("Verification: [0]={} (expected 10.0), [1]={} (expected 11.0), [1023]={} (expected 1033.0)\n", .{result_data[0], result_data[1], result_data[1023]});
    vk_ctx.vkd.unmapMemory(vk_ctx.device, staging.memory);

    try writer.interface.flush();
}

test "basic test" {
    try std.testing.expectEqual(10, 5 + 5);
}
