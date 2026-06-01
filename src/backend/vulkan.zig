const std = @import("std");
const vk = @import("vulkan");
const builtin = @import("builtin");
const compute_graph = @import("../compute_graph.zig");
const model = @import("../model.zig");
const tensor = @import("../tensor.zig");

const windows = if (builtin.os.tag == .windows) struct {
    pub const HMODULE = std.os.windows.HANDLE;
    pub const FARPROC = ?*anyopaque;
    pub extern "kernel32" fn LoadLibraryW(lpLibFileName: [*:0]const u16) callconv(.winapi) ?HMODULE;
    pub extern "kernel32" fn GetProcAddress(hModule: HMODULE, lpProcName: [*:0]const u8) callconv(.winapi) FARPROC;
    pub extern "kernel32" fn FreeLibrary(hModule: HMODULE) callconv(.winapi) std.os.windows.BOOL;
} else struct {};

pub const PushConstants = extern struct {
    p1: u32,
    p2: u32,
    p3: u32,
    p4: u32 = 0,
    p5: u32 = 0,
    p6: u32 = 0,
    p7: u32 = 0,
    p8: u32 = 0,
    a: u64,
    b: u64,
    c: u64,
};

pub const VulkanBackend = struct {
    allocator: std.mem.Allocator,
    handle: if (builtin.os.tag == .windows) windows.HMODULE else *anyopaque,
    instance: vk.Instance,
    pdev: vk.PhysicalDevice,
    device: vk.Device,
    compute_queue: vk.Queue,
    compute_family_index: u32,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    cmd_pool: vk.CommandPool,
    transfer_fence: vk.Fence = .null_handle,
    transfer_cmd: vk.CommandBuffer = .null_handle,
    transfer_submit_count: u32 = 0,

    vki: vk.InstanceWrapper,
    vkd: *vk.DeviceWrapper,

    pub fn init(allocator: std.mem.Allocator) !VulkanBackend {
        const loader_name = if (builtin.os.tag == .windows) "vulkan-1.dll" else "libvulkan.so.1";
        var handle: if (builtin.os.tag == .windows) windows.HMODULE else *anyopaque = undefined;
        var vkGetInstanceProcAddr: vk.PfnGetInstanceProcAddr = undefined;

        if (builtin.os.tag == .windows) {
            const loader_name_w = try std.unicode.utf8ToUtf16LeAllocZ(allocator, loader_name);
            defer allocator.free(loader_name_w);
            handle = windows.LoadLibraryW(loader_name_w.ptr) orelse return error.VulkanLoaderNotFound;
            const proc = windows.GetProcAddress(handle, "vkGetInstanceProcAddr") orelse return error.VulkanLoaderNotFound;
            vkGetInstanceProcAddr = @ptrCast(proc);
        } else {
            return error.UnsupportedPlatform;
        }

        const base_dispatch = vk.BaseWrapper.load(vkGetInstanceProcAddr);
        const app_info = vk.ApplicationInfo{
            .p_application_name = "llama.zig",
            .application_version = vk.makeApiVersion(0, 0, 1, 0).toU32(),
            .p_engine_name = "llama.zig",
            .engine_version = vk.makeApiVersion(0, 0, 1, 0).toU32(),
            .api_version = vk.API_VERSION_1_2.toU32(),
        };

        const instance = try base_dispatch.createInstance(&.{
            .p_application_info = &app_info,
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = null,
        }, null);

        const vki = vk.InstanceWrapper.load(instance, vkGetInstanceProcAddr);
        errdefer vki.destroyInstance(instance, null);

        var pdev_count: u32 = 0;
        _ = try vki.enumeratePhysicalDevices(instance, &pdev_count, null);
        const pdevs = try allocator.alloc(vk.PhysicalDevice, pdev_count);
        defer allocator.free(pdevs);
        _ = try vki.enumeratePhysicalDevices(instance, &pdev_count, pdevs.ptr);

        var best_pdev: ?vk.PhysicalDevice = null;
        var best_compute_index: u32 = 0;
        var max_score: u32 = 0;

        for (pdevs) |candidate| {
            const props = vki.getPhysicalDeviceProperties(candidate);
            var qf_count: u32 = 0;
            vki.getPhysicalDeviceQueueFamilyProperties(candidate, &qf_count, null);
            const qfs = try allocator.alloc(vk.QueueFamilyProperties, qf_count);
            defer allocator.free(qfs);
            vki.getPhysicalDeviceQueueFamilyProperties(candidate, &qf_count, qfs.ptr);

            for (qfs, 0..) |qf, i| {
                if (qf.queue_flags.compute_bit) {
                    var score: u32 = 1;
                    if (props.device_type == .discrete_gpu) score += 1000;
                    if (score > max_score) {
                        max_score = score;
                        best_pdev = candidate;
                        best_compute_index = @intCast(i);
                    }
                }
            }
        }

        const pdev = best_pdev orelse return error.NoSuitableDevice;
        const props = vki.getPhysicalDeviceProperties(pdev);
        std.debug.print("Selected GPU Device: {s}\n", .{std.mem.sliceTo(&props.device_name, 0)});
        const mem_props = vki.getPhysicalDeviceMemoryProperties(pdev);

        var bda_features = vk.PhysicalDeviceBufferDeviceAddressFeatures{
            .p_next = null,
            .buffer_device_address = .true,
        };
        var features2 = vk.PhysicalDeviceFeatures2{
            .p_next = &bda_features,
            .features = .{ .shader_int_64 = .true },
        };
        vki.getPhysicalDeviceFeatures2(pdev, &features2);

        const queue_priority = [_]f32{1.0};
        const queue_create_info = vk.DeviceQueueCreateInfo{
            .queue_family_index = best_compute_index,
            .queue_count = 1,
            .p_queue_priorities = &queue_priority,
        };

        const device_extensions = [_][*:0]const u8{vk.extensions.khr_buffer_device_address.name};
        const device = try vki.createDevice(pdev, &.{
            .p_next = &features2,
            .queue_create_info_count = 1,
            .p_queue_create_infos = (&queue_create_info)[0..1],
            .enabled_extension_count = 1,
            .pp_enabled_extension_names = &device_extensions,
        }, null);

        const vkd = try allocator.create(vk.DeviceWrapper);
        vkd.* = vk.DeviceWrapper.load(device, vki.dispatch.vkGetDeviceProcAddr.?);

        var compute_queue: vk.Queue = undefined;
        vkd.dispatch.vkGetDeviceQueue.?(device, best_compute_index, 0, &compute_queue);

        var cmd_pool: vk.CommandPool = undefined;
        _ = vkd.dispatch.vkCreateCommandPool.?(device, &.{ .flags = .{ .reset_command_buffer_bit = true }, .queue_family_index = best_compute_index }, null, &cmd_pool);

        return VulkanBackend{
            .allocator = allocator,
            .handle = handle,
            .instance = instance,
            .pdev = pdev,
            .device = device,
            .compute_queue = compute_queue,
            .compute_family_index = best_compute_index,
            .mem_props = mem_props,
            .cmd_pool = cmd_pool,
            .vki = vki,
            .vkd = vkd,
        };
    }

    pub fn deinit(self: *VulkanBackend) void {
        if (self.transfer_fence != .null_handle) {
            self.vkd.dispatch.vkDestroyFence.?(self.device, self.transfer_fence, null);
        }
        if (self.transfer_cmd != .null_handle) {
            self.vkd.dispatch.vkFreeCommandBuffers.?(self.device, self.cmd_pool, 1, (&self.transfer_cmd)[0..1]);
            self.transfer_cmd = .null_handle;
        }
        self.vkd.dispatch.vkDestroyCommandPool.?(self.device, self.cmd_pool, null);
        self.vkd.dispatch.vkDestroyDevice.?(self.device, null);
        self.allocator.destroy(self.vkd);
        self.vki.destroyInstance(self.instance, null);
        if (builtin.os.tag == .windows) _ = windows.FreeLibrary(self.handle);
    }

    pub fn findMemoryType(self: VulkanBackend, type_filter: u32, properties: vk.MemoryPropertyFlags) !u32 {
        for (0..self.mem_props.memory_type_count) |i| {
            if ((type_filter & (@as(u32, 1) << @as(u5, @intCast(i)))) != 0 and (self.mem_props.memory_types[i].property_flags.toInt() & properties.toInt()) == properties.toInt()) return @intCast(i);
        }
        return error.NoSuitableMemoryType;
    }

    pub fn createShaderModule(self: VulkanBackend, code: []const u8) !vk.ShaderModule {
        var shader: vk.ShaderModule = undefined;
        const result = self.vkd.dispatch.vkCreateShaderModule.?(self.device, &.{ .flags = .{}, .code_size = code.len, .p_code = @ptrCast(@alignCast(code.ptr)) }, null, &shader);
        if (result != .success) return error.ShaderModuleCreationFailed;
        return shader;
    }

    fn ensureTransferFence(self: *VulkanBackend) !vk.Fence {
        if (self.transfer_fence == .null_handle) {
            _ = self.vkd.dispatch.vkCreateFence.?(self.device, &.{ .flags = .{} }, null, &self.transfer_fence);
        }
        return self.transfer_fence;
    }

    fn ensureTransferCmd(self: *VulkanBackend) !vk.CommandBuffer {
        if (self.transfer_cmd == .null_handle) {
            _ = self.vkd.dispatch.vkAllocateCommandBuffers.?(
                self.device,
                &.{ .command_pool = self.cmd_pool, .level = .primary, .command_buffer_count = 1 },
                (&self.transfer_cmd)[0..1],
            );
        }
        return self.transfer_cmd;
    }

    pub fn copyBufferOffset(self: *VulkanBackend, src: Buffer, src_off: u64, dst: Buffer, dst_off: u64, size: u64) !void {
        const cmd = try self.ensureTransferCmd();
        const reset_flags = if ((self.transfer_submit_count & 63) == 63)
            vk.CommandBufferResetFlags{ .release_resources_bit = true }
        else
            vk.CommandBufferResetFlags{};
        _ = self.vkd.dispatch.vkResetCommandBuffer.?(cmd, reset_flags);
        _ = self.vkd.dispatch.vkBeginCommandBuffer.?(cmd, &.{ .flags = .{ .one_time_submit_bit = true }, .p_inheritance_info = null });
        self.vkd.dispatch.vkCmdCopyBuffer.?(cmd, src.buffer, dst.buffer, 1, &[_]vk.BufferCopy{.{ .src_offset = src_off, .dst_offset = dst_off, .size = size }});
        _ = self.vkd.dispatch.vkEndCommandBuffer.?(cmd);
        const fence = try self.ensureTransferFence();
        _ = self.vkd.dispatch.vkResetFences.?(self.device, 1, (&fence)[0..1]);
        _ = self.vkd.dispatch.vkQueueSubmit.?(self.compute_queue, 1, &[_]vk.SubmitInfo{.{ .wait_semaphore_count = 0, .p_wait_semaphores = null, .p_wait_dst_stage_mask = null, .command_buffer_count = 1, .p_command_buffers = (&cmd)[0..1], .signal_semaphore_count = 0, .p_signal_semaphores = null }}, fence);
        _ = self.vkd.dispatch.vkWaitForFences.?(self.device, 1, (&fence)[0..1], @enumFromInt(1), std.math.maxInt(u64));
        self.transfer_submit_count += 1;
    }

    pub fn copyBuffer(self: *VulkanBackend, src: Buffer, dst: Buffer, size: u64) !void {
        try self.copyBufferOffset(src, 0, dst, 0, size);
    }
};

pub const Buffer = struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
    size: u64,
    address: u64 = 0,

    pub fn init(ctx: *const VulkanBackend, size: u64, usage: vk.BufferUsageFlags, props: vk.MemoryPropertyFlags) !Buffer {
        var b: vk.Buffer = undefined;
        const res1 = ctx.vkd.dispatch.vkCreateBuffer.?(ctx.device, &.{ .flags = .{}, .size = size, .usage = usage, .sharing_mode = .exclusive, .queue_family_index_count = 0, .p_queue_family_indices = null }, null, &b);
        if (res1 != .success) return error.BufferCreationFailed;

        var reqs: vk.MemoryRequirements = undefined;
        ctx.vkd.dispatch.vkGetBufferMemoryRequirements.?(ctx.device, b, &reqs);
        const mem_type = try ctx.findMemoryType(reqs.memory_type_bits, props);

        var flags = vk.MemoryAllocateFlagsInfo{ .flags = .{ .device_address_bit = true }, .device_mask = 0 };
        const alloc_info = vk.MemoryAllocateInfo{ .p_next = if (usage.shader_device_address_bit) &flags else null, .allocation_size = reqs.size, .memory_type_index = mem_type };
        var memory: vk.DeviceMemory = undefined;
        const res2 = ctx.vkd.dispatch.vkAllocateMemory.?(ctx.device, &alloc_info, null, &memory);
        if (res2 != .success) return error.MemoryAllocationFailed;

        const res3 = ctx.vkd.dispatch.vkBindBufferMemory.?(ctx.device, b, memory, 0);
        if (res3 != .success) return error.MemoryBindingFailed;

        var address: u64 = 0;
        if (usage.shader_device_address_bit) {
            address = ctx.vkd.dispatch.vkGetBufferDeviceAddress.?(ctx.device, &.{ .buffer = b });
        }

        return Buffer{ .buffer = b, .memory = memory, .size = size, .address = address };
    }

    pub fn deinit(self: Buffer, ctx: *const VulkanBackend) void {
        ctx.vkd.dispatch.vkDestroyBuffer.?(ctx.device, self.buffer, null);
        ctx.vkd.dispatch.vkFreeMemory.?(ctx.device, self.memory, null);
    }
};

pub const Pipeline = struct {
    pipeline: vk.Pipeline,
    layout: vk.PipelineLayout,

    pub fn init(ctx: *const VulkanBackend, shader: vk.ShaderModule, entry: [*:0]const u8) !Pipeline {
        const pc_range = vk.PushConstantRange{ .stage_flags = .{ .compute_bit = true }, .offset = 0, .size = @sizeOf(PushConstants) };
        var layout: vk.PipelineLayout = undefined;
        const res1 = ctx.vkd.dispatch.vkCreatePipelineLayout.?(ctx.device, &.{ .flags = .{}, .set_layout_count = 0, .p_set_layouts = null, .push_constant_range_count = 1, .p_push_constant_ranges = (&pc_range)[0..1] }, null, &layout);
        if (res1 != .success) return error.PipelineLayoutCreationFailed;

        var pipeline: vk.Pipeline = undefined;
        const res2 = ctx.vkd.dispatch.vkCreateComputePipelines.?(ctx.device, .null_handle, 1, &[_]vk.ComputePipelineCreateInfo{.{ .flags = .{}, .stage = .{ .flags = .{}, .stage = .{ .compute_bit = true }, .module = shader, .p_name = entry, .p_specialization_info = null }, .layout = layout, .base_pipeline_handle = .null_handle, .base_pipeline_index = -1 }}, null, (&pipeline)[0..1]);
        if (res2 != .success) return error.PipelineCreationFailed;

        return Pipeline{ .pipeline = pipeline, .layout = layout };
    }

    pub fn deinit(self: Pipeline, ctx: *const VulkanBackend) void {
        ctx.vkd.dispatch.vkDestroyPipeline.?(ctx.device, self.pipeline, null);
        ctx.vkd.dispatch.vkDestroyPipelineLayout.?(ctx.device, self.layout, null);
    }
};

pub const PipelineRegistry = struct {
    allocator: std.mem.Allocator,
    pipelines: std.StringHashMap(Pipeline),
    shader_modules: std.StringHashMap(vk.ShaderModule),

    pub fn init(allocator: std.mem.Allocator) !PipelineRegistry {
        return PipelineRegistry{ .allocator = allocator, .pipelines = std.StringHashMap(Pipeline).init(allocator), .shader_modules = std.StringHashMap(vk.ShaderModule).init(allocator) };
    }

    pub fn deinit(self: *PipelineRegistry, ctx: *const VulkanBackend) void {
        var it = self.pipelines.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit(ctx);
            const shader = self.shader_modules.get(entry.key_ptr.*).?;
            ctx.vkd.dispatch.vkDestroyShaderModule.?(ctx.device, shader, null);
            self.allocator.free(entry.key_ptr.*);
        }
        self.pipelines.deinit();
        self.shader_modules.deinit();
    }

    pub fn register(self: *PipelineRegistry, ctx: *const VulkanBackend, name: []const u8, code: []const u8, entry: [*:0]const u8) !void {
        const owned_name = try self.allocator.dupe(u8, name);
        const shader = try ctx.createShaderModule(code);
        const pipeline = try Pipeline.init(ctx, shader, entry);
        try self.pipelines.put(owned_name, pipeline);
        try self.shader_modules.put(owned_name, shader);
    }

    pub fn get(self: *const PipelineRegistry, name: []const u8) ?Pipeline {
        return self.pipelines.get(name);
    }
};

pub const Dispatcher = struct {
    graph: *compute_graph.Graph,
    ctx: *VulkanBackend,
    registry: *PipelineRegistry,
    scratchpad: Buffer,
    kv_cache: Buffer,
    cfg: *const model.ModelConfig,
    cmd: vk.CommandBuffer = .null_handle,
    fence: vk.Fence = .null_handle,
    flash_attn_threshold: u32 = 1,
    submit_count: u32 = 0,
    reported_graph: bool = false,

    pub fn init(graph: *compute_graph.Graph, ctx: *VulkanBackend, registry: *PipelineRegistry, scratch: Buffer, kv: Buffer, cfg: *const model.ModelConfig) !Dispatcher {
        var self = Dispatcher{
            .graph = graph,
            .ctx = ctx,
            .registry = registry,
            .scratchpad = scratch,
            .kv_cache = kv,
            .cfg = cfg,
        };
        try self.ensureSubmitResources();
        return self;
    }

    pub fn deinit(self: *Dispatcher) void {
        if (self.fence != .null_handle) {
            self.ctx.vkd.dispatch.vkDestroyFence.?(self.ctx.device, self.fence, null);
            self.fence = .null_handle;
        }
        if (self.cmd != .null_handle) {
            self.ctx.vkd.dispatch.vkFreeCommandBuffers.?(self.ctx.device, self.ctx.cmd_pool, 1, (&self.cmd)[0..1]);
            self.cmd = .null_handle;
        }
    }

    pub fn ensureSubmitResources(self: *Dispatcher) !void {
        if (self.fence == .null_handle) {
            _ = self.ctx.vkd.dispatch.vkCreateFence.?(self.ctx.device, &.{ .flags = .{} }, null, &self.fence);
        }
        if (self.cmd == .null_handle) {
            _ = self.ctx.vkd.dispatch.vkAllocateCommandBuffers.?(
                self.ctx.device,
                &.{ .command_pool = self.ctx.cmd_pool, .level = .primary, .command_buffer_count = 1 },
                (&self.cmd)[0..1],
            );
        }
    }

    fn tensorAddr(self: *Dispatcher, name: []const u8) u64 {
        const t = self.graph.tensors.get(name) orelse return 0;
        return switch (t.role) {
            .weight => @as(*Buffer, @ptrCast(@alignCast(t.buffer.?))).address,
            .kv_cache => self.kvCacheLayerOffset(t.layer),
            .input, .activation, .output => self.scratchpad.address + t.offset,
        };
    }

    fn kvCacheLayerOffset(self: *Dispatcher, layer: u32) u64 {
        const per_layer = @as(u64, self.cfg.max_ctx) * self.cfg.n_kv_heads * self.cfg.head_dim * 2 * 2;
        return self.kv_cache.address + per_layer * layer;
    }

    fn emitComputeBarrier(self: *Dispatcher, cmd: vk.CommandBuffer) void {
        const barrier = vk.MemoryBarrier{
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
        };
        self.ctx.vkd.dispatch.vkCmdPipelineBarrier.?(
            cmd,
            .{ .compute_shader_bit = true },
            .{ .compute_shader_bit = true },
            .{},
            1,
            (&barrier)[0..1],
            0,
            null,
            0,
            null,
        );
    }

    fn pipelineNameForNode(self: *Dispatcher, node: compute_graph.GraphNode) ?[]const u8 {
        return switch (node.op_type) {
            .matmul_q => blk: {
                break :blk quantPipelineName(node.p5, node.p1 <= 1);
            },
            .get_rows_q => blk: {
                const qtype = @as(u32, node.p5);
                const qt: tensor.Type = @enumFromInt(qtype);
                break :blk switch (qt) {
                    .q4_0 => "get_rows_q4_0",
                    .q4_1 => "get_rows_q4_1",
                    .q4_k => "get_rows_q4_k",
                    .q6_k => "get_rows_q6_k",
                    else => "get_rows_q",
                };
            },
            .attention => blk: {
                if (node.p4 + 1 >= self.flash_attn_threshold) break :blk "attention_flash";
                break :blk "attention";
            },
            .gelu_mul => "gelu_mul",
            else => @tagName(node.op_type),
        };
    }

    pub fn quantPipelineName(qtype: u32, is_matvec: bool) []const u8 {
        if (qtype == compute_graph.q4_0_f16_fallback_qtype) {
            return if (is_matvec) "matvec_f16" else "matmul_f16";
        }
        const qt: tensor.Type = @enumFromInt(qtype);
        return if (is_matvec)
            switch (qt) { .q4_0 => "matvec_q4_0", .q4_1 => "matvec_q4_1", .q4_k => "matvec_q4_k", .q6_k => "matvec_q6_k", else => "matvec_q8_0" }
        else
            switch (qt) { .q4_0 => "matmul_q4_0", .q4_1 => "matmul_q4_1", .q4_k => "matmul_q4_k", .q6_k => "matmul_q6_k", else => "matmul_q8_0" };
    }

    fn dispatchNode(self: *Dispatcher, cmd: vk.CommandBuffer, node: compute_graph.GraphNode, pos: u32) void {
        const pipe_name = self.pipelineNameForNode(node) orelse return;
        const pipe = self.registry.get(pipe_name) orelse return;

        var pc = PushConstants{
            .p1 = node.p1,
            .p2 = node.p2,
            .p3 = node.p3,
            .p4 = node.p4,
            .p5 = node.p5,
            .p6 = node.p6,
            .p7 = node.p7,
            .p8 = node.p8,
            .a = 0,
            .b = 0,
            .c = 0,
        };

        switch (node.op_type) {
            .kv_write => {
                pc.a = self.tensorAddr(node.input_names[0]) + node.p5;
                pc.b = self.tensorAddr(node.input_names[1]) + node.p6;
                pc.c = self.tensorAddr(node.input_names[2]);
                pc.p4 = pos;
            },
            .attention => {
                pc.a = self.tensorAddr(node.input_names[0]) + node.p6;
                pc.b = self.tensorAddr(node.input_names[1]);
                pc.c = self.tensorAddr(node.output_name);
                pc.p4 = pos;
                if (pos + 1 >= self.flash_attn_threshold) pc.p6 = 64;
            },
            .rope => {
                if (node.input_names.len >= 1) pc.a = self.tensorAddr(node.input_names[0]) + node.p5;
                pc.c = self.tensorAddr(node.output_name) + node.p5;
                pc.p3 = pos;
            },
            .get_rows_q => {
                pc.a = self.tensorAddr(node.input_names[0]);
                pc.b = self.tensorAddr(node.input_names[1]);
                pc.c = self.tensorAddr(node.output_name);
            },
            .copy => {
                pc.a = self.tensorAddr(node.input_names[0]);
                pc.c = self.tensorAddr(node.output_name);
            },
            .silu_mul, .gelu_mul => {
                pc.a = self.tensorAddr(node.input_names[0]) + node.p5;
                pc.b = self.tensorAddr(node.input_names[1]) + node.p6;
                pc.c = self.tensorAddr(node.output_name) + node.p7;
            },
            else => {
                if (node.input_names.len >= 1) pc.a = self.tensorAddr(node.input_names[0]);
                if (node.input_names.len >= 2) pc.b = self.tensorAddr(node.input_names[1]);
                pc.c = self.tensorAddr(node.output_name);
            },
        }

        var dx = node.dispatch_x;
        var dy = node.dispatch_y;
        if (node.op_type == .matmul_q and node.p1 <= 1) {
            dx = (node.p2 + 7) / 8;
            dy = 1;
        }

        self.ctx.vkd.dispatch.vkCmdBindPipeline.?(cmd, .compute, pipe.pipeline);
        self.ctx.vkd.dispatch.vkCmdPushConstants.?(cmd, pipe.layout, .{ .compute_bit = true }, 0, @sizeOf(PushConstants), @ptrCast(&pc));
        self.ctx.vkd.dispatch.vkCmdDispatch.?(cmd, dx, dy, node.dispatch_z);
    }

    pub fn submitAndWait(self: *Dispatcher, cmd: vk.CommandBuffer) !void {
        try self.ensureSubmitResources();
        _ = self.ctx.vkd.dispatch.vkResetFences.?(self.ctx.device, 1, (&self.fence)[0..1]);
        const submit_info = vk.SubmitInfo{
            .wait_semaphore_count = 0,
            .p_wait_semaphores = null,
            .p_wait_dst_stage_mask = null,
            .command_buffer_count = 1,
            .p_command_buffers = (&cmd)[0..1],
            .signal_semaphore_count = 0,
            .p_signal_semaphores = null,
        };
        _ = self.ctx.vkd.dispatch.vkQueueSubmit.?(self.ctx.compute_queue, 1, (&submit_info)[0..1], self.fence);
        _ = self.ctx.vkd.dispatch.vkWaitForFences.?(self.ctx.device, 1, (&self.fence)[0..1], @enumFromInt(1), std.math.maxInt(u64));
        self.submit_count += 1;
    }

    fn hasDependency(node1: compute_graph.GraphNode, node2: compute_graph.GraphNode) bool {
        for (node1.input_names) |in_name| {
            if (std.mem.eql(u8, in_name, node2.output_name)) return true;
        }
        for (node2.input_names) |in_name| {
            if (std.mem.eql(u8, node1.output_name, in_name)) return true;
        }
        if (std.mem.eql(u8, node1.output_name, node2.output_name)) return true;
        return false;
    }

    pub fn recordGraph(self: *Dispatcher, cmd: vk.CommandBuffer, pos: u32) void {
        var last_barrier_idx: usize = 0;
        const nodes = self.graph.nodes.items;

        for (nodes, 0..) |node, i| {
            var need_barrier = false;
            if (i > 0) {
                var j = i - 1;
                while (true) {
                    if (hasDependency(node, nodes[j])) {
                        need_barrier = true;
                        break;
                    }
                    if (j == last_barrier_idx) break;
                    j -= 1;
                }
            }

            if (need_barrier) {
                self.emitComputeBarrier(cmd);
                last_barrier_idx = i;
            }

            self.dispatchNode(cmd, node, pos);
        }
    }

    pub fn execute(self: *Dispatcher, pos: u32) !void {
        try self.ensureSubmitResources();
        const reset_flags = if ((self.submit_count & 63) == 63)
            vk.CommandBufferResetFlags{ .release_resources_bit = true }
        else
            vk.CommandBufferResetFlags{};
        _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, reset_flags);
        _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true }, .p_inheritance_info = null });
        
        self.recordGraph(self.cmd, pos);
        
        _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
        try self.submitAndWait(self.cmd);
    }

    pub fn executePrefillBatch(self: *Dispatcher, pos_start: u32, n_tokens: u32, input_batch: Buffer, input_stride: u64) !void {
        const input_tensor = self.graph.tensors.get("input") orelse return error.MissingInputTensor;
        try self.ensureSubmitResources();
        const reset_flags = if ((self.submit_count & 63) == 63)
            vk.CommandBufferResetFlags{ .release_resources_bit = true }
        else
            vk.CommandBufferResetFlags{};
        _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, reset_flags);
        _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true }, .p_inheritance_info = null });

        var i: u32 = 0;
        while (i < n_tokens) : (i += 1) {
            const copy_region = vk.BufferCopy{
                .src_offset = @as(u64, i) * input_stride,
                .dst_offset = input_tensor.offset,
                .size = input_stride,
            };
            self.ctx.vkd.dispatch.vkCmdCopyBuffer.?(self.cmd, input_batch.buffer, self.scratchpad.buffer, 1, (&copy_region)[0..1]);
            const copy_barrier = vk.MemoryBarrier{
                .src_access_mask = .{ .transfer_write_bit = true },
                .dst_access_mask = .{ .shader_read_bit = true },
            };
            self.ctx.vkd.dispatch.vkCmdPipelineBarrier.?(
                self.cmd,
                .{ .transfer_bit = true },
                .{ .compute_shader_bit = true },
                .{},
                1,
                (&copy_barrier)[0..1],
                0,
                null,
                0,
                null,
            );
            self.recordGraph(self.cmd, pos_start + i);
        }

        _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
        try self.submitAndWait(self.cmd);
    }

    pub fn executeGetRowsQ(
        self: *Dispatcher,
        indices_buf: Buffer,
        weights_buf: Buffer,
        out_offset: u64,
        token_id: u32,
        n_embd: u32,
        qtype: u32,
        row_bytes: u32,
        scale_bits: u32,
    ) !void {
        try self.ensureSubmitResources();
        const reset_flags = if ((self.submit_count & 63) == 63)
            vk.CommandBufferResetFlags{ .release_resources_bit = true }
        else
            vk.CommandBufferResetFlags{};
        _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, reset_flags);
        _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true }, .p_inheritance_info = null });

        const mapped = try self.ctx.vkd.mapMemory(self.ctx.device, indices_buf.memory, 0, 4, .{});
        @as(*u32, @ptrCast(@alignCast(mapped))).* = token_id;
        self.ctx.vkd.unmapMemory(self.ctx.device, indices_buf.memory);

        const pipe_name = switch (qtype) {
            @intFromEnum(tensor.Type.q4_0) => "get_rows_q4_0",
            @intFromEnum(tensor.Type.q4_1) => "get_rows_q4_1",
            @intFromEnum(tensor.Type.q4_k) => "get_rows_q4_k",
            @intFromEnum(tensor.Type.q6_k) => "get_rows_q6_k",
            else => "get_rows_q",
        };
        const pipe = self.registry.get(pipe_name) orelse return error.MissingPipeline;
        var pc = PushConstants{
            .p1 = n_embd,
            .p2 = 1,
            .p3 = qtype,
            .p4 = scale_bits,
            .p5 = row_bytes,
            .a = indices_buf.address,
            .b = weights_buf.address,
            .c = self.scratchpad.address + out_offset,
        };
        self.ctx.vkd.dispatch.vkCmdBindPipeline.?(self.cmd, .compute, pipe.pipeline);
        self.ctx.vkd.dispatch.vkCmdPushConstants.?(self.cmd, pipe.layout, .{ .compute_bit = true }, 0, @sizeOf(PushConstants), @ptrCast(&pc));
        self.ctx.vkd.dispatch.vkCmdDispatch.?(self.cmd, (n_embd + 255) / 256, 1, 1);

        _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
        try self.submitAndWait(self.cmd);
    }

    pub fn recordEmbedAndGraph(
        self: *Dispatcher,
        cmd: vk.CommandBuffer,
        pos: u32,
        indices_buf: Buffer,
        weights_buf: Buffer,
        out_offset: u64,
        n_embd: u32,
        qtype: u32,
        row_bytes: u32,
        scale_bits: u32,
    ) void {
        const pipe_name = switch (qtype) {
            @intFromEnum(tensor.Type.q4_0) => "get_rows_q4_0",
            @intFromEnum(tensor.Type.q4_1) => "get_rows_q4_1",
            @intFromEnum(tensor.Type.q4_k) => "get_rows_q4_k",
            @intFromEnum(tensor.Type.q6_k) => "get_rows_q6_k",
            else => "get_rows_q",
        };
        const pipe = self.registry.get(pipe_name) orelse return;
        var pc = PushConstants{
            .p1 = n_embd,
            .p2 = 1,
            .p3 = qtype,
            .p4 = scale_bits,
            .p5 = row_bytes,
            .a = indices_buf.address,
            .b = weights_buf.address,
            .c = self.scratchpad.address + out_offset,
        };
        self.ctx.vkd.dispatch.vkCmdBindPipeline.?(cmd, .compute, pipe.pipeline);
        self.ctx.vkd.dispatch.vkCmdPushConstants.?(cmd, pipe.layout, .{ .compute_bit = true }, 0, @sizeOf(PushConstants), @ptrCast(&pc));
        self.ctx.vkd.dispatch.vkCmdDispatch.?(cmd, (n_embd + 255) / 256, 1, 1);

        const barrier = vk.MemoryBarrier{
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
        };
        self.ctx.vkd.dispatch.vkCmdPipelineBarrier.?(
            cmd,
            .{ .compute_shader_bit = true },
            .{ .compute_shader_bit = true },
            .{},
            1,
            (&barrier)[0..1],
            0,
            null,
            0,
            null,
        );

        self.recordGraph(cmd, pos);
    }

    pub fn executeTopK(
        self: *Dispatcher,
        logits_offset: u64,
        vocab_size: u32,
        out_indices_buf: Buffer,
        out_values_buf: Buffer,
        logit_scale_bits: u32,
    ) !u32 {
        try self.ensureSubmitResources();
        const reset_flags = if ((self.submit_count & 63) == 63)
            vk.CommandBufferResetFlags{ .release_resources_bit = true }
        else
            vk.CommandBufferResetFlags{};
        _ = self.ctx.vkd.dispatch.vkResetCommandBuffer.?(self.cmd, reset_flags);
        _ = self.ctx.vkd.dispatch.vkBeginCommandBuffer.?(self.cmd, &.{ .flags = .{ .one_time_submit_bit = true }, .p_inheritance_info = null });

        const pipe = self.registry.get("topk") orelse return error.MissingPipeline;
        var pc = PushConstants{
            .p1 = vocab_size,
            .p2 = 1,
            .p3 = logit_scale_bits,
            .a = self.scratchpad.address + logits_offset,
            .b = out_indices_buf.address,
            .c = out_values_buf.address,
        };
        self.ctx.vkd.dispatch.vkCmdBindPipeline.?(self.cmd, .compute, pipe.pipeline);
        self.ctx.vkd.dispatch.vkCmdPushConstants.?(self.cmd, pipe.layout, .{ .compute_bit = true }, 0, @sizeOf(PushConstants), @ptrCast(&pc));
        self.ctx.vkd.dispatch.vkCmdDispatch.?(self.cmd, 1, 1, 1);

        _ = self.ctx.vkd.dispatch.vkEndCommandBuffer.?(self.cmd);
        try self.submitAndWait(self.cmd);

        const mapped = try self.ctx.vkd.mapMemory(self.ctx.device, out_indices_buf.memory, 0, 4, .{});
        const id: u32 = @as(*u32, @ptrCast(@alignCast(mapped))).*;
        self.ctx.vkd.unmapMemory(self.ctx.device, out_indices_buf.memory);
        return id;
    }
};
