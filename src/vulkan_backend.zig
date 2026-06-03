const std = @import("std");
const vk = @import("vulkan");
const builtin = @import("builtin");

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
    /// Buffer device addresses. Most ops use only a/b/c; the SSM ops
    /// (ssm_conv1d, ssm_delta_net_decode, ssm_gated_norm) and the
    /// multi-input dispatcher paths use d/e/f/g.
    a: u64 = 0,
    b: u64 = 0,
    c: u64 = 0,
    d: u64 = 0,
    e: u64 = 0,
    f: u64 = 0,
    g: u64 = 0,
};

pub const Context = struct {
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

    pub fn init(allocator: std.mem.Allocator) !Context {
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

        return Context{
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

    pub fn deinit(self: *Context) void {
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

    pub fn findMemoryType(self: Context, type_filter: u32, properties: vk.MemoryPropertyFlags) !u32 {
        for (0..self.mem_props.memory_type_count) |i| {
            if ((type_filter & (@as(u32, 1) << @as(u5, @intCast(i)))) != 0 and (self.mem_props.memory_types[i].property_flags.toInt() & properties.toInt()) == properties.toInt()) return @intCast(i);
        }
        return error.NoSuitableMemoryType;
    }

    pub fn createShaderModule(self: Context, code: []const u8) !vk.ShaderModule {
        // Vulkan's pCode is [*]const u32 (4-byte aligned per spec). @embedFile
        // returns 1-byte aligned data and the linker provides no alignment
        // guarantees, so we copy into a 4-aligned scratch buffer.
        const aligned_code = try self.allocator.alignedAlloc(u8, .@"4", code.len);
        defer self.allocator.free(aligned_code);
        @memcpy(aligned_code, code);

        var shader: vk.ShaderModule = undefined;
        const result = self.vkd.dispatch.vkCreateShaderModule.?(
            self.device,
            &.{ .flags = .{}, .code_size = code.len, .p_code = @ptrCast(aligned_code.ptr) },
            null,
            &shader,
        );
        if (result != .success) return error.ShaderModuleCreationFailed;
        return shader;
    }

    fn ensureTransferFence(self: *Context) !vk.Fence {
        if (self.transfer_fence == .null_handle) {
            _ = self.vkd.dispatch.vkCreateFence.?(self.device, &.{ .flags = .{} }, null, &self.transfer_fence);
        }
        return self.transfer_fence;
    }

    fn ensureTransferCmd(self: *Context) !vk.CommandBuffer {
        if (self.transfer_cmd == .null_handle) {
            _ = self.vkd.dispatch.vkAllocateCommandBuffers.?(
                self.device,
                &.{ .command_pool = self.cmd_pool, .level = .primary, .command_buffer_count = 1 },
                (&self.transfer_cmd)[0..1],
            );
        }
        return self.transfer_cmd;
    }

    pub fn copyBufferOffset(self: *Context, src: Buffer, src_off: u64, dst: Buffer, dst_off: u64, size: u64) !void {
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

    pub fn copyBuffer(self: *Context, src: Buffer, dst: Buffer, size: u64) !void {
        try self.copyBufferOffset(src, 0, dst, 0, size);
    }
};

pub const Buffer = struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
    size: u64,
    address: u64 = 0,

    pub fn init(ctx: *const Context, size: u64, usage: vk.BufferUsageFlags, props: vk.MemoryPropertyFlags) !Buffer {
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

    pub fn deinit(self: Buffer, ctx: *const Context) void {
        ctx.vkd.dispatch.vkDestroyBuffer.?(ctx.device, self.buffer, null);
        ctx.vkd.dispatch.vkFreeMemory.?(ctx.device, self.memory, null);
    }
};

pub const Pipeline = struct {
    pipeline: vk.Pipeline,
    layout: vk.PipelineLayout,

    pub fn init(ctx: *const Context, shader: vk.ShaderModule, entry: [*:0]const u8) !Pipeline {
        const pc_range = vk.PushConstantRange{ .stage_flags = .{ .compute_bit = true }, .offset = 0, .size = @sizeOf(PushConstants) };
        var layout: vk.PipelineLayout = undefined;
        const res1 = ctx.vkd.dispatch.vkCreatePipelineLayout.?(ctx.device, &.{ .flags = .{}, .set_layout_count = 0, .p_set_layouts = null, .push_constant_range_count = 1, .p_push_constant_ranges = (&pc_range)[0..1] }, null, &layout);
        if (res1 != .success) return error.PipelineLayoutCreationFailed;

        var pipeline: vk.Pipeline = undefined;
        const res2 = ctx.vkd.dispatch.vkCreateComputePipelines.?(ctx.device, .null_handle, 1, &[_]vk.ComputePipelineCreateInfo{.{ .flags = .{}, .stage = .{ .flags = .{}, .stage = .{ .compute_bit = true }, .module = shader, .p_name = entry, .p_specialization_info = null }, .layout = layout, .base_pipeline_handle = .null_handle, .base_pipeline_index = -1 }}, null, (&pipeline)[0..1]);
        if (res2 != .success) return error.PipelineCreationFailed;

        return Pipeline{ .pipeline = pipeline, .layout = layout };
    }

    pub fn deinit(self: Pipeline, ctx: *const Context) void {
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

    pub fn deinit(self: *PipelineRegistry, ctx: *const Context) void {
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

    pub fn register(self: *PipelineRegistry, ctx: *const Context, name: []const u8, code: []const u8, entry: [*:0]const u8) !void {
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
