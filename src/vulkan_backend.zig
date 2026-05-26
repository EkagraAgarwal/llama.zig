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
    n: u32,
    d: u32 = 0,
    a: u64,
    b: u64,
    c: u64,
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

    vki: vk.InstanceWrapper,
    vkd: vk.DeviceWrapper,

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

        var layer_count: u32 = 0;
        _ = try base_dispatch.enumerateInstanceLayerProperties(&layer_count, null);
        const layers = try allocator.alloc(vk.LayerProperties, layer_count);
        defer allocator.free(layers);
        _ = try base_dispatch.enumerateInstanceLayerProperties(&layer_count, layers.ptr);

        var validation_layer_enabled = false;
        const validation_layer_name = "VK_LAYER_KHRONOS_validation";
        for (layers) |layer| {
            if (std.mem.eql(u8, std.mem.sliceTo(&layer.layer_name, 0), validation_layer_name)) {
                validation_layer_enabled = true;
                break;
            }
        }

        const app_info = vk.ApplicationInfo{
            .p_application_name = "llama.zig",
            .application_version = vk.makeApiVersion(0, 0, 1, 0).toU32(),
            .p_engine_name = "llama.zig",
            .engine_version = vk.makeApiVersion(0, 0, 1, 0).toU32(),
            .api_version = vk.API_VERSION_1_2.toU32(),
        };

        const layers_to_enable = if (validation_layer_enabled) &[_][*:0]const u8{validation_layer_name} else &[_][*:0]const u8{};

        const instance = try base_dispatch.createInstance(&.{
            .p_application_info = &app_info,
            .enabled_layer_count = @as(u32, @intCast(layers_to_enable.len)),
            .pp_enabled_layer_names = layers_to_enable.ptr,
        }, null);

        const vki = vk.InstanceWrapper.load(instance, vkGetInstanceProcAddr);
        errdefer vki.destroyInstance(instance, null);

        var pdev_count: u32 = 0;
        _ = try vki.enumeratePhysicalDevices(instance, &pdev_count, null);
        if (pdev_count == 0) return error.NoVulkanDevicesFound;

        const pdevs = try allocator.alloc(vk.PhysicalDevice, pdev_count);
        defer allocator.free(pdevs);
        _ = try vki.enumeratePhysicalDevices(instance, &pdev_count, pdevs.ptr);

        var best_pdev: ?vk.PhysicalDevice = null;
        var best_score: u32 = 0;
        var best_compute_index: u32 = 0;

        for (pdevs) |candidate| {
            const props = vki.getPhysicalDeviceProperties(candidate);

            var qf_count: u32 = 0;
            vki.getPhysicalDeviceQueueFamilyProperties(candidate, &qf_count, null);
            const qfs = try allocator.alloc(vk.QueueFamilyProperties, qf_count);
            defer allocator.free(qfs);
            vki.getPhysicalDeviceQueueFamilyProperties(candidate, &qf_count, qfs.ptr);

            var compute_index: ?u32 = null;
            for (qfs, 0..) |qf, i| {
                if (qf.queue_flags.compute_bit) {
                    compute_index = @as(u32, @intCast(i));
                    break;
                }
            }

            if (compute_index == null) continue;

            var score: u32 = 1;
            if (props.device_type == .discrete_gpu) score += 1000;
            if (props.device_type == .integrated_gpu) score += 500;

            if (score > best_score) {
                best_score = score;
                best_pdev = candidate;
                best_compute_index = compute_index.?;
            }
        }

        const pdev = best_pdev orelse return error.NoSuitableVulkanDevice;
        const props = vki.getPhysicalDeviceProperties(pdev);
        const mem_props = vki.getPhysicalDeviceMemoryProperties(pdev);
        std.debug.print("Selected GPU: {s}\n", .{std.mem.sliceTo(&props.device_name, 0)});

        var bda_features = vk.PhysicalDeviceBufferDeviceAddressFeatures{
            .p_next = null,
            .buffer_device_address = .true,
            .buffer_device_address_capture_replay = .false,
            .buffer_device_address_multi_device = .false,
        };
        var features2 = vk.PhysicalDeviceFeatures2{
            .p_next = &bda_features,
            .features = .{
                .shader_int_64 = .true,
            },
        };
        vki.getPhysicalDeviceFeatures2(pdev, &features2);

        if (bda_features.buffer_device_address == .false) {
            return error.VulkanDeviceDoesNotSupportBufferDeviceAddress;
        }

        const queue_priority = [_]f32{1.0};
        const queue_create_info = vk.DeviceQueueCreateInfo{
            .queue_family_index = best_compute_index,
            .queue_count = 1,
            .p_queue_priorities = &queue_priority,
        };

        const device_extensions = [_][*:0]const u8{
            vk.extensions.khr_buffer_device_address.name,
        };

        const device = try vki.createDevice(pdev, &.{
            .p_next = &features2,
            .queue_create_info_count = 1,
            .p_queue_create_infos = (&queue_create_info)[0..1],
            .enabled_extension_count = @as(u32, @intCast(device_extensions.len)),
            .pp_enabled_extension_names = &device_extensions,
        }, null);

        const vkd = vk.DeviceWrapper.load(device, vki.dispatch.vkGetDeviceProcAddr.?);
        errdefer vkd.destroyDevice(device, null);

        const compute_queue = vkd.getDeviceQueue(device, best_compute_index, 0);

        const cmd_pool = try vkd.createCommandPool(device, &.{
            .queue_family_index = best_compute_index,
            .flags = .{ .reset_command_buffer_bit = true },
        }, null);

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

    pub fn findMemoryType(self: Context, type_filter: u32, properties: vk.MemoryPropertyFlags) !u32 {
        for (0..self.mem_props.memory_type_count) |i| {
            const i_u32 = @as(u32, @intCast(i));
            if ((type_filter & (@as(u32, 1) << @as(u5, @intCast(i_u32)))) != 0 and
                (self.mem_props.memory_types[i].property_flags.toInt() & properties.toInt()) == properties.toInt())
            {
                return i_u32;
            }
        }
        return error.NoSuitableMemoryType;
    }

    pub fn deinit(self: *Context) void {
        self.vkd.destroyCommandPool(self.device, self.cmd_pool, null);
        self.vkd.destroyDevice(self.device, null);
        self.vki.destroyInstance(self.instance, null);
        if (builtin.os.tag == .windows) {
            _ = windows.FreeLibrary(self.handle);
        }
    }

    pub fn createShaderModule(self: Context, code: []const u8) !vk.ShaderModule {
        const create_info = vk.ShaderModuleCreateInfo{
            .code_size = code.len,
            .p_code = @ptrCast(@alignCast(code.ptr)),
        };
        return try self.vkd.createShaderModule(self.device, &create_info, null);
    }

    pub fn copyBuffer(self: Context, src: Buffer, dst: Buffer, size: u64) !void {
        try self.copyBufferOffset(src, 0, dst, 0, size);
    }

    pub fn copyBufferOffset(self: Context, src: Buffer, src_offset: u64, dst: Buffer, dst_offset: u64, size: u64) !void {
        var cmd_buf: vk.CommandBuffer = undefined;
        try self.vkd.allocateCommandBuffers(self.device, &.{
            .command_pool = self.cmd_pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, (&cmd_buf)[0..1]);

        try self.vkd.beginCommandBuffer(cmd_buf, &.{
            .flags = .{ .one_time_submit_bit = true },
        });

        const copy_region = vk.BufferCopy{
            .src_offset = src_offset,
            .dst_offset = dst_offset,
            .size = size,
        };
        self.vkd.cmdCopyBuffer(cmd_buf, src.buffer, dst.buffer, (&copy_region)[0..1]);

        try self.vkd.endCommandBuffer(cmd_buf);

        const submit_info = vk.SubmitInfo{
            .command_buffer_count = 1,
            .p_command_buffers = (&cmd_buf)[0..1],
        };

        try self.vkd.queueSubmit(self.compute_queue, (&submit_info)[0..1], .null_handle);
        try self.vkd.queueWaitIdle(self.compute_queue);

        self.vkd.freeCommandBuffers(self.device, self.cmd_pool, (&cmd_buf)[0..1]);
    }
};

pub const Buffer = struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
    size: u64,
    address: u64 = 0,

    pub fn init(ctx: Context, size: u64, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags) !Buffer {
        const buffer = try ctx.vkd.createBuffer(ctx.device, &.{
            .size = size,
            .usage = usage,
            .sharing_mode = .exclusive,
        }, null);
        errdefer ctx.vkd.destroyBuffer(ctx.device, buffer, null);

        const mem_requirements = ctx.vkd.getBufferMemoryRequirements(ctx.device, buffer);
        const mem_type = try ctx.findMemoryType(mem_requirements.memory_type_bits, properties);

        var flags_info = vk.MemoryAllocateFlagsInfo{
            .p_next = null,
            .flags = .{ .device_address_bit = true },
            .device_mask = 0,
        };

        var alloc_info = vk.MemoryAllocateInfo{
            .allocation_size = mem_requirements.size,
            .memory_type_index = mem_type,
            .p_next = if (usage.shader_device_address_bit) &flags_info else null,
        };

        const memory = try ctx.vkd.allocateMemory(ctx.device, &alloc_info, null);
        errdefer ctx.vkd.freeMemory(ctx.device, memory, null);

        try ctx.vkd.bindBufferMemory(ctx.device, buffer, memory, 0);

        var address: u64 = 0;
        if (usage.shader_device_address_bit) {
            address = ctx.vkd.getBufferDeviceAddress(ctx.device, &.{ .buffer = buffer });
        }

        return Buffer{
            .buffer = buffer,
            .memory = memory,
            .size = size,
            .address = address,
        };
    }

    pub fn deinit(self: Buffer, ctx: Context) void {
        ctx.vkd.destroyBuffer(ctx.device, self.buffer, null);
        ctx.vkd.freeMemory(ctx.device, self.memory, null);
    }
};

pub const Pipeline = struct {
    pipeline: vk.Pipeline,
    layout: vk.PipelineLayout,

    pub fn init(ctx: Context, shader: vk.ShaderModule, entry_point: [*:0]const u8) !Pipeline {
        const pc_range = vk.PushConstantRange{
            .stage_flags = .{ .compute_bit = true },
            .offset = 0,
            .size = @sizeOf(PushConstants),
        };

        const layout = try ctx.vkd.createPipelineLayout(ctx.device, &.{
            .flags = .{},
            .set_layout_count = 0,
            .p_set_layouts = null,
            .push_constant_range_count = 1,
            .p_push_constant_ranges = (&pc_range)[0..1],
        }, null);
        errdefer ctx.vkd.destroyPipelineLayout(ctx.device, layout, null);

        const stage_info = vk.PipelineShaderStageCreateInfo{
            .stage = .{ .compute_bit = true },
            .module = shader,
            .p_name = entry_point,
            .p_specialization_info = null,
        };

        const compute_info = vk.ComputePipelineCreateInfo{
            .stage = stage_info,
            .layout = layout,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
        };

        var pipeline: vk.Pipeline = undefined;
        _ = try ctx.vkd.createComputePipelines(ctx.device, .null_handle, (&compute_info)[0..1], null, (&pipeline)[0..1]);

        return Pipeline{
            .pipeline = pipeline,
            .layout = layout,
        };
    }

    pub fn deinit(self: Pipeline, ctx: Context) void {
        ctx.vkd.destroyPipeline(ctx.device, self.pipeline, null);
        ctx.vkd.destroyPipelineLayout(ctx.device, self.layout, null);
    }
};

pub const PipelineRegistry = struct {
    allocator: std.mem.Allocator,
    add_shader: vk.ShaderModule,
    mul_shader: vk.ShaderModule,
    rmsnorm_shader: vk.ShaderModule,
    softmax_shader: vk.ShaderModule,
    add_pipeline: Pipeline,
    mul_pipeline: Pipeline,
    rmsnorm_pipeline: Pipeline,
    softmax_pipeline: Pipeline,

    pub fn init(allocator: std.mem.Allocator) !PipelineRegistry {
        return PipelineRegistry{
            .allocator = allocator,
            .add_shader = .null_handle,
            .mul_shader = .null_handle,
            .rmsnorm_shader = .null_handle,
            .softmax_shader = .null_handle,
            .add_pipeline = .{ .pipeline = .null_handle, .layout = .null_handle },
            .mul_pipeline = .{ .pipeline = .null_handle, .layout = .null_handle },
            .rmsnorm_pipeline = .{ .pipeline = .null_handle, .layout = .null_handle },
            .softmax_pipeline = .{ .pipeline = .null_handle, .layout = .null_handle },
        };
    }

    pub fn deinit(self: *PipelineRegistry, ctx: Context) void {
        if (self.add_shader != .null_handle) {
            self.add_pipeline.deinit(ctx);
            ctx.vkd.destroyShaderModule(ctx.device, self.add_shader, null);
        }
        if (self.mul_shader != .null_handle) {
            self.mul_pipeline.deinit(ctx);
            ctx.vkd.destroyShaderModule(ctx.device, self.mul_shader, null);
        }
        if (self.rmsnorm_shader != .null_handle) {
            self.rmsnorm_pipeline.deinit(ctx);
            ctx.vkd.destroyShaderModule(ctx.device, self.rmsnorm_shader, null);
        }
        if (self.softmax_shader != .null_handle) {
            self.softmax_pipeline.deinit(ctx);
            ctx.vkd.destroyShaderModule(ctx.device, self.softmax_shader, null);
        }
    }

    pub fn register(self: *PipelineRegistry, ctx: Context, name: []const u8, shader_code: []const u8, entry_point: [*:0]const u8) !void {
        const shader = try ctx.createShaderModule(shader_code);
        const pipeline = try Pipeline.init(ctx, shader, entry_point);

        if (std.mem.eql(u8, name, "add")) {
            self.add_shader = shader;
            self.add_pipeline = pipeline;
        } else if (std.mem.eql(u8, name, "mul")) {
            self.mul_shader = shader;
            self.mul_pipeline = pipeline;
        } else if (std.mem.eql(u8, name, "rmsnorm")) {
            self.rmsnorm_shader = shader;
            self.rmsnorm_pipeline = pipeline;
        } else if (std.mem.eql(u8, name, "softmax")) {
            self.softmax_shader = shader;
            self.softmax_pipeline = pipeline;
        }
    }
};
