const std = @import("std");
const vk = @import("vulkan");

/// Manages the Vulkan instance, physical device, logical device, and compute queues.
pub const Context = struct {
    allocator: std.mem.Allocator,
    instance: vk.Instance,
    pdev: vk.PhysicalDevice,
    device: vk.Device,
    compute_queue: vk.Queue,
    compute_family_index: u32,

    vki: InstanceDispatch,
    vkd: DeviceDispatch,

    const InstanceDispatch = vk.InstanceWrapper(.{
        .destroyInstance = true,
        .getPhysicalDeviceQueueFamilyProperties = true,
        .createDevice = true,
        .getDeviceProcAddr = true,
    });

    const DeviceDispatch = vk.DeviceWrapper(.{
        .destroyDevice = true,
        .getDeviceQueue = true,
        .createBuffer = true,
        .allocateMemory = true,
        .bindBufferMemory = true,
        .mapMemory = true,
        .unmapMemory = true,
        .createShaderModule = true,
        .createPipelineLayout = true,
        .createComputePipelines = true,
        .createCommandPool = true,
        .allocateCommandBuffers = true,
        .beginCommandBuffer = true,
        .endCommandBuffer = true,
        .cmdBindPipeline = true,
        .cmdDispatch = true,
        .queueSubmit = true,
        .queueWaitIdle = true,
    });

    pub fn init(allocator: std.mem.Allocator) !Context {
        _ = allocator;
        // In a real scenario, we'd need to load vkGetInstanceProcAddr from the dynamic library.
        // For now, this is a structural template.
        return error.NotImplemented;
    }

    pub fn deinit(self: *Context) void {
        self.vkd.destroyDevice(self.device, null);
        self.vki.destroyInstance(self.instance, null);
    }
};
