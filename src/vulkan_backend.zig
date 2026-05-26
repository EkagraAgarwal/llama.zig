const std = @import("std");
const vk = @import("vulkan");
const builtin = @import("builtin");

/// Manages the Vulkan instance, physical device, logical device, and compute queues.
pub const Context = struct {
    allocator: std.mem.Allocator,
    instance: vk.Instance,
    pdev: vk.PhysicalDevice,
    device: vk.Device,
    compute_queue: vk.Queue,
    compute_family_index: u32,

    vki: vk.InstanceWrapper,
    vkd: vk.DeviceWrapper,

    pub fn init(allocator: std.mem.Allocator) !Context {
        _ = allocator;
        return error.NotImplemented;
    }

    pub fn deinit(self: *Context) void {
        self.vkd.destroyDevice(self.device, null);
        self.vki.destroyInstance(self.instance, null);
    }
};
