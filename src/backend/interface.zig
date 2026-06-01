const std = @import("std");
pub const vulkan = @import("vulkan.zig");

// Compile-time backend selection. Currently only Vulkan is implemented.
pub const Backend = vulkan.VulkanBackend;
pub const Buffer = vulkan.Buffer;
