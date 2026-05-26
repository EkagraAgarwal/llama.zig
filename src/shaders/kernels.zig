const std = @import("std");
const gpu = std.gpu;

pub const SPIRV_CAPABILITIES = [_]u32{
    1, // Shader
};

pub const SPIRV_EXTENSIONS = [_][]const u8{
    "SPV_KHR_storage_buffer_storage_class",
};

// Element-wise addition
export fn add(
    a: [*]addrspace(.global) const f32,
    b: [*]addrspace(.global) const f32,
    c: [*]addrspace(.global) f32,
) callconv(.spirv_kernel) void {
    const gid = gpu.global_invocation_id[0];
    c[gid] = a[gid] + b[gid];
}
