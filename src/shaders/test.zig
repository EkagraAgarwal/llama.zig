pub const SPIRV_CAPABILITIES = [_]u32{
    1, // Shader
};

export fn test_all(a: [*]addrspace(.global) f32) void {
    const gid = @workGroupId(0);
    a[gid] = 1.0;
}
