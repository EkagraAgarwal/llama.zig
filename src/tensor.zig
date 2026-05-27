const std = @import("std");

pub const Type = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    // k-quants
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    i32 = 16,
    i16 = 17,
    i8 = 18,
    f64 = 19,
    i64 = 20,
    bf16 = 30,
    
    pub fn sizeOf(self: Type) usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .i16, .bf16 => 2,
            .i8 => 1,
            .f64, .i64 => 8,
            else => 0, // quantized types have complex sizing
        };
    }

    pub fn blockSize(self: Type) usize {
        return switch (self) {
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
            else => 1,
        };
    }

    /// Bytes per block for quantized types, or bytes per element for scalar types.
    pub fn bytesPerBlock(self: Type) usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .i16, .bf16 => 2,
            .i8 => 1,
            .f64, .i64 => 8,
            .q4_0 => 18,   // 2 + 16
            .q4_1 => 20,   // 4 + 16
            .q5_0 => 22,   // 2 + 4 + 16
            .q5_1 => 24,   // 4 + 4 + 16
            .q8_0 => 34,   // 2 + 32
            .q8_1 => 36,   // 4 + 32
            .q2_k => 84,   // 256 elements in 84 bytes
            .q3_k => 110,  // 256 elements in 110 bytes
            .q4_k => 144,  // 256 elements in 144 bytes
            .q5_k => 176,  // 256 elements in 176 bytes
            .q6_k => 210,  // 256 elements in 210 bytes
            .q8_k => 292,  // 256 elements in 292 bytes
        };
    }
};

pub const Tensor = struct {
    name: []const u8,
    type: Type,
    n_dims: u32,
    ne: [4]u64, // number of elements in each dimension
    nb: [4]u64, // stride in bytes
    offset: u64, // offset in the data section
    data: ?*anyopaque,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8, tensor_type: Type, dims: []const u64) !*Tensor {
        const tensor = try allocator.create(Tensor);
        
        var ne: [4]u64 = .{ 1, 1, 1, 1 };
        var nb: [4]u64 = .{ 0, 0, 0, 0 };
        
        for (dims, 0..) |dim, i| {
            if (i < 4) ne[i] = dim;
        }
        
        nb[0] = tensor_type.sizeOf();
        nb[1] = nb[0] * ne[0];
        nb[2] = nb[1] * ne[1];
        nb[3] = nb[2] * ne[2];
        
        tensor.* = .{
            .name = try allocator.dupe(u8, name),
            .type = tensor_type,
            .n_dims = @as(u32, @intCast(dims.len)),
            .ne = ne,
            .nb = nb,
            .offset = 0,
            .data = null,
        };
        return tensor;
    }

    pub fn size(self: *const Tensor) u64 {
        const n_elems = self.ne[0] * self.ne[1] * self.ne[2] * self.ne[3];
        const blk = self.type.blockSize();
        if (blk > 1) {
            // Quantized: size = ceil(n_elems / blockSize) * bytesPerBlock
            const n_blocks = (n_elems + blk - 1) / blk;
            return n_blocks * self.type.bytesPerBlock();
        }
        return n_elems * self.type.bytesPerBlock();
    }

    pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.destroy(self);
    }
};
