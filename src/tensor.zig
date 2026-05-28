const std = @import("std");

pub const Type = enum(u32) {
    f32 = 0,
    f16 = 1,
    q8_0 = 8,
    bf16 = 30,

    pub fn sizeOf(self: Type) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            else => 1,
        };
    }

    pub fn blockSize(self: Type) usize {
        return switch (self) {
            .q8_0 => 32,
            else => 1,
        };
    }

    pub fn bytesPerBlock(self: Type) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            .q8_0 => 34,
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
