const std = @import("std");

pub const Type = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    // Add other types as needed
    // ...
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    i32 = 14,
    f64 = 15,
    i8 = 16,
    i16 = 17,
    i64 = 18,
    
    pub fn sizeOf(self: Type) usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .i16 => 2,
            .i8 => 1,
            .f64, .i64 => 8,
            else => @panic("Unsupported type size requested"),
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

    pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.destroy(self);
    }
};
