const std = @import("std");
const gguf = @import("gguf.zig");
const tensor = @import("tensor.zig");

pub fn isSupportedType(tt: tensor.Type) bool {
    return switch (tt) {
        .f32, .f16, .bf16, .q8_0 => true,
        else => false,
    };
}

pub fn typeName(tt: tensor.Type) []const u8 {
    return @tagName(tt);
}

pub fn f16ToF32(bits: u16) f32 {
    const sign: u32 = (@as(u32, bits) >> 15) & 1;
    const exp: u32 = (@as(u32, bits) >> 10) & 0x1F;
    const mant: u32 = @as(u32, bits) & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return @as(f32, @bitCast(sign << 31));
        return @as(f32, @bitCast(sign << 31)) + @as(f32, @floatFromInt(mant)) * std.math.pow(f32, 2, -24);
    }
    if (exp == 31) return @as(f32, @bitCast((sign << 31) | 0x7F800000 | (mant << 13)));
    return @as(f32, @bitCast((sign << 31) | ((exp + 112) << 23) | (mant << 13)));
}

pub fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// Dequantize GGUF tensor data into an f32 buffer (host).
pub fn dequantToF32(ctx: *gguf.GGUFContext, t: *tensor.Tensor, dst: []f32) !void {
    const n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
    if (dst.len < n) return error.BufferTooSmall;

    switch (t.type) {
        .f32 => {
            var raw = try ctx.allocator.alloc(u8, t.size());
            defer ctx.allocator.free(raw);
            try ctx.readTensorData(t, raw);
            @memcpy(std.mem.sliceAsBytes(dst[0..n]), raw[0 .. n * 4]);
        },
        .f16 => {
            var raw = try ctx.allocator.alloc(u8, t.size());
            defer ctx.allocator.free(raw);
            try ctx.readTensorData(t, raw);
            const src = std.mem.bytesAsSlice(u16, raw[0 .. n * 2]);
            for (src, 0..) |bits, i| dst[i] = f16ToF32(bits);
        },
        .bf16 => {
            var raw = try ctx.allocator.alloc(u8, t.size());
            defer ctx.allocator.free(raw);
            try ctx.readTensorData(t, raw);
            const src = std.mem.bytesAsSlice(u16, raw[0 .. n * 2]);
            for (src, 0..) |bits, i| dst[i] = bf16ToF32(bits);
        },
        .q8_0 => try dequantQ80(ctx, t, dst[0..n]),
    }
}

fn dequantQ80(ctx: *gguf.GGUFContext, t: *tensor.Tensor, dst: []f32) !void {
    const raw = try ctx.allocator.alloc(u8, t.size());
    defer ctx.allocator.free(raw);
    try ctx.readTensorData(t, raw);
    dequantQ80Raw(raw, dst);
}

pub fn readEmbeddingF32(ctx: *gguf.GGUFContext, t: *tensor.Tensor, token_id: u32, dst: []f32, n_embd: u32, scale: f32) !void {
    if (dst.len < n_embd) return error.BufferTooSmall;
    if (t.ne[0] != n_embd) return error.UnsupportedEmbeddingLayout;

    switch (t.type) {
        .f32 => {
            const row_bytes = n_embd * 4;
            var raw: [8192]u8 = undefined;
            if (row_bytes > raw.len) {
                const heap = try ctx.allocator.alloc(u8, row_bytes);
                defer ctx.allocator.free(heap);
                const off = t.offset + @as(u64, token_id) * row_bytes;
                _ = try ctx.file.readPositionalAll(std.Io.Threaded.global_single_threaded.io(), heap, ctx.data_offset + off);
                @memcpy(std.mem.sliceAsBytes(dst[0..n_embd]), heap);
            } else {
                const off = t.offset + @as(u64, token_id) * row_bytes;
                _ = try ctx.file.readPositionalAll(std.Io.Threaded.global_single_threaded.io(), raw[0..row_bytes], ctx.data_offset + off);
                @memcpy(std.mem.sliceAsBytes(dst[0..n_embd]), raw[0..row_bytes]);
            }
        },
        .bf16 => {
            const row_bytes = n_embd * 2;
            const raw = try ctx.allocator.alloc(u8, row_bytes);
            defer ctx.allocator.free(raw);
            const off = t.offset + @as(u64, token_id) * row_bytes;
            _ = try ctx.file.readPositionalAll(std.Io.Threaded.global_single_threaded.io(), raw, ctx.data_offset + off);
            const src = std.mem.bytesAsSlice(u16, raw);
            for (src, 0..) |bits, i| dst[i] = bf16ToF32(bits);
        },
        .q8_0 => {
            const row_bytes = quantRowBytes(t.type, n_embd) orelse return error.UnsupportedQuantType;
            const raw = try ctx.allocator.alloc(u8, row_bytes);
            defer ctx.allocator.free(raw);
            const off = t.offset + @as(u64, token_id) * row_bytes;
            _ = try ctx.file.readPositionalAll(std.Io.Threaded.global_single_threaded.io(), raw, ctx.data_offset + off);
            dequantQ80Raw(raw, dst[0..n_embd]);
        },
        else => return error.UnsupportedQuantType,
    }

    if (scale != 1.0) {
        var i: usize = 0;
        while (i < n_embd) : (i += 1) dst[i] *= scale;
    }
}

pub fn quantRowBytes(tt: tensor.Type, n_embd: u64) ?usize {
    const blk = tt.blockSize();
    if (blk <= 1) return null;
    const blocks = (n_embd + blk - 1) / blk;
    return @intCast(blocks * tt.bytesPerBlock());
}

fn dequantQ80Raw(raw: []const u8, dst: []f32) void {
    const QK: usize = 32;
    var out: usize = 0;
    var ib: usize = 0;
    while (out < dst.len and ib + 34 <= raw.len) {
        const d = f16ToF32(std.mem.readInt(u16, raw[ib..][0..2], .little));
        const qs = raw[ib + 2 .. ib + 34];
        ib += 34;
        var j: usize = 0;
        while (j < QK and out < dst.len) : (j += 1) {
            const v: i8 = @bitCast(qs[j]);
            dst[out] = d * @as(f32, @floatFromInt(v));
            out += 1;
        }
    }
}

test "q8_0 dequant raw block shape" {
    var raw: [34]u8 = [_]u8{0} ** 34;
    std.mem.writeInt(u16, raw[0..2], @as(u16, 0x3c00), .little);
    for (0..32) |i| raw[2 + i] = @bitCast(@as(i8, i % 256));
    var out: [32]f32 = undefined;
    dequantQ80Raw(&raw, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out[2], 0.001);
}
