const std = @import("std");
const gguf = @import("gguf.zig");
const tensor = @import("tensor.zig");

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
        .q4_k => try dequantQ4K(ctx, t, dst[0..n]),
        else => return error.UnsupportedQuantType,
    }
}

fn dequantQ4K(ctx: *gguf.GGUFContext, t: *tensor.Tensor, dst: []f32) !void {
    const QK_K: usize = 256;
    const raw = try ctx.allocator.alloc(u8, t.size());
    defer ctx.allocator.free(raw);
    try ctx.readTensorData(t, raw);

    const n = dst.len;
    var row: usize = 0;
    var ib: usize = 0;
    while (row < n) : (row += QK_K) {
        if (ib + 144 > raw.len) break;
        const d = f16ToF32(std.mem.readInt(u16, raw[ib..][0..2], .little));
        const dmin = f16ToF32(std.mem.readInt(u16, raw[ib + 2 ..][0..2], .little));
        const scales = raw[ib + 4 .. ib + 16];
        const qs = raw[ib + 16 .. ib + 144];
        ib += 144;

        var is: usize = 0;
        while (is < QK_K / 64) : (is += 1) {
            const sc = scales[is];
            const dl = d * @as(f32, @floatFromInt(@as(i32, sc & 15) - 8));
            const ml = dmin * @as(f32, @floatFromInt(@as(i32, sc >> 4) - 8));
            var j: usize = 0;
            while (j < 32) : (j += 1) {
                const q_idx = is * 32 + j;
                const q_lo = (qs[q_idx] & 0x0F);
                const q_hi = (qs[q_idx] >> 4);
                const r = row + is * 64 + j;
                if (r < n) dst[r] = dl * @as(f32, @floatFromInt(q_lo)) - ml;
                if (r + 32 < n) dst[r + 32] = dl * @as(f32, @floatFromInt(q_hi)) - ml;
            }
        }
    }
}

pub fn readEmbeddingF32(ctx: *gguf.GGUFContext, t: *tensor.Tensor, token_id: u32, dst: []f32, scale: f32) !void {
    const n_embd = t.ne[0];
    if (dst.len < n_embd) return error.BufferTooSmall;

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
        else => {
            const full_row = try ctx.allocator.alloc(f32, n_embd);
            defer ctx.allocator.free(full_row);
            const tmp = try ctx.allocator.create(tensor.Tensor);
            defer ctx.allocator.destroy(tmp);
            tmp.* = t.*;
            try dequantToF32(ctx, tmp, full_row);
            @memcpy(dst[0..n_embd], full_row);
        },
    }

    if (scale != 1.0) {
        var i: usize = 0;
        while (i < n_embd) : (i += 1) dst[i] *= scale;
    }
}
