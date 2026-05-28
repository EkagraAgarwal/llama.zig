const std = @import("std");
const gguf = @import("gguf.zig");
const tensor = @import("tensor.zig");

pub fn isSupportedType(tt: tensor.Type) bool {
    return switch (tt) {
        .f32, .f16, .bf16, .q4_0, .q4_1, .q5_0, .q8_0, .q4_k, .q6_k => true,
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
        .q4_0 => try dequantQ40(ctx, t, dst[0..n]),
        .q4_1 => try dequantQ41(ctx, t, dst[0..n]),
        .q5_0 => try dequantQ50(ctx, t, dst[0..n]),
        .q8_0 => try dequantQ80(ctx, t, dst[0..n]),
        .q4_k => try dequantQ4K(ctx, t, dst[0..n]),
        .q6_k => try dequantQ6K(ctx, t, dst[0..n]),
        else => return error.UnsupportedQuantType,
    }
}

fn dequantQ40(ctx: *gguf.GGUFContext, t: *tensor.Tensor, dst: []f32) !void {
    const QK: usize = 32;
    const raw = try ctx.allocator.alloc(u8, t.size());
    defer ctx.allocator.free(raw);
    try ctx.readTensorData(t, raw);

    var out: usize = 0;
    var ib: usize = 0;
    while (out < dst.len and ib + 18 <= raw.len) {
        const d = f16ToF32(std.mem.readInt(u16, raw[ib..][0..2], .little));
        const qs = raw[ib + 2 .. ib + 18];
        ib += 18;

        var j: usize = 0;
        while (j < QK and out < dst.len) : (j += 1) {
            const q = qs[j / 2];
            const nib: i32 = if ((j & 1) == 0) @as(i32, q & 0x0F) else @as(i32, q >> 4);
            dst[out] = d * @as(f32, @floatFromInt(nib - 8));
            out += 1;
        }
    }
}

fn dequantQ50(ctx: *gguf.GGUFContext, t: *tensor.Tensor, dst: []f32) !void {
    const QK: usize = 32;
    const raw = try ctx.allocator.alloc(u8, t.size());
    defer ctx.allocator.free(raw);
    try ctx.readTensorData(t, raw);

    var out: usize = 0;
    var ib: usize = 0;
    while (out < dst.len and ib + 22 <= raw.len) {
        const d = f16ToF32(std.mem.readInt(u16, raw[ib..][0..2], .little));
        const qh = raw[ib + 2 .. ib + 6];
        const qs = raw[ib + 6 .. ib + 22];
        ib += 22;

        var j: usize = 0;
        while (j < QK and out < dst.len) : (j += 1) {
            const lo = if ((j & 1) == 0) qs[j / 2] & 0x0F else qs[j / 2] >> 4;
            const hi: u8 = (qh[j / 8] >> @as(u3, @intCast(j & 7))) & 1;
            const qv: i32 = @as(i32, (lo | (hi << 4))) - 16;
            dst[out] = d * @as(f32, @floatFromInt(qv));
            out += 1;
        }
    }
}

fn dequantQ41(ctx: *gguf.GGUFContext, t: *tensor.Tensor, dst: []f32) !void {
    const QK: usize = 32;
    const raw = try ctx.allocator.alloc(u8, t.size());
    defer ctx.allocator.free(raw);
    try ctx.readTensorData(t, raw);

    var out: usize = 0;
    var ib: usize = 0;
    while (out < dst.len and ib + 20 <= raw.len) {
        const d = f16ToF32(std.mem.readInt(u16, raw[ib..][0..2], .little));
        const m = f16ToF32(std.mem.readInt(u16, raw[ib + 2 ..][0..2], .little));
        const qs = raw[ib + 4 .. ib + 20];
        ib += 20;

        var j: usize = 0;
        while (j < QK and out < dst.len) : (j += 1) {
            const q = qs[j / 2];
            const nib: i32 = if ((j & 1) == 0) @as(i32, q & 0x0F) else @as(i32, q >> 4);
            dst[out] = d * @as(f32, @floatFromInt(nib)) + m;
            out += 1;
        }
    }
}

fn dequantQ80(ctx: *gguf.GGUFContext, t: *tensor.Tensor, dst: []f32) !void {
    const QK: usize = 32;
    const raw = try ctx.allocator.alloc(u8, t.size());
    defer ctx.allocator.free(raw);
    try ctx.readTensorData(t, raw);

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

fn dequantQ4K(ctx: *gguf.GGUFContext, t: *tensor.Tensor, dst: []f32) !void {
    const raw = try ctx.allocator.alloc(u8, t.size());
    defer ctx.allocator.free(raw);
    try ctx.readTensorData(t, raw);
    dequantQ4KRaw(raw, dst);
}

fn dequantQ4KRaw(raw: []const u8, dst: []f32) void {
    const QK_K: usize = 256;
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

fn dequantQ6K(ctx: *gguf.GGUFContext, t: *tensor.Tensor, dst: []f32) !void {
    const raw = try ctx.allocator.alloc(u8, t.size());
    defer ctx.allocator.free(raw);
    try ctx.readTensorData(t, raw);
    dequantQ6KRaw(raw, dst);
}

fn dequantQ6KRaw(raw: []const u8, dst: []f32) void {
    const QK_K: usize = 256;
    var out: usize = 0;
    var ib: usize = 0;

    while (out < dst.len and ib + 210 <= raw.len) {
        const ql = raw[ib .. ib + 128];
        const qh = raw[ib + 128 .. ib + 192];
        const sc_bytes = raw[ib + 192 .. ib + 208];
        const d = f16ToF32(std.mem.readInt(u16, raw[ib + 208 ..][0..2], .little));
        ib += 210;

        var scales: [16]i8 = undefined;
        for (0..16) |i| {
            scales[i] = @bitCast(sc_bytes[i]);
        }

        var n: usize = 0;
        while (n < QK_K and out + n < dst.len) : (n += 128) {
            var l: usize = 0;
            while (l < 32 and out + n + l + 96 < dst.len + 96) : (l += 1) {
                const is = l / 16;
                const qh_v = qh[(n / 4) + l];
                const ql0 = ql[(n / 2) + l];
                const ql1 = ql[(n / 2) + l + 32];

                const q1: i8 = @intCast(@as(i16, (ql0 & 0x0F) | (((qh_v >> 0) & 0x03) << 4)) - 32);
                const q2: i8 = @intCast(@as(i16, (ql1 & 0x0F) | (((qh_v >> 2) & 0x03) << 4)) - 32);
                const q3: i8 = @intCast(@as(i16, (ql0 >> 4) | (((qh_v >> 4) & 0x03) << 4)) - 32);
                const q4: i8 = @intCast(@as(i16, (ql1 >> 4) | (((qh_v >> 6) & 0x03) << 4)) - 32);

                const base = out + n + l;
                if (base < dst.len) dst[base] = d * @as(f32, @floatFromInt(scales[is + 0])) * @as(f32, @floatFromInt(q1));
                if (base + 32 < dst.len) dst[base + 32] = d * @as(f32, @floatFromInt(scales[is + 2])) * @as(f32, @floatFromInt(q2));
                if (base + 64 < dst.len) dst[base + 64] = d * @as(f32, @floatFromInt(scales[is + 4])) * @as(f32, @floatFromInt(q3));
                if (base + 96 < dst.len) dst[base + 96] = d * @as(f32, @floatFromInt(scales[is + 6])) * @as(f32, @floatFromInt(q4));
            }
        }
        out += QK_K;
    }
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
        .q4_k => {
            const row_bytes = quantRowBytes(t.type, n_embd) orelse return error.UnsupportedQuantType;
            const raw = try ctx.allocator.alloc(u8, row_bytes);
            defer ctx.allocator.free(raw);
            const off = t.offset + @as(u64, token_id) * row_bytes;
            _ = try ctx.file.readPositionalAll(std.Io.Threaded.global_single_threaded.io(), raw, ctx.data_offset + off);
            dequantQ4KRaw(raw, dst[0..n_embd]);
        },
        .q6_k => {
            const row_bytes = quantRowBytes(t.type, n_embd) orelse return error.UnsupportedQuantType;
            const raw = try ctx.allocator.alloc(u8, row_bytes);
            defer ctx.allocator.free(raw);
            const off = t.offset + @as(u64, token_id) * row_bytes;
            _ = try ctx.file.readPositionalAll(std.Io.Threaded.global_single_threaded.io(), raw, ctx.data_offset + off);
            dequantQ6KRaw(raw, dst[0..n_embd]);
        },
        .q4_0 => {
            const row_bytes = quantRowBytes(t.type, n_embd) orelse return error.UnsupportedQuantType;
            const raw = try ctx.allocator.alloc(u8, row_bytes);
            defer ctx.allocator.free(raw);
            const off = t.offset + @as(u64, token_id) * row_bytes;
            _ = try ctx.file.readPositionalAll(std.Io.Threaded.global_single_threaded.io(), raw, ctx.data_offset + off);
            dequantQ40Raw(raw, dst[0..n_embd]);
        },
        .q4_1 => {
            const row_bytes = quantRowBytes(t.type, n_embd) orelse return error.UnsupportedQuantType;
            const raw = try ctx.allocator.alloc(u8, row_bytes);
            defer ctx.allocator.free(raw);
            const off = t.offset + @as(u64, token_id) * row_bytes;
            _ = try ctx.file.readPositionalAll(std.Io.Threaded.global_single_threaded.io(), raw, ctx.data_offset + off);
            dequantQ41Raw(raw, dst[0..n_embd]);
        },
        .q8_0 => {
            const row_bytes = quantRowBytes(t.type, n_embd) orelse return error.UnsupportedQuantType;
            const raw = try ctx.allocator.alloc(u8, row_bytes);
            defer ctx.allocator.free(raw);
            const off = t.offset + @as(u64, token_id) * row_bytes;
            _ = try ctx.file.readPositionalAll(std.Io.Threaded.global_single_threaded.io(), raw, ctx.data_offset + off);
            dequantQ80Raw(raw, dst[0..n_embd]);
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

pub fn quantRowBytes(tt: tensor.Type, n_embd: u64) ?usize {
    const blk = tt.blockSize();
    if (blk <= 1) return null;
    const blocks = (n_embd + blk - 1) / blk;
    return @intCast(blocks * tt.bytesPerBlock());
}

fn dequantQ40Raw(raw: []const u8, dst: []f32) void {
    const QK: usize = 32;
    var out: usize = 0;
    var ib: usize = 0;
    while (out < dst.len and ib + 18 <= raw.len) {
        const d = f16ToF32(std.mem.readInt(u16, raw[ib..][0..2], .little));
        const qs = raw[ib + 2 .. ib + 18];
        ib += 18;
        var j: usize = 0;
        while (j < QK and out < dst.len) : (j += 1) {
            const q = qs[j / 2];
            const nib: i32 = if ((j & 1) == 0) @as(i32, q & 0x0F) else @as(i32, q >> 4);
            dst[out] = d * @as(f32, @floatFromInt(nib - 8));
            out += 1;
        }
    }
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

fn dequantQ41Raw(raw: []const u8, dst: []f32) void {
    const QK: usize = 32;
    var out: usize = 0;
    var ib: usize = 0;
    while (out < dst.len and ib + 20 <= raw.len) {
        const d = f16ToF32(std.mem.readInt(u16, raw[ib..][0..2], .little));
        const m = f16ToF32(std.mem.readInt(u16, raw[ib + 2 ..][0..2], .little));
        const qs = raw[ib + 4 .. ib + 20];
        ib += 20;
        var j: usize = 0;
        while (j < QK and out < dst.len) : (j += 1) {
            const q = qs[j / 2];
            const nib: i32 = if ((j & 1) == 0) @as(i32, q & 0x0F) else @as(i32, q >> 4);
            dst[out] = d * @as(f32, @floatFromInt(nib)) + m;
            out += 1;
        }
    }
}

test "supported quant type matrix includes q4_k for q4_k_m models" {
    try std.testing.expect(isSupportedType(.q4_k));
    try std.testing.expect(isSupportedType(.q6_k));
    try std.testing.expect(isSupportedType(.q4_0));
    try std.testing.expect(isSupportedType(.q4_1));
    try std.testing.expect(isSupportedType(.q5_0));
    try std.testing.expect(isSupportedType(.q8_0));
}

test "q6_k dequant raw block shape" {
    var raw: [210]u8 = [_]u8{0} ** 210;
    // d = 1.0
    std.mem.writeInt(u16, raw[208..210], @as(u16, 0x3c00), .little);
    // unit scales
    for (192..208) |i| raw[i] = @bitCast(@as(i8, 1));
    // ql/qh left as zero -> quant value maps to -32

    var out: [256]f32 = undefined;
    dequantQ6KRaw(&raw, &out);
    try std.testing.expectApproxEqAbs(@as(f32, -32.0), out[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -32.0), out[255], 0.001);
}
