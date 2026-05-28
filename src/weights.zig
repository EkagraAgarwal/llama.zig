const std = @import("std");
const gguf = @import("gguf.zig");
const tensor = @import("tensor.zig");

pub fn isSupportedType(tt: tensor.Type) bool {
    return switch (tt) {
        .f32, .f16, .bf16, .q8_0, .q4_0, .q4_1, .q6_k => true,
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
        .q4_0 => try dequantQ40(ctx, t, dst[0..n]),
        .q4_1 => try dequantQ41(ctx, t, dst[0..n]),
        .q6_k => try dequantQ6K(ctx, t, dst[0..n]),
    }
}

fn dequantQ80(ctx: *gguf.GGUFContext, t: *tensor.Tensor, dst: []f32) !void {
    const raw = try ctx.allocator.alloc(u8, t.size());
    defer ctx.allocator.free(raw);
    try ctx.readTensorData(t, raw);
    dequantQ80Raw(raw, dst);
}

fn dequantQ40(ctx: *gguf.GGUFContext, t: *tensor.Tensor, dst: []f32) !void {
    const raw = try ctx.allocator.alloc(u8, t.size());
    defer ctx.allocator.free(raw);
    try ctx.readTensorData(t, raw);
    dequantQ40Raw(raw, dst);
}

fn dequantQ41(ctx: *gguf.GGUFContext, t: *tensor.Tensor, dst: []f32) !void {
    const raw = try ctx.allocator.alloc(u8, t.size());
    defer ctx.allocator.free(raw);
    try ctx.readTensorData(t, raw);
    dequantQ41Raw(raw, dst);
}

fn dequantQ6K(ctx: *gguf.GGUFContext, t: *tensor.Tensor, dst: []f32) !void {
    const raw = try ctx.allocator.alloc(u8, t.size());
    defer ctx.allocator.free(raw);
    try ctx.readTensorData(t, raw);
    dequantQ6KRaw(raw, dst);
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
        .q6_k => {
            const row_bytes = quantRowBytes(t.type, n_embd) orelse return error.UnsupportedQuantType;
            const raw = try ctx.allocator.alloc(u8, row_bytes);
            defer ctx.allocator.free(raw);
            const off = t.offset + @as(u64, token_id) * row_bytes;
            _ = try ctx.file.readPositionalAll(std.Io.Threaded.global_single_threaded.io(), raw, ctx.data_offset + off);
            dequantQ6KRaw(raw, dst[0..n_embd]);
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

fn dequantQ40Raw(raw: []const u8, dst: []f32) void {
    const QK: usize = 32;
    const BPR: usize = 18;
    var out: usize = 0;
    var ib: usize = 0;
    while (out < dst.len and ib + BPR <= raw.len) {
        const d = f16ToF32(std.mem.readInt(u16, raw[ib..][0..2], .little));
        ib += 2;
        var j: usize = 0;
        while (j < QK and out < dst.len) : (j += 1) {
            const qb: u8 = raw[ib + j / 2];
            const nibble: u4 = @truncate(qb >> (4 * @as(u3, @intCast(j & 1))));
            const qv: i32 = @as(i32, @intCast(nibble)) - 8;
            dst[out] = d * @as(f32, @floatFromInt(qv));
            out += 1;
        }
        ib += 16;
    }
}

fn dequantQ41Raw(raw: []const u8, dst: []f32) void {
    const QK: usize = 32;
    const BPR: usize = 20;
    var out: usize = 0;
    var ib: usize = 0;
    while (out < dst.len and ib + BPR <= raw.len) {
        const d = f16ToF32(std.mem.readInt(u16, raw[ib..][0..2], .little));
        const m = f16ToF32(std.mem.readInt(u16, raw[ib + 2 ..][0..2], .little));
        ib += 4;
        var j: usize = 0;
        while (j < QK and out < dst.len) : (j += 1) {
            const qb: u8 = raw[ib + j / 2];
            const qv: u8 = (qb >> (4 * @as(u3, @intCast(j & 1)))) & 0x0F;
            dst[out] = d * @as(f32, @floatFromInt(qv)) + m;
            out += 1;
        }
        ib += 16;
    }
}

fn dequantQ6KRaw(raw: []const u8, dst: []f32) void {
    const QK: usize = 256;
    const BPR: usize = 210;
    var out: usize = 0;
    var ib: usize = 0;
    while (out < dst.len and ib + BPR <= raw.len) {
        const d = f16ToF32(std.mem.readInt(u16, raw[ib..][0..2], .little));
        var sc_off: usize = ib + 2;
        var ql_off: usize = ib + 18;
        var qh_off: usize = ib + 146;

        var n: usize = 0;
        while (n < QK and out < dst.len) : (n += 128) {
            var l: usize = 0;
            while (l < 32) : (l += 1) {
                if (out + l + 128 >= dst.len) return;
                const is: usize = (l / 16) * 4;
                const scale_base: usize = sc_off + is;

                const sc0: f32 = @as(f32, @floatFromInt(@as(i8, @bitCast(raw[scale_base + 0]))));
                const sc1: f32 = @as(f32, @floatFromInt(@as(i8, @bitCast(raw[scale_base + 2]))));
                const sc2: f32 = @as(f32, @floatFromInt(@as(i8, @bitCast(raw[scale_base + 1]))));
                const sc3: f32 = @as(f32, @floatFromInt(@as(i8, @bitCast(raw[scale_base + 3]))));

                const qb0: u8 = raw[ql_off + l];
                const qb1: u8 = raw[ql_off + l + 32];
                const qh_b: u8 = raw[qh_off + l];

                const q1: i8 = @as(i8, @bitCast((qb0 & 0xF) | ((qh_b & 0x03) << 4))) - 32;
                const q2: i8 = @as(i8, @bitCast((qb0 >> 4) | (((qh_b >> 2) & 0x03) << 4))) - 32;
                const q3: i8 = @as(i8, @bitCast((qb1 & 0xF) | (((qh_b >> 4) & 0x03) << 4))) - 32;
                const q4: i8 = @as(i8, @bitCast((qb1 >> 4) | (((qh_b >> 6) & 0x03) << 4))) - 32;

                dst[out + l + 0] = d * sc0 * @as(f32, @floatFromInt(q1));
                dst[out + l + 32] = d * sc1 * @as(f32, @floatFromInt(q2));
                dst[out + l + 64] = d * sc2 * @as(f32, @floatFromInt(q3));
                dst[out + l + 96] = d * sc3 * @as(f32, @floatFromInt(q4));
            }
            out += 128;
            sc_off += 8;
            ql_off += 64;
            qh_off += 32;
        }
        ib += BPR;
    }
}

test "q8_0 dequant raw block shape" {
    var raw: [34]u8 = [_]u8{0} ** 34;
    std.mem.writeInt(u16, raw[0..2], @as(u16, 0x3c00), .little);
    raw[2] = 0x01;
    raw[3] = 0x01;
    raw[4] = 0xFF;
    var out: [32]f32 = undefined;
    dequantQ80Raw(&raw, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out[2], 0.001);
}

test "q4_0 dequant raw block shape" {
    var raw: [18]u8 = [_]u8{0} ** 18;
    std.mem.writeInt(u16, raw[0..2], @as(u16, 0x3c00), .little);
    raw[2] = 0x78;
    raw[3] = 0x01;
    var out: [32]f32 = undefined;
    dequantQ40Raw(&raw, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out[1], 0.001);
}
