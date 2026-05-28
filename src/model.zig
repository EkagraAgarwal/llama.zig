const std = @import("std");
const gguf = @import("gguf.zig");
const tensor = @import("tensor.zig");

pub const Architecture = enum {
    llama,
    granite,
    gemma,
    qwen,
    unknown,
};

pub const ModelConfig = struct {
    arch: Architecture,
    arch_prefix: []const u8,
    n_embd: u32,
    n_layer: u32,
    n_heads: u32,
    n_kv_heads: u32,
    n_ff: u32,
    head_dim: u32,
    vocab_size: u32,
    max_ctx: u32,
    rope_theta: f32,
    rms_norm_eps: f32,
    wtype: tensor.Type,
    embedding_scale: f32,
    attention_scale: f32,
    residual_scale: f32,
    logit_scale: f32,
    final_logit_softcapping: f32,

    pub fn init(allocator: std.mem.Allocator, ctx: *gguf.GGUFContext, vocab_size: u32) !ModelConfig {
        const arch_str: []const u8 = blk: {
            if (ctx.kvs.get("general.architecture")) |v| {
                break :blk switch (v) {
                    .string => |s| s,
                    else => "llama",
                };
            }
            break :blk "llama";
        };

        const arch = parseArchitecture(arch_str);
        const arch_prefix = arch_str;
        const arch_alt_prefix = canonicalArchitecturePrefix(arch_str);

        const n_embd = try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "embedding_length") orelse
            getU32(ctx, "llama.embedding_length") orelse 768;
        const n_layer = try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "block_count") orelse
            getU32(ctx, "llama.block_count") orelse 1;
        const n_heads = try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "attention.head_count") orelse
            getU32(ctx, "llama.attention.head_count") orelse 12;
        const head_dim = n_embd / n_heads;
        const n_kv_heads = blk: {
            if (try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "attention.head_count_kv")) |v| break :blk v;
            if (getU32(ctx, "llama.attention.head_count_kv")) |v| break :blk v;
            // Derive from the actual K projection tensor shape (GQA support).
            if (ctx.tensors.get("blk.0.attn_k.weight")) |t| {
                const kv_out = @as(u32, @intCast(t.ne[1]));
                if (head_dim > 0 and kv_out % head_dim == 0) break :blk kv_out / head_dim;
            }
            break :blk n_heads;
        };
        const n_ff = try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "feed_forward_length") orelse
            getU32(ctx, "llama.feed_forward_length") orelse (n_embd * 8) / 3;

        const rope_theta: f32 = blk: {
            if (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "rope.freq_base")) |v| break :blk v;
            if (getF32(ctx, "llama.rope.freq_base")) |v| break :blk v;
            break :blk if (arch == .llama and n_embd >= 2048) 500000.0 else 10000.0;
        };

        const rms_norm_eps: f32 = blk: {
            if (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "attention.layer_norm_rms_epsilon")) |v| break :blk v;
            if (getF32(ctx, "llama.attention.layer_norm_rms_epsilon")) |v| break :blk v;
            break :blk 1e-5;
        };

        const max_ctx = try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "context_length") orelse
            getU32(ctx, "llama.context_length") orelse 2048;

        const wtype: tensor.Type = blk: {
            if (ctx.tensors.get("token_embd.weight")) |t| break :blk t.type;
            break :blk .f32;
        };

        const embedding_scale = (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "embedding_scale")) orelse
            (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "embedding_multiplier")) orelse 1.0;
        const attention_scale = (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "attention.scale")) orelse
            (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "attention_scale")) orelse
            (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "attention_multiplier")) orelse 0.0;
        const residual_scale = (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "residual_scale")) orelse
            (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "residual_multiplier")) orelse 1.0;
        const logit_scale = (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "logit_scale")) orelse
            (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "logits_scaling")) orelse 1.0;
        const final_logit_softcapping = (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "final_logit_softcapping")) orelse 0.0;

        return ModelConfig{
            .arch = arch,
            .arch_prefix = try allocator.dupe(u8, arch_prefix),
            .n_embd = n_embd,
            .n_layer = n_layer,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .n_ff = n_ff,
            .head_dim = head_dim,
            .vocab_size = vocab_size,
            .max_ctx = max_ctx,
            .rope_theta = rope_theta,
            .rms_norm_eps = rms_norm_eps,
            .wtype = wtype,
            .embedding_scale = embedding_scale,
            .attention_scale = attention_scale,
            .residual_scale = residual_scale,
            .logit_scale = logit_scale,
            .final_logit_softcapping = final_logit_softcapping,
        };
    }

    pub fn deinit(self: *ModelConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.arch_prefix);
    }
};

pub fn parseArchitecture(raw_arch: []const u8) Architecture {
    if (std.ascii.eqlIgnoreCase(raw_arch, "llama")) return .llama;
    if (std.ascii.eqlIgnoreCase(raw_arch, "granite")) return .granite;
    if (raw_arch.len >= 5 and std.ascii.eqlIgnoreCase(raw_arch[0..5], "gemma")) return .gemma;
    if (raw_arch.len >= 4 and std.ascii.eqlIgnoreCase(raw_arch[0..4], "qwen")) return .qwen;
    return .unknown;
}

pub fn canonicalArchitecturePrefix(raw_arch: []const u8) []const u8 {
    return switch (parseArchitecture(raw_arch)) {
        .llama => "llama",
        .granite => "granite",
        .gemma => "gemma",
        .qwen => "qwen",
        .unknown => raw_arch,
    };
}

fn getMetaU32(ctx: *gguf.GGUFContext, prefix: []const u8, suffix: []const u8) !?u32 {
    var buf: [128]u8 = undefined;
    const key = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, suffix });
    return getU32(ctx, key);
}

fn getMetaF32(ctx: *gguf.GGUFContext, prefix: []const u8, suffix: []const u8) !?f32 {
    var buf: [128]u8 = undefined;
    const key = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, suffix });
    return getF32(ctx, key);
}

fn getMetaU32Any(ctx: *gguf.GGUFContext, primary: []const u8, alt: []const u8, suffix: []const u8) !?u32 {
    if (try getMetaU32(ctx, primary, suffix)) |v| return v;
    if (!std.mem.eql(u8, primary, alt)) return try getMetaU32(ctx, alt, suffix);
    return null;
}

fn getMetaF32Any(ctx: *gguf.GGUFContext, primary: []const u8, alt: []const u8, suffix: []const u8) !?f32 {
    if (try getMetaF32(ctx, primary, suffix)) |v| return v;
    if (!std.mem.eql(u8, primary, alt)) return try getMetaF32(ctx, alt, suffix);
    return null;
}

pub fn getU32(ctx: *gguf.GGUFContext, key: []const u8) ?u32 {
    const val = ctx.kvs.get(key) orelse return null;
    return switch (val) {
        .u8 => |v| @as(u32, v),
        .u16 => |v| @as(u32, v),
        .u32 => |v| v,
        .u64 => |v| @as(u32, @intCast(v)),
        .i32 => |v| @as(u32, @intCast(v)),
        .i64 => |v| @as(u32, @intCast(v)),
        else => null,
    };
}

pub fn getF32(ctx: *gguf.GGUFContext, key: []const u8) ?f32 {
    const val = ctx.kvs.get(key) orelse return null;
    return switch (val) {
        .f32 => |v| v,
        .f64 => |v| @floatCast(v),
        .u32 => |v| @floatFromInt(v),
        .u64 => |v| @floatFromInt(v),
        .i32 => |v| @floatFromInt(v),
        else => null,
    };
}

pub fn f32Bytes(count: u64) u64 {
    return count * 4;
}

pub fn weightF32Size(t: *tensor.Tensor) u64 {
    const n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
    return f32Bytes(n);
}

test "architecture acceptance matrix parsing" {
    const t = std.testing;
    try t.expectEqual(Architecture.llama, parseArchitecture("llama"));
    try t.expectEqual(Architecture.granite, parseArchitecture("granite"));
    try t.expectEqual(Architecture.gemma, parseArchitecture("gemma"));
    try t.expectEqual(Architecture.gemma, parseArchitecture("gemma2"));
    try t.expectEqual(Architecture.qwen, parseArchitecture("qwen"));
    try t.expectEqual(Architecture.qwen, parseArchitecture("qwen2"));
    try t.expectEqual(Architecture.qwen, parseArchitecture("qwen2moe"));
    try t.expectEqual(Architecture.unknown, parseArchitecture("mistral"));
}

test "architecture canonical metadata prefix" {
    const t = std.testing;
    try t.expectEqualStrings("llama", canonicalArchitecturePrefix("llama"));
    try t.expectEqualStrings("granite", canonicalArchitecturePrefix("granite"));
    try t.expectEqualStrings("gemma", canonicalArchitecturePrefix("gemma2"));
    try t.expectEqualStrings("qwen", canonicalArchitecturePrefix("qwen2moe"));
    try t.expectEqualStrings("unknown_arch", canonicalArchitecturePrefix("unknown_arch"));
}
