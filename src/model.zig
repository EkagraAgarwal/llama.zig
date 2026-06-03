const std = @import("std");
const gguf = @import("gguf.zig");
const tensor = @import("tensor.zig");

pub const Architecture = enum {
    llama,
    granite,
    gemma,
    qwen,
    qwen35,
    unknown,
};

pub const Activation = enum {
    silu,
    gelu,
};

pub const ModelConfig = struct {
    arch: Architecture,
    arch_prefix: []const u8,
    activation: Activation,
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

    // Qwen 3.5 hybrid (Gated Delta Net + periodic full attention) parameters.
    ssm_d_conv: u32 = 0,
    ssm_d_inner: u32 = 0,
    ssm_d_state: u32 = 0,
    ssm_dt_rank: u32 = 0,
    ssm_n_group: u32 = 0,
    nextn_predict_layers: u32 = 0,
    full_attn_interval: u32 = 4,
    rope_sections: [4]u32 = .{ 0, 0, 0, 0 },

    pub fn isRecurrent(self: *const ModelConfig, layer: u32) bool {
        const n_main = self.n_layer -| self.nextn_predict_layers;
        if (layer >= n_main) return false;
        if (self.full_attn_interval == 0) return false;
        return (layer + 1) % self.full_attn_interval != 0;
    }

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
        // Qwen 3.5 head_dim is independent of n_embd / n_heads — prefer attention.key_length.
        const head_dim: u32 = blk: {
            if (try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "attention.key_length")) |v| {
                if (v > 0) break :blk v;
            }
            if (getU32(ctx, "llama.attention.key_length")) |v| {
                if (v > 0) break :blk v;
            }
            break :blk if (n_heads > 0) n_embd / n_heads else 0;
        };
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

        const final_logit_softcapping = (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "final_logit_softcapping")) orelse 0.0;

        const activation: Activation = if (arch == .gemma) .gelu else .silu;
        const default_embedding_scale: f32 = if (arch == .gemma) @sqrt(@as(f32, @floatFromInt(n_embd))) else 1.0;

        const embedding_scale = (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "embedding_scale")) orelse
            (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "embedding_multiplier")) orelse default_embedding_scale;
        const attention_scale = (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "attention.scale")) orelse
            (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "attention_scale")) orelse
            (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "attention_multiplier")) orelse 0.0;
        const residual_scale = (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "residual_scale")) orelse
            (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "residual_multiplier")) orelse 1.0;
        const logit_scale = (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "logit_scale")) orelse
            (try getMetaF32Any(ctx, arch_prefix, arch_alt_prefix, "logits_scaling")) orelse 1.0;

        // Qwen 3.5 Gated Delta Net + MRoPE + NextN parameters.
        // All optional; default to zero so non-Qwen35 GGUFs are unaffected.
        const ssm_d_conv: u32 = (try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "ssm.conv_kernel")) orelse 0;
        const ssm_d_inner: u32 = (try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "ssm.inner_size")) orelse 0;
        const ssm_d_state: u32 = (try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "ssm.state_size")) orelse 0;
        const ssm_dt_rank: u32 = (try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "ssm.time_step_rank")) orelse 0;
        const ssm_n_group: u32 = (try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "ssm.group_count")) orelse 0;
        const nextn_predict_layers: u32 = (try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "nextn_predict_layers")) orelse 0;
        const full_attn_interval: u32 = (try getMetaU32Any(ctx, arch_prefix, arch_alt_prefix, "full_attention_interval")) orelse 4;

        var rope_sections: [4]u32 = .{ 0, 0, 0, 0 };
        if (ctx.kvs.get("qwen35.rope.dimension_sections")) |v| {
            if (v == .array) {
                const arr = v.array;
                const n = @min(arr.len, @as(usize, 4));
                for (arr[0..n], 0..) |item, i| {
                    if (item == .u32) {
                        rope_sections[i] = item.u32;
                    } else if (item == .i32) {
                        rope_sections[i] = @intCast(@max(item.i32, 0));
                    } else if (item == .u64) {
                        rope_sections[i] = @intCast(item.u64);
                    } else if (item == .i64) {
                        rope_sections[i] = @intCast(@max(item.i64, 0));
                    }
                }
            }
        }

        return ModelConfig{
            .arch = arch,
            .arch_prefix = try allocator.dupe(u8, arch_prefix),
            .activation = activation,
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
            .ssm_d_conv = ssm_d_conv,
            .ssm_d_inner = ssm_d_inner,
            .ssm_d_state = ssm_d_state,
            .ssm_dt_rank = ssm_dt_rank,
            .ssm_n_group = ssm_n_group,
            .nextn_predict_layers = nextn_predict_layers,
            .full_attn_interval = full_attn_interval,
            .rope_sections = rope_sections,
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
    if (std.ascii.eqlIgnoreCase(raw_arch, "qwen35")) return .qwen35;
    if (raw_arch.len >= 4 and std.ascii.eqlIgnoreCase(raw_arch[0..4], "qwen")) return .qwen;
    return .unknown;
}

pub fn canonicalArchitecturePrefix(raw_arch: []const u8) []const u8 {
    return switch (parseArchitecture(raw_arch)) {
        .llama => "llama",
        .granite => "granite",
        .gemma => "gemma",
        .qwen => "qwen",
        .qwen35 => "qwen35",
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
    try t.expectEqual(Architecture.qwen35, parseArchitecture("qwen35"));
    try t.expectEqual(Architecture.qwen35, parseArchitecture("QWEN35"));
    try t.expectEqual(Architecture.unknown, parseArchitecture("mistral"));
}

test "architecture canonical metadata prefix" {
    const t = std.testing;
    try t.expectEqualStrings("llama", canonicalArchitecturePrefix("llama"));
    try t.expectEqualStrings("granite", canonicalArchitecturePrefix("granite"));
    try t.expectEqualStrings("gemma", canonicalArchitecturePrefix("gemma2"));
    try t.expectEqualStrings("qwen", canonicalArchitecturePrefix("qwen2moe"));
    try t.expectEqualStrings("qwen35", canonicalArchitecturePrefix("qwen35"));
    try t.expectEqualStrings("qwen35", canonicalArchitecturePrefix("QWEN35"));
    try t.expectEqualStrings("unknown_arch", canonicalArchitecturePrefix("unknown_arch"));
}

test "isRecurrent schedule for Qwen 3.5 n_layer=32 interval=4" {
    const t = std.testing;
    // 32 layers, 4 attention layers (3,7,11,15,19,23,27,31) and 24 SSM layers.
    var cfg = ModelConfig{
        .arch = .qwen35,
        .arch_prefix = "qwen35",
        .activation = .silu,
        .n_embd = 2560,
        .n_layer = 32,
        .n_heads = 16,
        .n_kv_heads = 4,
        .n_ff = 9216,
        .head_dim = 160,
        .vocab_size = 100,
        .max_ctx = 128,
        .rope_theta = 1.0e6,
        .rms_norm_eps = 1.0e-6,
        .wtype = .q4_k,
        .embedding_scale = 1.0,
        .attention_scale = 0.0,
        .residual_scale = 1.0,
        .logit_scale = 1.0,
        .final_logit_softcapping = 0.0,
        .full_attn_interval = 4,
    };
    var recurrent_count: u32 = 0;
    var attention_count: u32 = 0;
    var l: u32 = 0;
    while (l < 32) : (l += 1) {
        if (cfg.isRecurrent(l)) {
            recurrent_count += 1;
        } else {
            attention_count += 1;
        }
    }
    try t.expectEqual(@as(u32, 24), recurrent_count);
    try t.expectEqual(@as(u32, 8), attention_count);
    // Boundary checks: layer 0,1,2 are recurrent; layer 3 is attention.
    try t.expect(cfg.isRecurrent(0));
    try t.expect(cfg.isRecurrent(1));
    try t.expect(cfg.isRecurrent(2));
    try t.expect(!cfg.isRecurrent(3));
    try t.expect(cfg.isRecurrent(4));
    try t.expect(!cfg.isRecurrent(31));
}

test "isRecurrent returns false for NextN layers" {
    const t = std.testing;
    var cfg = ModelConfig{
        .arch = .qwen35,
        .arch_prefix = "qwen35",
        .activation = .silu,
        .n_embd = 256,
        .n_layer = 4,
        .n_heads = 4,
        .n_kv_heads = 2,
        .n_ff = 1024,
        .head_dim = 64,
        .vocab_size = 100,
        .max_ctx = 128,
        .rope_theta = 1.0e6,
        .rms_norm_eps = 1.0e-6,
        .wtype = .q4_k,
        .embedding_scale = 1.0,
        .attention_scale = 0.0,
        .residual_scale = 1.0,
        .logit_scale = 1.0,
        .final_logit_softcapping = 0.0,
        .nextn_predict_layers = 1, // last layer is NextN
        .full_attn_interval = 4,
    };
    // n_main = 3, so layer 3 is NextN — never recurrent.
    try t.expect(cfg.isRecurrent(0));
    try t.expect(!cfg.isRecurrent(3));
}

test "isRecurrent guards against zero interval" {
    const t = std.testing;
    var cfg = ModelConfig{
        .arch = .qwen35,
        .arch_prefix = "qwen35",
        .activation = .silu,
        .n_embd = 256,
        .n_layer = 4,
        .n_heads = 4,
        .n_kv_heads = 2,
        .n_ff = 1024,
        .head_dim = 64,
        .vocab_size = 100,
        .max_ctx = 128,
        .rope_theta = 1.0e6,
        .rms_norm_eps = 1.0e-6,
        .wtype = .q4_k,
        .embedding_scale = 1.0,
        .attention_scale = 0.0,
        .residual_scale = 1.0,
        .logit_scale = 1.0,
        .final_logit_softcapping = 0.0,
        .full_attn_interval = 0, // degenerate; should not divide-by-zero
    };
    try t.expect(!cfg.isRecurrent(0));
    try t.expect(!cfg.isRecurrent(1));
}

test "ModelConfig dynamic activation and scaling" {
    const t = std.testing;
    const allocator = t.allocator;

    // Use an arena allocator so all test allocations are released together
    // when the arena is destroyed. The Zig 0.16 leak detector was flagging
    // the hashmap internal-storage allocations as leaks because the deferred
    // iterator-then-deinit pattern was not running in all control paths.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Test Llama (defaults)
    {
        var kvs = std.StringHashMap(gguf.MetadataValue).init(a);
        defer kvs.deinit();
        var tensors = std.StringHashMap(*tensor.Tensor).init(a);
        defer tensors.deinit();

        var ctx = gguf.GGUFContext{
            .allocator = a,
            .file = undefined,
            .version = 3,
            .tensor_count = 0,
            .kv_count = 0,
            .kvs = kvs,
            .tensors = tensors,
            .data_offset = 0,
            .mmap_file = null,
        };
        try ctx.kvs.put(try a.dupe(u8, "general.architecture"), .{ .string = try a.dupe(u8, "llama") });
        try ctx.kvs.put(try a.dupe(u8, "llama.embedding_length"), .{ .u32 = 4096 });
        try ctx.kvs.put(try a.dupe(u8, "llama.block_count"), .{ .u32 = 32 });

        var cfg = try ModelConfig.init(a, &ctx, 32000);
        defer cfg.deinit(a);

        try t.expectEqual(Architecture.llama, cfg.arch);
        try t.expectEqual(Activation.silu, cfg.activation);
        try t.expectEqual(@as(f32, 1.0), cfg.embedding_scale);
    }

    // Test Gemma
    {
        var kvs = std.StringHashMap(gguf.MetadataValue).init(a);
        defer kvs.deinit();
        var tensors = std.StringHashMap(*tensor.Tensor).init(a);
        defer tensors.deinit();

        var ctx = gguf.GGUFContext{
            .allocator = a,
            .file = undefined,
            .version = 3,
            .tensor_count = 0,
            .kv_count = 0,
            .kvs = kvs,
            .tensors = tensors,
            .data_offset = 0,
            .mmap_file = null,
        };
        try ctx.kvs.put(try a.dupe(u8, "general.architecture"), .{ .string = try a.dupe(u8, "gemma") });
        try ctx.kvs.put(try a.dupe(u8, "gemma.embedding_length"), .{ .u32 = 2048 });

        var cfg_gemma = try ModelConfig.init(a, &ctx, 256000);
        defer cfg_gemma.deinit(a);

        try t.expectEqual(Architecture.gemma, cfg_gemma.arch);
        try t.expectEqual(Activation.gelu, cfg_gemma.activation);
        try t.expectEqual(@sqrt(@as(f32, 2048.0)), cfg_gemma.embedding_scale);
    }
}
