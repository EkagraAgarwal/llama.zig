//! Qwen 3.5 hybrid model graph builder.
//!
//! Qwen 3.5 is a hybrid architecture alternating between:
//!   - Full attention layers (Qwen3-style: Q/K RMS norm, MRoPE, joint Q+gate
//!     projection, sigmoid-gated attention output) — every Nth layer where
//!     N is `full_attn_interval` (default 4).
//!   - Linear-attention "Gated Delta Net" layers for the rest. The
//!     recurrent state lives on the CPU between decode steps; the
//!     prefill-step GDN runs on GPU.
//!
//! After each attention/SSM block, the residual is added, then
//! `post_attention_norm` → dense FFN (gate, up, down, silu_mul) → residual.
//!
//! Optionally followed by NextN/MTP decoder blocks (when
//! `nextn_predict_layers > 0`); the MTP block is registered with the
//! graph but is NOT executed by the main forward pass in this port
//! (speculative-decoding is out of scope for this pass).

const std = @import("std");
const compute_graph = @import("compute_graph.zig");
const model = @import("model.zig");
const tensor = @import("tensor.zig");
const GraphBuilder = compute_graph.GraphBuilder;
const OpType = compute_graph.OpType;

fn f32Size(n: u32) u64 {
    return @as(u64, n) * 4;
}

fn build_lm_head(
    builder: *GraphBuilder,
    cfg: *const model.ModelConfig,
    in_name: []const u8,
    logits_name: []const u8,
    has_output_weight: bool,
) !void {
    const eps_bits: u32 = @bitCast(cfg.rms_norm_eps);

    const norm_w = "output_norm.weight";
    const normed = "final.normed";
    _ = try builder.add_tensor(norm_w, f32Size(cfg.n_embd), .weight);
    _ = try builder.add_tensor(normed, f32Size(cfg.n_embd), .activation);
    try builder.add_node(.rms_norm, &.{ in_name, norm_w }, normed, (cfg.n_embd + 63) / 64, 1, cfg.n_embd, cfg.n_embd, eps_bits, 0);

    const out_w = if (has_output_weight) "output.weight" else "token_embd.weight";
    const out_dims = builder.matmulDims(out_w, cfg.vocab_size, cfg.n_embd);
    _ = try builder.add_tensor(out_w, f32Size(out_dims.out * out_dims.in), .weight);

    _ = try builder.add_tensor(logits_name, f32Size(out_dims.out), .activation);
    try builder.add_nodeP(.matmul, &.{ normed, out_w }, logits_name, (out_dims.out + 15) / 16, 1, 1, out_dims.out, out_dims.in, 0, 0);

    const out_bias = "output.bias";
    if (builder.has_tensor(out_bias)) {
        _ = try builder.add_tensor(out_bias, f32Size(out_dims.out), .weight);
        try builder.add_node(.add, &.{ logits_name, out_bias }, logits_name, (out_dims.out + 63) / 64, 1, out_dims.out, out_dims.out, 0, 0);
    }
}

/// Full attention block (Qwen3-style with Qwen 3.5 joint Q+gate projection).
fn buildAttentionBlock(
    builder: *GraphBuilder,
    cfg: *const model.ModelConfig,
    layer: u32,
    in_name: []const u8,
) !void {
    const n_embd = cfg.n_embd;
    const n_heads = cfg.n_heads;
    const n_kv = cfg.n_kv_heads;
    const head_dim = cfg.head_dim;
    const eps_bits: u32 = @bitCast(cfg.rms_norm_eps);
    const rope_bits: u32 = @bitCast(cfg.rope_theta);

    var ln_buf: [32]u8 = undefined;
    const ln = try std.fmt.bufPrint(&ln_buf, "blk.{d}", .{layer});

    // 1. Attention Norm
    var nw_buf: [64]u8 = undefined;
    const nw = try std.fmt.bufPrint(&nw_buf, "{s}.attn_norm.weight", .{ln});
    var normed_buf: [64]u8 = undefined;
    const normed = try std.fmt.bufPrint(&normed_buf, "{s}.normed", .{ln});
    if (!builder.has_tensor(nw)) {
        std.log.err("qwen35: missing required tensor '{s}' for layer {d} attention block", .{ nw, layer });
        return error.MissingWeight;
    }
    _ = try builder.add_tensor(nw, f32Size(n_embd), .weight);
    _ = try builder.add_tensor(normed, f32Size(n_embd), .activation);
    try builder.add_node(.rms_norm, &.{ in_name, nw }, normed, (n_embd + 63) / 64, 1, n_embd, n_embd, eps_bits, 0);

    // 2. Joint Q+gate projection.
    var qw_buf: [64]u8 = undefined;
    const qw = try std.fmt.bufPrint(&qw_buf, "{s}.attn_q.weight", .{ln});
    if (!builder.has_tensor(qw)) {
        std.log.err("qwen35: missing required tensor '{s}' for layer {d} attention block", .{ qw, layer });
        return error.MissingWeight;
    }
    const qg_dims = builder.matmulDims(qw, 2 * head_dim * n_heads, n_embd);
    _ = try builder.add_tensor(qw, f32Size(qg_dims.out * qg_dims.in), .weight);
    var qg_buf: [64]u8 = undefined;
    const qg = try std.fmt.bufPrint(&qg_buf, "{s}.qg", .{ln});
    _ = try builder.add_tensor(qg, f32Size(qg_dims.out), .activation);
    try builder.add_nodeP(.attn_qg_matmul, &.{ normed, qw }, qg, (qg_dims.out + 15) / 16, 1, 1, qg_dims.out, qg_dims.in, 0, 0);

    const q_bytes = head_dim * n_heads;
    var q_name_buf: [64]u8 = undefined;
    const q_name = try builder.graph.allocator.dupe(u8, try std.fmt.bufPrint(&q_name_buf, "{s}.q", .{ln}));
    try builder.add_tensor(q_name, f32Size(q_bytes), .activation);
    var gate_name_buf: [64]u8 = undefined;
    const gate_name = try builder.graph.allocator.dupe(u8, try std.fmt.bufPrint(&gate_name_buf, "{s}.gate", .{ln}));
    try builder.add_tensor(gate_name, f32Size(q_bytes), .activation);

    try builder.add_nodeP(.qwen_deinterleave, &.{ qg, gate_name }, q_name, (q_bytes + 63) / 64, 1, q_bytes, n_heads, head_dim, 0, 0);

    const q_owned = q_name;
    const gate_owned = gate_name;

    // 3. Q RMS norm
    const q_norm_w = try std.fmt.allocPrint(builder.graph.allocator, "{s}.attn_q_norm.weight", .{ln});
    defer builder.graph.allocator.free(q_norm_w);
    if (builder.has_tensor(q_norm_w)) {
        _ = try builder.add_tensor(q_norm_w, f32Size(head_dim), .weight);
        try builder.add_node(.rms_norm, &.{ q_owned, q_norm_w }, q_owned, (n_heads + 63) / 64, 1, q_bytes, head_dim, eps_bits, 0);
    }

    // 4. K, V matmuls
    var kw_buf: [64]u8 = undefined;
    const kw = try std.fmt.bufPrint(&kw_buf, "{s}.attn_k.weight", .{ln});
    var vw_buf: [64]u8 = undefined;
    const vw = try std.fmt.bufPrint(&vw_buf, "{s}.attn_v.weight", .{ln});
    const kv_out = n_kv * head_dim;

    var kn_buf: [64]u8 = undefined;
    const kn = try std.fmt.bufPrint(&kn_buf, "{s}.k", .{ln});
    var vn_buf: [64]u8 = undefined;
    const vn = try std.fmt.bufPrint(&vn_buf, "{s}.v", .{ln});
    const k_dims = builder.matmulDims(kw, kv_out, n_embd);
    const v_dims = builder.matmulDims(vw, kv_out, n_embd);
    _ = try builder.add_tensor(kw, f32Size(k_dims.out * k_dims.in), .weight);
    _ = try builder.add_tensor(vw, f32Size(v_dims.out * v_dims.in), .weight);
    _ = try builder.add_tensor(kn, f32Size(k_dims.out), .activation);
    _ = try builder.add_tensor(vn, f32Size(v_dims.out), .activation);
    try builder.add_nodeP(.matmul, &.{ normed, kw }, kn, (kv_out + 15) / 16, 1, 1, kv_out, n_embd, 0, 0);
    try builder.add_nodeP(.matmul, &.{ normed, vw }, vn, (kv_out + 15) / 16, 1, 1, kv_out, n_embd, 0, 0);

    // 5. K RMS norm
    const k_norm_w = try std.fmt.allocPrint(builder.graph.allocator, "{s}.attn_k_norm.weight", .{ln});
    defer builder.graph.allocator.free(k_norm_w);
    if (builder.has_tensor(k_norm_w)) {
        _ = try builder.add_tensor(k_norm_w, f32Size(head_dim), .weight);
        try builder.add_node(.rms_norm, &.{ kn, k_norm_w }, kn, (n_kv + 63) / 64, 1, kv_out, head_dim, eps_bits, 0);
    }

    // 6. MRoPE / RoPE
    const q_head_dim = if (n_heads > 0) q_bytes / n_heads else head_dim;
    const k_head_dim = if (n_kv > 0) kv_out / n_kv else head_dim;
    try builder.emitRoPEMulti(q_owned, kn, q_bytes, kv_out, q_head_dim, k_head_dim, 0, rope_bits, 0, 0);

    // 7. KV write + attention
    var attn_buf: [64]u8 = undefined;
    const attn = try std.fmt.bufPrint(&attn_buf, "{s}.attn", .{ln});
    _ = try builder.add_tensor(attn, f32Size(q_bytes), .activation);

    var kv_name_buf: [16]u8 = undefined;
    const kv_name = try std.fmt.bufPrint(&kv_name_buf, "kv.{d}", .{layer});
    try builder.add_nodeP8(.kv_write, &.{ kn, vn, kv_name }, kn, ((kv_out / 2) + 63) / 64, 1, n_kv, k_head_dim, cfg.max_ctx, 0, 0, 0, 0, 0);
    const attn_p2 = k_head_dim | (n_kv << 16);
    const attn_scale_bits: u32 = @bitCast(cfg.attention_scale);
    try builder.add_nodeP8(.attention, &.{ q_owned, kv_name }, attn, n_heads, 1, n_heads, attn_p2, cfg.max_ctx, 0, attn_scale_bits, 0, 0, 0);

    // 8. sigmoid(gate) * attn
    try builder.add_nodeP8(.attn_gate_mul, &.{ attn, gate_owned }, attn, (q_bytes + 63) / 64, 1, q_bytes, 0, 0, 0, 0, 0, 0, 0);

    // 9. Output projection
    var ow_buf: [64]u8 = undefined;
    const ow = try std.fmt.bufPrint(&ow_buf, "{s}.attn_output.weight", .{ln});
    if (!builder.has_tensor(ow)) {
        std.log.err("qwen35: missing required tensor '{s}' for layer {d} attention block", .{ ow, layer });
        return error.MissingWeight;
    }
    const o_dims = builder.matmulDims(ow, n_embd, q_bytes);
    _ = try builder.add_tensor(ow, f32Size(o_dims.out * o_dims.in), .weight);
    var attn_out_buf: [64]u8 = undefined;
    const attn_out = try std.fmt.bufPrint(&attn_out_buf, "{s}.attn_out", .{ln});
    _ = try builder.add_tensor(attn_out, f32Size(o_dims.out), .activation);
    try builder.add_nodeP(.matmul, &.{ attn, ow }, attn_out, (o_dims.out + 15) / 16, 1, 1, o_dims.out, o_dims.in, 0, 0);

    // 10. Residual add
    var res1_buf: [64]u8 = undefined;
    const res1 = try std.fmt.bufPrint(&res1_buf, "{s}.res1", .{ln});
    _ = try builder.add_tensor(res1, f32Size(o_dims.out), .activation);
    const res_scale_bits: u32 = @bitCast(cfg.residual_scale);
    if (cfg.residual_scale != 1.0) {
        try builder.add_node(.scaled_add, &.{ in_name, attn_out }, res1, (o_dims.out + 63) / 64, 1, o_dims.out, res_scale_bits, 0, 0);
    } else {
        try builder.add_node(.add, &.{ in_name, attn_out }, res1, (o_dims.out + 63) / 64, 1, o_dims.out, 0, 0, 0);
    }
}

/// Linear-attention block: Gated Delta Net (Qwen 3.5).
fn buildSsmBlock(
    builder: *GraphBuilder,
    cfg: *const model.ModelConfig,
    layer: u32,
    in_name: []const u8,
) !void {
    const n_embd = cfg.n_embd;
    const d_inner = cfg.ssm_d_inner;
    const n_group = cfg.ssm_n_group;
    const d_state = cfg.ssm_d_state;
    const d_conv: u32 = if (cfg.ssm_d_conv > 0) cfg.ssm_d_conv else 4;
    const dt_rank = cfg.ssm_dt_rank;
    const head_k_dim = d_state;
    const num_k_heads = n_group;
    const num_v_heads = dt_rank;
    const head_v_dim: u32 = if (num_v_heads > 0) d_inner / num_v_heads else d_inner;
    const conv_channels = d_inner + 2 * n_group * d_state;
    const eps_bits: u32 = @bitCast(cfg.rms_norm_eps);

    var ln_buf: [32]u8 = undefined;
    const ln = try std.fmt.bufPrint(&ln_buf, "blk.{d}", .{layer});

    // 1. attn_norm
    var nw_buf: [64]u8 = undefined;
    const nw = try std.fmt.bufPrint(&nw_buf, "{s}.attn_norm.weight", .{ln});
    if (!builder.has_tensor(nw)) {
        std.log.err("qwen35: missing required tensor '{s}' for layer {d} SSM block", .{ nw, layer });
        return error.MissingWeight;
    }
    _ = try builder.add_tensor(nw, f32Size(n_embd), .weight);
    var normed_buf: [64]u8 = undefined;
    const normed = try std.fmt.bufPrint(&normed_buf, "{s}.normed", .{ln});
    _ = try builder.add_tensor(normed, f32Size(n_embd), .activation);
    try builder.add_node(.rms_norm, &.{ in_name, nw }, normed, (n_embd + 63) / 64, 1, n_embd, n_embd, eps_bits, 0);

    // 2. QKVZ projections.
    const layout = builder.resolveSsmQkvLayout(ln);
    var conv_input: []const u8 = undefined;
    var z_owned: []const u8 = undefined;
    switch (layout) {
        .fused_gate => {
            var qkv_buf: [64]u8 = undefined;
            var qkv_name = try std.fmt.bufPrint(&qkv_buf, "{s}.attn_qkv.weight", .{ln});
            if (!builder.has_tensor(qkv_name)) {
                qkv_name = try std.fmt.bufPrint(&qkv_buf, "{s}.attn_q.weight", .{ln});
            }
            var gate_buf: [64]u8 = undefined;
            const gate_name = try std.fmt.bufPrint(&gate_buf, "{s}.attn_gate.weight", .{ln});
            const qkv_dims = builder.matmulDims(qkv_name, conv_channels, n_embd);
            const gate_dims = builder.matmulDims(gate_name, d_inner, n_embd);
            _ = try builder.add_tensor(qkv_name, f32Size(qkv_dims.out * qkv_dims.in), .weight);
            _ = try builder.add_tensor(gate_name, f32Size(gate_dims.out * gate_dims.in), .weight);

            var qkv_act_buf: [64]u8 = undefined;
            const qkv_act = try std.fmt.bufPrint(&qkv_act_buf, "{s}.qkv_mixed", .{ln});
            _ = try builder.add_tensor(qkv_act, f32Size(qkv_dims.out), .activation);
            try builder.add_nodeP(.matmul, &.{ normed, qkv_name }, qkv_act, (qkv_dims.out + 15) / 16, 1, 1, qkv_dims.out, qkv_dims.in, 0, 0);

            var gate_act_buf: [64]u8 = undefined;
            const gate_act = try std.fmt.bufPrint(&gate_act_buf, "{s}.z", .{ln});
            _ = try builder.add_tensor(gate_act, f32Size(gate_dims.out), .activation);
            try builder.add_nodeP(.matmul, &.{ normed, gate_name }, gate_act, (gate_dims.out + 15) / 16, 1, 1, gate_dims.out, gate_dims.in, 0, 0);

            var ci_buf: [64]u8 = undefined;
            const ci_name = try std.fmt.bufPrint(&ci_buf, "{s}.conv_in", .{ln});
            conv_input = try builder.addSliceF32(ci_name, qkv_act, 0, @intCast(conv_channels));
            z_owned = gate_act;
        },
        .legacy_ssm_in, .synthetic => {
            const qkvz_dim = conv_channels + d_inner;
            var ssm_in_buf: [64]u8 = undefined;
            const ssm_in_name = try std.fmt.bufPrint(&ssm_in_buf, "{s}.ssm_in.weight", .{ln});
            if (builder.has_tensor(ssm_in_name)) {
                const qkvz_dims = builder.matmulDims(ssm_in_name, qkvz_dim, n_embd);
                _ = try builder.add_tensor(ssm_in_name, f32Size(qkvz_dims.out * qkvz_dims.in), .weight);
                var qkvz_act_buf: [64]u8 = undefined;
                const qkvz_act = try std.fmt.bufPrint(&qkvz_act_buf, "{s}.qkvz_mixed", .{ln});
                _ = try builder.add_tensor(qkvz_act, f32Size(qkvz_dims.out), .activation);
                try builder.add_nodeP(.matmul, &.{ normed, ssm_in_name }, qkvz_act, (qkvz_dims.out + 15) / 16, 1, 1, qkvz_dims.out, qkvz_dims.in, 0, 0);

                var ci_buf: [64]u8 = undefined;
                const ci_name = try std.fmt.bufPrint(&ci_buf, "{s}.conv_in", .{ln});
                conv_input = try builder.addSliceF32(ci_name, qkvz_act, 0, @intCast(conv_channels));

                var z_buf: [64]u8 = undefined;
                const z_name = try std.fmt.bufPrint(&z_buf, "{s}.z", .{ln});
                z_owned = try builder.addSliceF32(z_name, qkvz_act, @intCast(conv_channels), @intCast(d_inner));
            } else {
                try builder.addSyntheticWeight(ssm_in_name, 1);
                _ = try builder.add_tensor(ssm_in_name, f32Size(1), .weight);
                var qkvz_act_buf: [64]u8 = undefined;
                const qkvz_act = try std.fmt.bufPrint(&qkvz_act_buf, "{s}.qkvz_mixed", .{ln});
                _ = try builder.add_tensor(qkvz_act, f32Size(qkvz_dim), .activation);
                // Zero input path
                try builder.add_node(.copy, &.{in_name}, qkvz_act, (qkvz_dim + 63) / 64, 1, @min(n_embd, qkvz_dim), 0, 0, 0);

                var ci_buf: [64]u8 = undefined;
                const ci_name = try std.fmt.bufPrint(&ci_buf, "{s}.conv_in", .{ln});
                conv_input = try builder.addSliceF32(ci_name, qkvz_act, 0, @intCast(conv_channels));

                var z_buf: [64]u8 = undefined;
                const z_name = try std.fmt.bufPrint(&z_buf, "{s}.z", .{ln});
                z_owned = try builder.addSliceF32(z_name, qkvz_act, @intCast(conv_channels), @intCast(d_inner));
            }
        },
    }

    // 3. ssm_alpha
    var sa_buf: [64]u8 = undefined;
    const sa_name = try std.fmt.bufPrint(&sa_buf, "{s}.ssm_alpha.weight", .{ln});
    const alpha_dims = builder.matmulDims(sa_name, dt_rank, n_embd);
    if (builder.has_tensor(sa_name)) {
        _ = try builder.add_tensor(sa_name, f32Size(alpha_dims.out * alpha_dims.in), .weight);
    } else {
        try builder.addSyntheticWeight(sa_name, dt_rank * n_embd);
        _ = try builder.add_tensor(sa_name, f32Size(dt_rank * n_embd), .weight);
    }
    var alpha_buf: [64]u8 = undefined;
    const alpha_name = try std.fmt.bufPrint(&alpha_buf, "{s}.alpha", .{ln});
    _ = try builder.add_tensor(alpha_name, f32Size(dt_rank), .activation);
    try builder.add_nodeP(.matmul, &.{ normed, sa_name }, alpha_name, (dt_rank + 15) / 16, 1, 1, dt_rank, n_embd, 0, 0);

    // 4. ssm_beta
    var sb_buf: [64]u8 = undefined;
    const sb_name = try std.fmt.bufPrint(&sb_buf, "{s}.ssm_beta.weight", .{ln});
    const beta_dims = builder.matmulDims(sb_name, dt_rank, n_embd);
    if (builder.has_tensor(sb_name)) {
        _ = try builder.add_tensor(sb_name, f32Size(beta_dims.out * beta_dims.in), .weight);
    } else {
        try builder.addSyntheticWeight(sb_name, dt_rank * n_embd);
        _ = try builder.add_tensor(sb_name, f32Size(dt_rank * n_embd), .weight);
    }
    var beta_pre_buf: [64]u8 = undefined;
    const beta_pre_name = try std.fmt.bufPrint(&beta_pre_buf, "{s}.beta_pre", .{ln});
    _ = try builder.add_tensor(beta_pre_name, f32Size(dt_rank), .activation);
    try builder.add_nodeP(.matmul, &.{ normed, sb_name }, beta_pre_name, (dt_rank + 15) / 16, 1, 1, dt_rank, n_embd, 0, 0);

    // 5. alpha + ssm_dt.bias
    var dt_buf: [64]u8 = undefined;
    const dt_name = try std.fmt.bufPrint(&dt_buf, "{s}.ssm_dt.bias", .{ln});
    _ = try builder.add_tensor(dt_name, f32Size(dt_rank), .weight);
    var alpha_b_buf: [64]u8 = undefined;
    const alpha_b_name = try std.fmt.bufPrint(&alpha_b_buf, "{s}.alpha_biased", .{ln});
    _ = try builder.add_tensor(alpha_b_name, f32Size(dt_rank), .activation);
    try builder.add_node(.add, &.{ alpha_name, dt_name }, alpha_b_name, (dt_rank + 63) / 64, 1, dt_rank, 0, 0, 0);

    // 6. softplus
    var alpha_s_buf: [64]u8 = undefined;
    const alpha_s_name = try std.fmt.bufPrint(&alpha_s_buf, "{s}.alpha_soft", .{ln});
    _ = try builder.add_tensor(alpha_s_name, f32Size(dt_rank), .activation);
    try builder.add_node(.softplus, &.{alpha_b_name}, alpha_s_name, (dt_rank + 63) / 64, 1, dt_rank, 0, 0, 0);

    // 7. gate
    var ssm_a_buf: [64]u8 = undefined;
    const ssm_a_name = try std.fmt.bufPrint(&ssm_a_buf, "{s}.ssm_a", .{ln});
    _ = try builder.add_tensor(ssm_a_name, f32Size(dt_rank), .weight);
    var gate_buf: [64]u8 = undefined;
    const gate_name = try std.fmt.bufPrint(&gate_buf, "{s}.gate", .{ln});
    _ = try builder.add_tensor(gate_name, f32Size(dt_rank), .output);
    try builder.add_node(.mul, &.{ alpha_s_name, ssm_a_name }, gate_name, (dt_rank + 63) / 64, 1, dt_rank, 0, 0, 0);

    // 8. beta
    var beta_buf: [64]u8 = undefined;
    const beta_name = try std.fmt.bufPrint(&beta_buf, "{s}.beta", .{ln});
    _ = try builder.add_tensor(beta_name, f32Size(dt_rank), .output);
    try builder.add_node(.sigmoid, &.{beta_pre_name}, beta_name, (dt_rank + 63) / 64, 1, dt_rank, 0, 0, 0);

    // 9. SSM conv1d
    var ssm_conv_state_buf: [24]u8 = undefined;
    const ssm_conv_state_name = try std.fmt.bufPrint(&ssm_conv_state_buf, "ssm_conv.{d}", .{layer});

    var conv1d_w_buf: [64]u8 = undefined;
    const conv1d_w_name = try std.fmt.bufPrint(&conv1d_w_buf, "{s}.ssm_conv1d.weight", .{ln});
    _ = try builder.add_tensor(conv1d_w_name, f32Size(d_conv * conv_channels), .weight);
    var conv_out_buf: [64]u8 = undefined;
    const conv_out_name = try std.fmt.bufPrint(&conv_out_buf, "{s}.conv_out", .{ln});
    _ = try builder.add_tensor(conv_out_name, f32Size(conv_channels), .activation);

    try builder.add_nodeP8(.ssm_conv1d, &.{ ssm_conv_state_name, conv_input, conv1d_w_name }, conv_out_name, (conv_channels + 63) / 64, 1, conv_channels, conv_channels, d_conv, 0, 0, 0, 0, 0);

    // 10. silu of conv_out
    var conv_s_buf: [64]u8 = undefined;
    const conv_s_name = try std.fmt.bufPrint(&conv_s_buf, "{s}.conv_silu", .{ln});
    _ = try builder.add_tensor(conv_s_name, f32Size(conv_channels), .output);
    try builder.add_node(.silu, &.{conv_out_name}, conv_s_name, (conv_channels + 63) / 64, 1, conv_channels, 0, 0, 0);

    // 11. Slices
    var q_conv_buf: [64]u8 = undefined;
    const q_conv_name = try std.fmt.bufPrint(&q_conv_buf, "{s}.q_conv", .{ln});
    var k_conv_buf: [64]u8 = undefined;
    const k_conv_name = try std.fmt.bufPrint(&k_conv_buf, "{s}.k_conv", .{ln});
    var v_conv_buf: [64]u8 = undefined;
    const v_conv_name = try std.fmt.bufPrint(&v_conv_buf, "{s}.v_conv", .{ln});

    const q_conv_bytes = head_k_dim * num_k_heads;
    const k_conv_bytes = head_k_dim * num_k_heads;
    const v_conv_bytes = head_v_dim * num_v_heads;
    const q_owned = try builder.addSliceF32(q_conv_name, conv_s_name, 0, @intCast(q_conv_bytes));
    const k_owned = try builder.addSliceF32(k_conv_name, conv_s_name, @intCast(q_conv_bytes), @intCast(k_conv_bytes));
    const v_owned = try builder.addSliceF32(v_conv_name, conv_s_name, @intCast(q_conv_bytes + k_conv_bytes), @intCast(v_conv_bytes));

    // 12. l2_norm Q and K
    var q_norm_buf: [64]u8 = undefined;
    const q_norm_name = try std.fmt.bufPrint(&q_norm_buf, "{s}.q_norm", .{ln});
    _ = try builder.add_tensor(q_norm_name, f32Size(q_conv_bytes), .output);
    try builder.add_nodeP8(.l2_norm, &.{q_owned}, q_norm_name, num_k_heads, 1, head_k_dim, head_k_dim, eps_bits, 0, 0, 0, 0, 0);
    var k_norm_buf: [64]u8 = undefined;
    const k_norm_name = try std.fmt.bufPrint(&k_norm_buf, "{s}.k_norm", .{ln});
    _ = try builder.add_tensor(k_norm_name, f32Size(k_conv_bytes), .output);
    try builder.add_nodeP8(.l2_norm, &.{k_owned}, k_norm_name, num_k_heads, 1, head_k_dim, head_k_dim, eps_bits, 0, 0, 0, 0, 0);

    // 13. recurrence
    var ssm_state_buf: [24]u8 = undefined;
    const ssm_state_name = try std.fmt.bufPrint(&ssm_state_buf, "ssm_state.{d}", .{layer});
    var core_buf: [64]u8 = undefined;
    const core_name = try std.fmt.bufPrint(&core_buf, "{s}.core", .{ln});
    _ = try builder.add_tensor(core_name, f32Size(v_conv_bytes), .output);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_v_dim)));
    const scale_bits: u32 = @bitCast(scale);
    try builder.add_nodeP8(.ssm_delta_net_decode, &.{ ssm_state_name, q_norm_name, k_norm_name, v_owned, gate_name, beta_name }, core_name, num_v_heads, 1, num_v_heads, head_v_dim, head_k_dim, num_k_heads, scale_bits, 0, 0, 0);

    // 14. ssm_gated_norm
    var ssm_norm_w_buf: [64]u8 = undefined;
    const ssm_norm_w_name = try std.fmt.bufPrint(&ssm_norm_w_buf, "{s}.ssm_norm.weight", .{ln});
    _ = try builder.add_tensor(ssm_norm_w_name, f32Size(head_v_dim), .weight);
    var gated_buf: [64]u8 = undefined;
    const gated_name = try std.fmt.bufPrint(&gated_buf, "{s}.gated", .{ln});
    _ = try builder.add_tensor(gated_name, f32Size(v_conv_bytes), .activation);
    try builder.add_nodeP8(.ssm_gated_norm, &.{ core_name, z_owned, ssm_norm_w_name }, gated_name, num_v_heads, 1, num_v_heads, head_v_dim, 0, eps_bits, 0, 0, 0, 0);

    // 15. ssm_out projection
    var ssm_out_buf: [64]u8 = undefined;
    const ssm_out_name = try std.fmt.bufPrint(&ssm_out_buf, "{s}.ssm_out.weight", .{ln});
    var alt_buf: [64]u8 = undefined;
    const alt_name = try std.fmt.bufPrint(&alt_buf, "{s}.attn_output.weight", .{ln});

    var linear_name: []const u8 = undefined;
    if (builder.has_tensor(ssm_out_name) or builder.has_tensor(alt_name)) {
        const ssm_out_w_name = if (builder.has_tensor(ssm_out_name)) ssm_out_name else alt_name;
        const out_dims = builder.matmulDims(ssm_out_w_name, n_embd, v_conv_bytes);
        _ = try builder.add_tensor(ssm_out_w_name, f32Size(out_dims.out * out_dims.in), .weight);

        var linear_buf: [64]u8 = undefined;
        linear_name = try std.fmt.bufPrint(&linear_buf, "{s}.linear_attn_out", .{ln});
        _ = try builder.add_tensor(linear_name, f32Size(n_embd), .activation);
        try builder.add_nodeP(.matmul, &.{ gated_name, ssm_out_w_name }, linear_name, (n_embd + 15) / 16, 1, 1, n_embd, v_conv_bytes, 0, 0);
    } else {
        linear_name = "blk_zero";
        if (!builder.graph.tensors.contains("blk_zero")) {
            _ = try builder.add_tensor("blk_zero", f32Size(n_embd), .activation);
            try builder.addSyntheticWeight("blk_zero_w", 1);
            _ = try builder.add_tensor("blk_zero_w", f32Size(1), .weight);
        }
    }

    // 16. residual add
    var res1_buf: [64]u8 = undefined;
    const res1 = try std.fmt.bufPrint(&res1_buf, "{s}.res1", .{ln});
    _ = try builder.add_tensor(res1, f32Size(n_embd), .activation);
    const res_scale_bits: u32 = @bitCast(cfg.residual_scale);

    if (std.mem.eql(u8, linear_name, "blk_zero")) {
        try builder.add_node(.copy, &.{in_name}, res1, (n_embd + 63) / 64, 1, n_embd, 0, 0, 0);
    } else {
        if (cfg.residual_scale != 1.0) {
            try builder.add_node(.scaled_add, &.{ in_name, linear_name }, res1, (n_embd + 63) / 64, 1, n_embd, res_scale_bits, 0, 0);
        } else {
            try builder.add_node(.add, &.{ in_name, linear_name }, res1, (n_embd + 63) / 64, 1, n_embd, 0, 0, 0);
        }
    }
}

/// Shared FFN block.
fn buildFfnBlock(
    builder: *GraphBuilder,
    cfg: *const model.ModelConfig,
    layer: u32,
    in_name: []const u8,
    out_name: []const u8,
) !void {
    const n_embd = cfg.n_embd;
    const n_ff = cfg.n_ff;
    const eps_bits: u32 = @bitCast(cfg.rms_norm_eps);

    var ln_buf: [32]u8 = undefined;
    const ln = try std.fmt.bufPrint(&ln_buf, "blk.{d}", .{layer});

    var post_buf: [64]u8 = undefined;
    const post_name = try std.fmt.bufPrint(&post_buf, "{s}.post_attention_norm.weight", .{ln});
    var alt_post_buf: [64]u8 = undefined;
    const alt_post_name = try std.fmt.bufPrint(&alt_post_buf, "{s}.attn_post_norm.weight", .{ln});
    var ffn_buf: [64]u8 = undefined;
    const ffn_name = try std.fmt.bufPrint(&ffn_buf, "{s}.ffn_norm.weight", .{ln});
    var normed_buf: [64]u8 = undefined;
    const normed_name = try std.fmt.bufPrint(&normed_buf, "{s}.ffn_normed", .{ln});

    const ffn_in_w: []const u8 = blk: {
        if (builder.has_tensor(post_name)) {
            _ = try builder.add_tensor(post_name, f32Size(n_embd), .weight);
            break :blk post_name;
        }
        if (builder.has_tensor(alt_post_name)) {
            _ = try builder.add_tensor(alt_post_name, f32Size(n_embd), .weight);
            break :blk alt_post_name;
        }
        if (builder.has_tensor(ffn_name)) {
            _ = try builder.add_tensor(ffn_name, f32Size(n_embd), .weight);
            break :blk ffn_name;
        }
        return error.MissingWeight;
    };
    _ = try builder.add_tensor(normed_name, f32Size(n_embd), .activation);
    try builder.add_node(.rms_norm, &.{ in_name, ffn_in_w }, normed_name, (n_embd + 63) / 64, 1, n_embd, n_embd, eps_bits, 0);

    const gate_up = try builder.emitGateUp(ln, normed_name, n_ff, n_embd);
    defer {
        builder.graph.allocator.free(gate_up.gate);
        if (gate_up.up.ptr != gate_up.gate.ptr) builder.graph.allocator.free(gate_up.up);
    }
    const activation_op: OpType = if (cfg.activation == .gelu) .gelu_mul else .silu_mul;
    try builder.add_nodeP8(activation_op, &.{ gate_up.gate, gate_up.up }, gate_up.gate, (gate_up.out + 63) / 64, 1, gate_up.out, 0, 0, 0, gate_up.gate_off, gate_up.up_off, gate_up.gate_off, 0);

    var dw_buf: [64]u8 = undefined;
    const dw = try std.fmt.bufPrint(&dw_buf, "{s}.ffn_down.weight", .{ln});
    const d_dims = builder.matmulDims(dw, n_embd, gate_up.out);
    _ = try builder.add_tensor(dw, f32Size(d_dims.out * d_dims.in), .weight);
    var ffn_out_buf: [64]u8 = undefined;
    const ffn_out = try std.fmt.bufPrint(&ffn_out_buf, "{s}.ffn_out", .{ln});
    _ = try builder.add_tensor(ffn_out, f32Size(d_dims.out), .activation);
    try builder.add_nodeP(.matmul, &.{ gate_up.gate, dw }, ffn_out, (d_dims.out + 15) / 16, 1, 1, d_dims.out, d_dims.in, 0, 0);

    _ = try builder.add_tensor(out_name, f32Size(d_dims.out), .activation);
    const res_scale_bits: u32 = @bitCast(cfg.residual_scale);
    if (cfg.residual_scale != 1.0) {
        try builder.add_node(.scaled_add, &.{ in_name, ffn_out }, out_name, (d_dims.out + 63) / 64, 1, d_dims.out, res_scale_bits, 0, 0);
    } else {
        try builder.add_node(.add, &.{ in_name, ffn_out }, out_name, (d_dims.out + 63) / 64, 1, d_dims.out, 0, 0, 0);
    }
}

fn buildBlock(
    builder: *GraphBuilder,
    cfg: *const model.ModelConfig,
    layer: u32,
    in_name: []const u8,
    out_name: []const u8,
) !void {
    var res1_buf: [64]u8 = undefined;
    const res1 = try std.fmt.bufPrint(&res1_buf, "blk.{d}.res1", .{layer});
    if (cfg.isRecurrent(layer)) {
        try buildSsmBlock(builder, cfg, layer, in_name);
    } else {
        try buildAttentionBlock(builder, cfg, layer, in_name);
    }
    try buildFfnBlock(builder, cfg, layer, res1, out_name);
}

pub fn build(
    allocator: std.mem.Allocator,
    graph: *compute_graph.Graph,
    cfg: *const model.ModelConfig,
    model_tensors: *const std.StringHashMap(*tensor.Tensor),
) !void {
    var builder = GraphBuilder.init(graph, cfg, model_tensors);

    _ = try builder.add_tensor("input", model.f32Bytes(cfg.n_embd), .input);
    try builder.init_kv_caches();
    try builder.init_ssm_caches();

    const n_main = cfg.n_layer - cfg.nextn_predict_layers;
    var prev_out: []const u8 = "input";
    var l: u32 = 0;
    while (l < n_main) : (l += 1) {
        const out_owned = if (l == n_main - 1)
            try allocator.dupe(u8, "hidden")
        else
            try std.fmt.allocPrint(allocator, "blk.{d}.out", .{l});
        defer allocator.free(out_owned);

        try buildBlock(&builder, cfg, l, prev_out, out_owned);
        prev_out = graph.tensors.getPtr(out_owned).?.name;
    }

    const has_output = model_tensors.get("output.weight") != null;
    if (has_output) {
        _ = try builder.add_tensor("output.weight", model.f32Bytes(@as(u64, cfg.vocab_size) * cfg.n_embd), .weight);
    }
    _ = try builder.add_tensor("output_norm.weight", model.f32Bytes(cfg.n_embd), .weight);
    try build_lm_head(&builder, cfg, prev_out, "logits", has_output);

    // if (cfg.nextn_predict_layers > 0) {
    //     var it = model_tensors.iterator();
    //     while (it.next()) |entry| {
    //         if (std.mem.startsWith(u8, entry.key_ptr.*, "nextn.")) {
    //             _ = try builder.add_tensor(entry.key_ptr.*, entry.value_ptr.*.size(), .weight);
    //         }
    //     }
    // }

    try builder.finalize();
}

pub fn buildBlockEntry(
    builder: *GraphBuilder,
    cfg: *const model.ModelConfig,
    layer: u32,
    in_name: []const u8,
    out_name: []const u8,
) !void {
    try buildBlock(builder, cfg, layer, in_name, out_name);
}

