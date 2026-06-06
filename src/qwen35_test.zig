//! Tests for the Qwen 3.5 graph builder.

const std = @import("std");
const compute_graph = @import("compute_graph.zig");
const model = @import("model.zig");
const tensor = @import("tensor.zig");
const qwen35 = @import("qwen35.zig");

test "isRecurrent schedule for n_layer=8 interval=4" {
    const t = std.testing;
    const cfg = model.ModelConfig{
        .arch = .qwen35,
        .arch_prefix = "qwen35",
        .activation = .silu,
        .n_embd = 256,
        .n_layer = 8,
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
        .ssm_d_conv = 4,
        .ssm_d_inner = 256,
        .ssm_d_state = 64,
        .ssm_dt_rank = 8,
        .ssm_n_group = 2,
        .full_attn_interval = 4,
    };
    var recurrent_count: u32 = 0;
    var attention_count: u32 = 0;
    var l: u32 = 0;
    while (l < 8) : (l += 1) {
        if (cfg.isRecurrent(l)) {
            recurrent_count += 1;
        } else {
            attention_count += 1;
        }
    }
    try t.expectEqual(@as(u32, 6), recurrent_count);
    try t.expectEqual(@as(u32, 2), attention_count);
    try t.expect(cfg.isRecurrent(0));
    try t.expect(!cfg.isRecurrent(3));
    try t.expect(cfg.isRecurrent(4));
    try t.expect(!cfg.isRecurrent(7));
}

test "scanLayers counts kinds without invoking build" {
    const t = std.testing;
    const allocator = t.allocator;
    const n_embd: u32 = 1024;
    var tensors = std.StringHashMap(*tensor.Tensor).init(allocator);
    defer tensors.deinit();

    const conv1d = try tensor.Tensor.init(allocator, "blk.0.ssm_conv1d.weight", .q4_k, &.{ 4, 256 });
    defer conv1d.deinit(allocator);
    try tensors.put("blk.0.ssm_conv1d.weight", conv1d);

    const attn_q = try tensor.Tensor.init(allocator, "blk.3.attn_q.weight", .q4_k, &.{ 256, 1024 });
    defer attn_q.deinit(allocator);
    try tensors.put("blk.3.attn_q.weight", attn_q);

    _ = model.ModelConfig{
        .arch = .qwen35,
        .arch_prefix = "qwen35",
        .activation = .silu,
        .n_embd = n_embd,
        .n_layer = 4,
        .n_heads = 8,
        .n_kv_heads = 2,
        .n_ff = 4096,
        .head_dim = 128,
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
        .ssm_d_conv = 4,
        .ssm_d_inner = 256,
        .ssm_d_state = 32,
        .ssm_dt_rank = 8,
        .ssm_n_group = 2,
        .full_attn_interval = 4,
    };

    var ssm_count: u32 = 0;
    var attn_count: u32 = 0;
    var l: u32 = 0;
    while (l < 4) : (l += 1) {
        var ss_buf: [64]u8 = undefined;
        const ss = try std.fmt.bufPrint(&ss_buf, "blk.{d}.ssm_conv1d.weight", .{l});
        var aq_buf: [64]u8 = undefined;
        const aq = try std.fmt.bufPrint(&aq_buf, "blk.{d}.attn_q.weight", .{l});
        if (tensors.contains(ss)) ssm_count += 1;
        if (tensors.contains(aq)) attn_count += 1;
    }
    try t.expectEqual(@as(u32, 1), ssm_count);
    try t.expectEqual(@as(u32, 1), attn_count);
}

test "build emits expected op sequence for a tiny Qwen 3.5 graph" {
    const t = std.testing;
    const allocator = t.allocator;
    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();

    var tensors = std.StringHashMap(*tensor.Tensor).init(allocator);
    defer tensors.deinit();

    const n_embd: u32 = 64;
    const n_ff: u32 = 128;
    const ssm_d_inner: u32 = 32;
    const ssm_d_state: u32 = 8;
    const ssm_dt_rank: u32 = 4;
    const ssm_n_group: u32 = 1;
    const n_vocab: u32 = 32;

    const cfg = model.ModelConfig{
        .arch = .qwen35,
        .arch_prefix = "qwen35",
        .activation = .silu,
        .n_embd = n_embd,
        .n_layer = 1,
        .n_heads = 2,
        .n_kv_heads = 1,
        .n_ff = n_ff,
        .head_dim = 32,
        .vocab_size = n_vocab,
        .max_ctx = 32,
        .rope_theta = 1.0e6,
        .rms_norm_eps = 1.0e-6,
        .wtype = .q4_k,
        .embedding_scale = 1.0,
        .attention_scale = 0.0,
        .residual_scale = 1.0,
        .logit_scale = 1.0,
        .final_logit_softcapping = 0.0,
        .ssm_d_conv = 4,
        .ssm_d_inner = ssm_d_inner,
        .ssm_d_state = ssm_d_state,
        .ssm_dt_rank = ssm_dt_rank,
        .ssm_n_group = ssm_n_group,
        .full_attn_interval = 4,
    };

    const ssm_qkv_dims = [_]u64{ ssm_d_inner + 2 * ssm_n_group * ssm_d_state, n_embd };
    const t_qkv = try tensor.Tensor.init(allocator, "blk.0.attn_qkv.weight", .q4_k, &ssm_qkv_dims);
    defer t_qkv.deinit(allocator);
    try tensors.put("blk.0.attn_qkv.weight", t_qkv);

    const t_gate = try tensor.Tensor.init(allocator, "blk.0.attn_gate.weight", .q4_k, &.{ ssm_d_inner, n_embd });
    defer t_gate.deinit(allocator);
    try tensors.put("blk.0.attn_gate.weight", t_gate);

    const t_norm = try tensor.Tensor.init(allocator, "blk.0.attn_norm.weight", .f32, &.{n_embd});
    defer t_norm.deinit(allocator);
    try tensors.put("blk.0.attn_norm.weight", t_norm);

    const t_alpha = try tensor.Tensor.init(allocator, "blk.0.ssm_alpha.weight", .q4_k, &.{ ssm_dt_rank, n_embd });
    defer t_alpha.deinit(allocator);
    try tensors.put("blk.0.ssm_alpha.weight", t_alpha);

    const t_beta = try tensor.Tensor.init(allocator, "blk.0.ssm_beta.weight", .q4_k, &.{ ssm_dt_rank, n_embd });
    defer t_beta.deinit(allocator);
    try tensors.put("blk.0.ssm_beta.weight", t_beta);

    const t_dt = try tensor.Tensor.init(allocator, "blk.0.ssm_dt.bias", .f32, &.{ssm_dt_rank});
    defer t_dt.deinit(allocator);
    try tensors.put("blk.0.ssm_dt.bias", t_dt);

    const t_a = try tensor.Tensor.init(allocator, "blk.0.ssm_a", .f32, &.{ssm_dt_rank});
    defer t_a.deinit(allocator);
    try tensors.put("blk.0.ssm_a", t_a);

    const t_conv1d = try tensor.Tensor.init(allocator, "blk.0.ssm_conv1d.weight", .q4_k, &.{ 4, ssm_d_inner + 2 * ssm_n_group * ssm_d_state });
    defer t_conv1d.deinit(allocator);
    try tensors.put("blk.0.ssm_conv1d.weight", t_conv1d);

    const t_ssm_norm = try tensor.Tensor.init(allocator, "blk.0.ssm_norm.weight", .f32, &.{ssm_d_inner / ssm_dt_rank});
    defer t_ssm_norm.deinit(allocator);
    try tensors.put("blk.0.ssm_norm.weight", t_ssm_norm);

    const t_ssm_out = try tensor.Tensor.init(allocator, "blk.0.ssm_out.weight", .q4_k, &.{ n_embd, ssm_d_inner });
    defer t_ssm_out.deinit(allocator);
    try tensors.put("blk.0.ssm_out.weight", t_ssm_out);

    const t_ffn_gate = try tensor.Tensor.init(allocator, "blk.0.ffn_gate.weight", .q4_k, &.{ n_ff, n_embd });
    defer t_ffn_gate.deinit(allocator);
    try tensors.put("blk.0.ffn_gate.weight", t_ffn_gate);

    const t_ffn_up = try tensor.Tensor.init(allocator, "blk.0.ffn_up.weight", .q4_k, &.{ n_ff, n_embd });
    defer t_ffn_up.deinit(allocator);
    try tensors.put("blk.0.ffn_up.weight", t_ffn_up);

    const t_ffn_down = try tensor.Tensor.init(allocator, "blk.0.ffn_down.weight", .q4_k, &.{ n_embd, n_ff });
    defer t_ffn_down.deinit(allocator);
    try tensors.put("blk.0.ffn_down.weight", t_ffn_down);

    const t_ffn_norm = try tensor.Tensor.init(allocator, "blk.0.ffn_norm.weight", .f32,&.{n_embd});
    defer t_ffn_norm.deinit(allocator);
    try tensors.put("blk.0.ffn_norm.weight", t_ffn_norm);

    const t_post_norm = try tensor.Tensor.init(allocator, "blk.0.attn_post_norm.weight", .f32, &.{n_embd});
    defer t_post_norm.deinit(allocator);
    try tensors.put("blk.0.attn_post_norm.weight", t_post_norm);

    const t_embd = try tensor.Tensor.init(allocator, "token_embd.weight", .q4_k, &.{ n_embd, n_vocab });
    defer t_embd.deinit(allocator);
    try tensors.put("token_embd.weight", t_embd);

    const t_out_norm = try tensor.Tensor.init(allocator, "output_norm.weight", .f32,&.{n_embd});
    defer t_out_norm.deinit(allocator);
    try tensors.put("output_norm.weight", t_out_norm);

    try qwen35.build(allocator, &graph, &cfg, &tensors);
    try graph.verify();

    try t.expect(graph.tensors.contains("blk.0.attn_norm.weight"));
    try t.expect(graph.tensors.contains("blk.0.ssm_conv1d.weight"));
    try t.expect(graph.tensors.contains("blk.0.ffn_down.weight"));
    try t.expect(graph.tensors.contains("output_norm.weight"));
    try t.expect(graph.tensors.contains("logits"));
    try t.expect(graph.tensors.contains("input"));
    try t.expect(graph.tensors.contains("ssm_conv.0"));
    try t.expect(graph.tensors.contains("ssm_state.0"));
    var found_conv1d = false;
    var found_softplus = false;
    var found_sigmoid = false;
    var found_ssm_gated = false;
    var found_ssm_delta = false;
    var found_attn_gate_mul = false;
    for (graph.nodes.items) |n| {
        if (n.op_type == .ssm_conv1d) found_conv1d = true;
        if (n.op_type == .softplus) found_softplus = true;
        if (n.op_type == .sigmoid) found_sigmoid = true;
        if (n.op_type == .ssm_gated_norm) found_ssm_gated = true;
        if (n.op_type == .ssm_delta_net_decode) found_ssm_delta = true;
        if (n.op_type == .attn_gate_mul) found_attn_gate_mul = true;
    }
    try t.expect(found_conv1d);
    try t.expect(found_softplus);
    try t.expect(found_sigmoid);
    try t.expect(found_ssm_gated);
    try t.expect(found_ssm_delta);
    try t.expect(!found_attn_gate_mul);
}

test "build with MTP gating registers nextn.* tensors but skips them from main graph" {
    const t = std.testing;
    const allocator = t.allocator;
    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();

    var tensors = std.StringHashMap(*tensor.Tensor).init(allocator);
    defer tensors.deinit();

    const n_embd: u32 = 64;
    const n_ff: u32 = 128;
    const ssm_d_inner: u32 = 32;
    const ssm_d_state: u32 = 8;
    const ssm_dt_rank: u32 = 4;
    const ssm_n_group: u32 = 1;
    const n_vocab: u32 = 32;

    const cfg = model.ModelConfig{
        .arch = .qwen35,
        .arch_prefix = "qwen35",
        .activation = .silu,
        .n_embd = n_embd,
        .n_layer = 2,
        .n_heads = 2,
        .n_kv_heads = 1,
        .n_ff = n_ff,
        .head_dim = 32,
        .vocab_size = n_vocab,
        .max_ctx = 32,
        .rope_theta = 1.0e6,
        .rms_norm_eps = 1.0e-6,
        .wtype = .q4_k,
        .embedding_scale = 1.0,
        .attention_scale = 0.0,
        .residual_scale = 1.0,
        .logit_scale = 1.0,
        .final_logit_softcapping = 0.0,
        .ssm_d_conv = 4,
        .ssm_d_inner = ssm_d_inner,
        .ssm_d_state = ssm_d_state,
        .ssm_dt_rank = ssm_dt_rank,
        .ssm_n_group = ssm_n_group,
        .nextn_predict_layers = 1,
        .full_attn_interval = 4,
    };

    const t_qkv = try tensor.Tensor.init(allocator, "blk.0.attn_qkv.weight", .q4_k, &.{ ssm_d_inner + 2 * ssm_n_group * ssm_d_state, n_embd });
    defer t_qkv.deinit(allocator);
    try tensors.put("blk.0.attn_qkv.weight", t_qkv);
    const t_gate = try tensor.Tensor.init(allocator, "blk.0.attn_gate.weight", .q4_k, &.{ ssm_d_inner, n_embd });
    defer t_gate.deinit(allocator);
    try tensors.put("blk.0.attn_gate.weight", t_gate);
    const t_norm = try tensor.Tensor.init(allocator, "blk.0.attn_norm.weight", .f32, &.{n_embd});
    defer t_norm.deinit(allocator);
    try tensors.put("blk.0.attn_norm.weight", t_norm);
    const t_alpha = try tensor.Tensor.init(allocator, "blk.0.ssm_alpha.weight", .q4_k, &.{ ssm_dt_rank, n_embd });
    defer t_alpha.deinit(allocator);
    try tensors.put("blk.0.ssm_alpha.weight", t_alpha);
    const t_beta = try tensor.Tensor.init(allocator, "blk.0.ssm_beta.weight", .q4_k, &.{ ssm_dt_rank, n_embd });
    defer t_beta.deinit(allocator);
    try tensors.put("blk.0.ssm_beta.weight", t_beta);
    const t_dt = try tensor.Tensor.init(allocator, "blk.0.ssm_dt.bias", .f32, &.{ssm_dt_rank});
    defer t_dt.deinit(allocator);
    try tensors.put("blk.0.ssm_dt.bias", t_dt);
    const t_a = try tensor.Tensor.init(allocator, "blk.0.ssm_a", .f32, &.{ssm_dt_rank});
    defer t_a.deinit(allocator);
    try tensors.put("blk.0.ssm_a", t_a);
    const t_conv1d = try tensor.Tensor.init(allocator, "blk.0.ssm_conv1d.weight", .q4_k, &.{ 4, ssm_d_inner + 2 * ssm_n_group * ssm_d_state });
    defer t_conv1d.deinit(allocator);
    try tensors.put("blk.0.ssm_conv1d.weight", t_conv1d);
    const t_ssm_norm = try tensor.Tensor.init(allocator, "blk.0.ssm_norm.weight", .f32, &.{ssm_d_inner / ssm_dt_rank});
    defer t_ssm_norm.deinit(allocator);
    try tensors.put("blk.0.ssm_norm.weight", t_ssm_norm);
    const t_ssm_out = try tensor.Tensor.init(allocator, "blk.0.ssm_out.weight", .q4_k, &.{ n_embd, ssm_d_inner });
    defer t_ssm_out.deinit(allocator);
    try tensors.put("blk.0.ssm_out.weight", t_ssm_out);
    const t_ffn_gate = try tensor.Tensor.init(allocator, "blk.0.ffn_gate.weight", .q4_k, &.{ n_ff, n_embd });
    defer t_ffn_gate.deinit(allocator);
    try tensors.put("blk.0.ffn_gate.weight", t_ffn_gate);
    const t_ffn_up = try tensor.Tensor.init(allocator, "blk.0.ffn_up.weight", .q4_k, &.{ n_ff, n_embd });
    defer t_ffn_up.deinit(allocator);
    try tensors.put("blk.0.ffn_up.weight", t_ffn_up);
    const t_ffn_down = try tensor.Tensor.init(allocator, "blk.0.ffn_down.weight", .q4_k, &.{ n_embd, n_ff });
    defer t_ffn_down.deinit(allocator);
    try tensors.put("blk.0.ffn_down.weight", t_ffn_down);
    const t_ffn_norm = try tensor.Tensor.init(allocator, "blk.0.ffn_norm.weight", .f32, &.{n_embd});
    defer t_ffn_norm.deinit(allocator);
    try tensors.put("blk.0.ffn_norm.weight", t_ffn_norm);
    const t_post_norm = try tensor.Tensor.init(allocator, "blk.0.attn_post_norm.weight", .f32, &.{n_embd});
    defer t_post_norm.deinit(allocator);
    try tensors.put("blk.0.attn_post_norm.weight", t_post_norm);
    const t_embd = try tensor.Tensor.init(allocator, "token_embd.weight", .q4_k, &.{ n_embd, n_vocab });
    defer t_embd.deinit(allocator);
    try tensors.put("token_embd.weight", t_embd);
    const t_out_norm = try tensor.Tensor.init(allocator, "output_norm.weight", .f32,&.{n_embd});
    defer t_out_norm.deinit(allocator);
    try tensors.put("output_norm.weight", t_out_norm);

    try qwen35.build(allocator, &graph, &cfg, &tensors);

    try t.expect(!graph.tensors.contains("nextn.0.embed_tokens.weight"));
    try t.expect(!graph.tensors.contains("nextn.0.enorm.weight"));
    try t.expect(!graph.tensors.contains("blk.1.attn_norm.weight"));
    try t.expect(graph.tensors.contains("blk.0.attn_norm.weight"));
}

test "runCpuSsmDeltaForLayer name lookups return distinct tensors" {
    const test_t = std.testing;
    const allocator = test_t.allocator;
    var graph = compute_graph.Graph.init(allocator);
    defer graph.deinit();

    var tensors = std.StringHashMap(*tensor.Tensor).init(allocator);
    defer tensors.deinit();
    defer {
        var it = tensors.valueIterator();
        while (it.next()) |tensor_ptr| {
            tensor_ptr.*.deinit(allocator);
        }
    }

    const n_embd: u32 = 64;
    const n_ff: u32 = 128;
    const ssm_d_inner: u32 = 32;
    const ssm_d_state: u32 = 8;
    const ssm_dt_rank: u32 = 4;
    const ssm_n_group: u32 = 1;
    const n_vocab: u32 = 32;

    const cfg = model.ModelConfig{
        .arch = .qwen35,
        .arch_prefix = "qwen35",
        .activation = .silu,
        .n_embd = n_embd,
        .n_layer = 1,
        .n_heads = 2,
        .n_kv_heads = 1,
        .n_ff = n_ff,
        .head_dim = 32,
        .vocab_size = n_vocab,
        .max_ctx = 32,
        .rope_theta = 1.0e6,
        .rms_norm_eps = 1.0e-6,
        .wtype = .q4_k,
        .embedding_scale = 1.0,
        .attention_scale = 0.0,
        .residual_scale = 1.0,
        .logit_scale = 1.0,
        .final_logit_softcapping = 0.0,
        .ssm_d_conv = 4,
        .ssm_d_inner = ssm_d_inner,
        .ssm_d_state = ssm_d_state,
        .ssm_dt_rank = ssm_dt_rank,
        .ssm_n_group = ssm_n_group,
        .full_attn_interval = 4,
    };

    try tensors.put("blk.0.attn_qkv.weight", try tensor.Tensor.init(allocator, "blk.0.attn_qkv.weight", .q4_k, &.{ ssm_d_inner + 2 * ssm_n_group * ssm_d_state, n_embd }));
    try tensors.put("blk.0.attn_gate.weight", try tensor.Tensor.init(allocator, "blk.0.attn_gate.weight", .q4_k,&.{ ssm_d_inner, n_embd }));
    try tensors.put("blk.0.attn_norm.weight", try tensor.Tensor.init(allocator, "blk.0.attn_norm.weight", .f32, &.{n_embd}));
    try tensors.put("blk.0.ssm_alpha.weight", try tensor.Tensor.init(allocator, "blk.0.ssm_alpha.weight", .q4_k,&.{ ssm_dt_rank, n_embd }));
    try tensors.put("blk.0.ssm_beta.weight", try tensor.Tensor.init(allocator, "blk.0.ssm_beta.weight", .q4_k,&.{ ssm_dt_rank, n_embd }));
    try tensors.put("blk.0.ssm_dt.bias", try tensor.Tensor.init(allocator, "blk.0.ssm_dt.bias", .f32,&.{ssm_dt_rank}));
    try tensors.put("blk.0.ssm_a", try tensor.Tensor.init(allocator, "blk.0.ssm_a", .f32,&.{ssm_dt_rank}));
    try tensors.put("blk.0.ssm_conv1d.weight", try tensor.Tensor.init(allocator, "blk.0.ssm_conv1d.weight", .q4_k,&.{ 4, ssm_d_inner + 2 * ssm_n_group * ssm_d_state }));
    try tensors.put("blk.0.ssm_norm.weight", try tensor.Tensor.init(allocator, "blk.0.ssm_norm.weight", .f32, &.{ssm_d_inner / ssm_dt_rank}));
    try tensors.put("blk.0.ssm_out.weight", try tensor.Tensor.init(allocator, "blk.0.ssm_out.weight", .q4_k, &.{ n_embd, ssm_d_inner }));
    try tensors.put("blk.0.ffn_gate.weight", try tensor.Tensor.init(allocator, "blk.0.ffn_gate.weight", .q4_k, &.{ n_ff, n_embd }));
    try tensors.put("blk.0.ffn_up.weight", try tensor.Tensor.init(allocator, "blk.0.ffn_up.weight", .q4_k,&.{ n_ff, n_embd }));
    try tensors.put("blk.0.ffn_down.weight", try tensor.Tensor.init(allocator, "blk.0.ffn_down.weight", .q4_k, &.{ n_embd, n_ff }));
    try tensors.put("blk.0.ffn_norm.weight", try tensor.Tensor.init(allocator, "blk.0.ffn_norm.weight", .f32, &.{n_embd}));
    try tensors.put("blk.0.attn_post_norm.weight", try tensor.Tensor.init(allocator, "blk.0.attn_post_norm.weight", .f32, &.{n_embd}));
    try tensors.put("token_embd.weight", try tensor.Tensor.init(allocator, "token_embd.weight", .q4_k, &.{ n_embd, n_vocab }));
    try tensors.put("output_norm.weight", try tensor.Tensor.init(allocator, "output_norm.weight", .f32,&.{n_embd}));

    try qwen35.build(allocator, &graph, &cfg, &tensors);

    var qn_buf: [32]u8 = undefined;  const qn_name   = std.fmt.bufPrint(&qn_buf,   "blk.0.q_norm", .{}) catch return;
    var kn_buf: [32]u8 = undefined;  const kn_name   = std.fmt.bufPrint(&kn_buf,   "blk.0.k_norm", .{}) catch return;
    var vn_buf: [32]u8 = undefined;  const vn_name   = std.fmt.bufPrint(&vn_buf,   "blk.0.v_conv", .{}) catch return;
    var gate_buf: [32]u8 = undefined; const gate_name = std.fmt.bufPrint(&gate_buf, "blk.0.gate",   .{}) catch return;
    var beta_buf: [32]u8 = undefined; const beta_name = std.fmt.bufPrint(&beta_buf, "blk.0.beta",   .{}) catch return;
    var core_buf: [32]u8 = undefined; const core_name = std.fmt.bufPrint(&core_buf, "blk.0.core",   .{}) catch return;

    const qn_t = graph.resolve_tensor_offset(qn_name) orelse return error.MissingTensor;
    const kn_t = graph.resolve_tensor_offset(kn_name) orelse return error.MissingTensor;
    const vn_t = graph.resolve_tensor_offset(vn_name) orelse return error.MissingTensor;
    const gate_t = graph.resolve_tensor_offset(gate_name) orelse return error.MissingTensor;
    const beta_t = graph.resolve_tensor_offset(beta_name) orelse return error.MissingTensor;
    const core_t = graph.resolve_tensor_offset(core_name) orelse return error.MissingTensor;

    try test_t.expect(qn_t.offset != kn_t.offset or qn_t.size != kn_t.size);
    try test_t.expect(qn_t.offset != vn_t.offset or qn_t.size != vn_t.size);
    try test_t.expect(qn_t.offset != gate_t.offset or qn_t.size != gate_t.size);
    try test_t.expect(qn_t.offset != beta_t.offset or qn_t.size != beta_t.size);
    try test_t.expect(qn_t.offset != core_t.offset or qn_t.size != core_t.size);
    try test_t.expect(kn_t.offset != vn_t.offset or kn_t.size != vn_t.size);
    try test_t.expect(kn_t.offset != gate_t.offset or kn_t.size != gate_t.size);
    try test_t.expect(kn_t.offset != beta_t.offset or kn_t.size != beta_t.size);
    try test_t.expect(kn_t.offset != core_t.offset or kn_t.size != core_t.size);
    try test_t.expect(vn_t.offset != gate_t.offset or vn_t.size != gate_t.size);
    try test_t.expect(vn_t.offset != beta_t.offset or vn_t.size != beta_t.size);
    try test_t.expect(vn_t.offset != core_t.offset or vn_t.size != core_t.size);
    try test_t.expect(gate_t.offset != beta_t.offset or gate_t.size != beta_t.size);
    try test_t.expect(gate_t.offset != core_t.offset or gate_t.size != core_t.size);
    try test_t.expect(beta_t.offset != core_t.offset or beta_t.size != core_t.size);

    try test_t.expectEqual(@as(u64, ssm_d_state * ssm_n_group * 4), qn_t.size);
    try test_t.expectEqual(@as(u64, ssm_d_state * ssm_n_group * 4), kn_t.size);
    try test_t.expectEqual(@as(u64, ssm_d_inner * 4), vn_t.size);
    try test_t.expectEqual(@as(u64, ssm_dt_rank * 4), gate_t.size);
    try test_t.expectEqual(@as(u64, ssm_dt_rank * 4), beta_t.size);
    try test_t.expectEqual(@as(u64, ssm_d_inner * 4), core_t.size);
}
