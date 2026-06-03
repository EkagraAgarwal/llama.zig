//! CPU reference tests for the Gated Delta Net (GDN) step.
//!
//! The GDN update is:
//!   a. S = S * exp(g)
//!   b. s_k[v] = sum_k S[k,v] * k[k]
//!   c. d[v]   = (v[v] - s_k[v]) * beta
//!   d. S[k,v] = S[k,v] + k[k] * d[v]
//!   e. o[v]   = sum_k S[k,v] * q[k] * scale

const std = @import("std");
const model = @import("model.zig");
const ssm = @import("ssm_state.zig");

fn makeCfg(num_v_heads: u32, head_v_dim: u32) model.ModelConfig {
    return .{
        .arch = .qwen35,
        .arch_prefix = "qwen35",
        .activation = .silu,
        .n_embd = 256,
        .n_layer = 1,
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
        .ssm_d_inner = num_v_heads * head_v_dim,
        .ssm_d_state = head_v_dim,
        .ssm_dt_rank = num_v_heads,
        .ssm_n_group = 1,
    };
}

test "stepDeltaNet: hand-computed 1-head 2x2 reference" {
    const allocator = std.testing.allocator;
    // 1 head, head_v_dim=2 — easiest to hand-verify.
    var cfg = makeCfg(1, 2);
    var ctx = try ssm.SsmCpuContext.init(allocator, &cfg);
    defer ctx.deinit();

    var q = [_]f32{ 1.0, 0.0 };
    var k = [_]f32{ 0.5, 0.5 };
    var v = [_]f32{ 1.0, 1.0 };
    var g = [_]f32{0.0}; // no decay
    var beta = [_]f32{1.0};
    var out = [_]f32{ 0.0, 0.0 };

    // S starts zero. After first step:
    // a. S unchanged (exp(0)=1).
    // b. s_k[v] = 0 for both v (S is zero).
    // c. d[v] = (1 - 0) * 1 = 1.
    // d. S += outer(k, d) = [[0.5*1, 0.5*1], [0.5*1, 0.5*1]] = all 0.5.
    // e. o[v] = sum_k S[k,v] * q[k] = S[0,v]*1 + S[1,v]*0 = S[0,v] = 0.5.
    try ctx.stepDeltaNet(0, &q, &k, &v, &g, &beta, &out, 1.0);
    try std.testing.expectApproxEqRel(@as(f32, 0.5), out[0], 1e-5);
    try std.testing.expectApproxEqRel(@as(f32, 0.5), out[1], 1e-5);

    // Second step with same inputs (now S is all 0.5):
    // a. S unchanged.
    // b. s_k[v] = S[0,v]*k[0] + S[1,v]*k[1] = 0.5*0.5 + 0.5*0.5 = 0.5.
    // c. d[v] = (1 - 0.5) * 1 = 0.5.
    // d. S[k,v] += 0.5 * 0.5 = 0.25, so S = [[0.75, 0.75], [0.75, 0.75]].
    // e. o[v] = S[0,v]*q[0] = 0.75.
    q[0] = 1.0; q[1] = 0.0;
    try ctx.stepDeltaNet(0, &q, &k, &v, &g, &beta, &out, 1.0);
    try std.testing.expectApproxEqRel(@as(f32, 0.75), out[0], 1e-5);
    try std.testing.expectApproxEqRel(@as(f32, 0.75), out[1], 1e-5);
}

test "stepDeltaNet: decay shrinks state exponentially" {
    const allocator = std.testing.allocator;
    var cfg = makeCfg(1, 2);
    var ctx = try ssm.SsmCpuContext.init(allocator, &cfg);
    defer ctx.deinit();

    var q = [_]f32{ 1.0, 0.0 };
    var k = [_]f32{ 0.5, 0.5 };
    var v = [_]f32{ 1.0, 1.0 };
    var g_decay = [_]f32{ -1.0 }; // decay = e^-1 ≈ 0.3679
    var beta = [_]f32{1.0};
    var out = [_]f32{ 0.0, 0.0 };

    // After one step with decay -1:
    //   a. S *= e^-1 ≈ 0.3679
    //   b. s_k[v] = 0 (S still zero)
    //   c. d[v] = 1.0
    //   d. S[k,v] = 0.3679*0 + 0.5*1 = 0.5
    //   e. o[v] = S[0,v]*1 = 0.5
    try ctx.stepDeltaNet(0, &q, &k, &v, &g_decay, &beta, &out, 1.0);
    try std.testing.expectApproxEqRel(@as(f32, 0.5), out[0], 1e-5);
    try std.testing.expectApproxEqRel(@as(f32, 0.5), out[1], 1e-5);
}

test "stepDeltaNet: beta=0 makes S a pure outer product" {
    const allocator = std.testing.allocator;
    var cfg = makeCfg(1, 2);
    var ctx = try ssm.SsmCpuContext.init(allocator, &cfg);
    defer ctx.deinit();

    var q = [_]f32{ 1.0, 0.0 };
    var k = [_]f32{ 0.5, 0.5 };
    var v = [_]f32{ 1.0, 1.0 };
    var g = [_]f32{0.0};
    var beta_zero = [_]f32{0.0};
    var out = [_]f32{ 0.0, 0.0 };

    // First step: with beta=0, d[v] = (v - 0) * 0 = 0, so the state never
    // accumulates anything from the input. State stays zero, output is 0.
    try ctx.stepDeltaNet(0, &q, &k, &v, &g, &beta_zero, &out, 1.0);
    try std.testing.expectApproxEqRel(@as(f32, 0.0), out[0], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 0.0), out[1], 1e-6);
}

test "stepDeltaNet: gate clamp prevents overflow" {
    const allocator = std.testing.allocator;
    var cfg = makeCfg(1, 2);
    var ctx = try ssm.SsmCpuContext.init(allocator, &cfg);
    defer ctx.deinit();

    var q = [_]f32{ 1.0, 0.0 };
    var k = [_]f32{ 0.5, 0.5 };
    var v = [_]f32{ 1.0, 1.0 };
    // Without clamping, exp(1000) = inf. With clamp, the result is finite.
    var g_huge = [_]f32{1000.0};
    var beta = [_]f32{1.0};
    var out = [_]f32{ 0.0, 0.0 };

    try ctx.stepDeltaNet(0, &q, &k, &v, &g_huge, &beta, &out, 1.0);
    try std.testing.expect(std.math.isFinite(out[0]));
    try std.testing.expect(std.math.isFinite(out[1]));
}

test "stepDeltaNet: multi-head 2-head routing via num_k_heads" {
    const allocator = std.testing.allocator;
    // 2 v_heads, 1 k_head — K gets replicated to both V heads.
    var cfg = model.ModelConfig{
        .arch = .qwen35,
        .arch_prefix = "qwen35",
        .activation = .silu,
        .n_embd = 256,
        .n_layer = 1,
        .n_heads = 4,
        .n_kv_heads = 1,
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
        .ssm_d_inner = 4,        // 2 v_heads * 2 head_v_dim
        .ssm_d_state = 2,        // head_k_dim
        .ssm_dt_rank = 2,        // num_v_heads
        .ssm_n_group = 1,        // num_k_heads
    };
    var ctx = try ssm.SsmCpuContext.init(allocator, &cfg);
    defer ctx.deinit();

    var q = [_]f32{ 1.0, 0.0, 1.0, 0.0 }; // 2 heads * head_v_dim=2
    var k = [_]f32{ 0.5, 0.5 };            // 1 k_head * head_k_dim=2
    var v = [_]f32{ 1.0, 1.0, 2.0, 2.0 };
    var g = [_]f32{0.0};
    var beta = [_]f32{ 1.0, 1.0 };
    var out = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    try ctx.stepDeltaNet(0, &q, &k, &v, &g, &beta, &out, 1.0);
    // Both heads should see the same k (replicated) but different v.
    // Head 0: same as the single-head test → out = [0.5, 0.5].
    // Head 1: same k but v = [2, 2] → d = 2, S = [[1, 1], [1, 1]], out = [1, 1].
    try std.testing.expectApproxEqRel(@as(f32, 0.5), out[0], 1e-5);
    try std.testing.expectApproxEqRel(@as(f32, 0.5), out[1], 1e-5);
    try std.testing.expectApproxEqRel(@as(f32, 1.0), out[2], 1e-5);
    try std.testing.expectApproxEqRel(@as(f32, 1.0), out[3], 1e-5);
}

test "stepConv1d: rolling window shifts correctly" {
    const allocator = std.testing.allocator;
    // d_conv=4, conv_channels=2 — small for hand verification.
    var cfg = model.ModelConfig{
        .arch = .qwen35,
        .arch_prefix = "qwen35",
        .activation = .silu,
        .n_embd = 256,
        .n_layer = 1,
        .n_heads = 4,
        .n_kv_heads = 1,
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
        .ssm_d_inner = 2,
        .ssm_d_state = 0, // conv_channels = 2 + 2*1*0 = 2
        .ssm_dt_rank = 1,
        .ssm_n_group = 1,
    };
    var ctx = try ssm.SsmCpuContext.init(allocator, &cfg);
    defer ctx.deinit();

    const layer = ctx.getLayer(0);
    try std.testing.expectEqual(@as(usize, 6), layer.conv.len); // (4-1)*2

    // Push 4 chunks [10,20], [30,40], [50,60], [70,80].
    var c1 = [_]f32{ 10.0, 20.0 };
    ctx.stepConv1d(0, &c1);
    var c2 = [_]f32{ 30.0, 40.0 };
    ctx.stepConv1d(0, &c2);
    var c3 = [_]f32{ 50.0, 60.0 };
    ctx.stepConv1d(0, &c3);
    var c4 = [_]f32{ 70.0, 80.0 };
    ctx.stepConv1d(0, &c4);

    // With d_conv=4, the state has 3 slots. After 4 chunks, the oldest (c1) is
    // dropped and state holds the last 3 chunks in arrival order:
    //   Row 0: c2 = [30, 40]  (oldest of the kept set)
    //   Row 1: c3 = [50, 60]
    //   Row 2: c4 = [70, 80]  (newest in state — the *current* chunk is read
    //                           from a separate buffer by the shader)
    const s = ctx.getLayer(0);
    try std.testing.expectEqual(@as(f32, 30.0), s.conv[0]);
    try std.testing.expectEqual(@as(f32, 40.0), s.conv[1]);
    try std.testing.expectEqual(@as(f32, 50.0), s.conv[2]);
    try std.testing.expectEqual(@as(f32, 60.0), s.conv[3]);
    try std.testing.expectEqual(@as(f32, 70.0), s.conv[4]);
    try std.testing.expectEqual(@as(f32, 80.0), s.conv[5]);
}
