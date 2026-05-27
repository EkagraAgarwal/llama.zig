/// CPU reference roundtrip tests for core math ops.
/// These tests verify the formulas used in the GLSL shaders by computing
/// the same operations on CPU and checking agreement.
const std = @import("std");
const testing = std.testing;

/// RMSNorm: out[i] = in[i] / rms(in) * weight[i], where rms = sqrt(mean(in^2) + eps)
fn rmsNorm(in: []const f32, weight: []const f32, out: []f32, eps: f32) void {
    std.debug.assert(in.len == weight.len and in.len == out.len);
    const n: f32 = @floatFromInt(in.len);
    var sum_sq: f32 = 0;
    for (in) |v| sum_sq += v * v;
    const rms_inv = 1.0 / @sqrt(sum_sq / n + eps);
    for (0..in.len) |i| out[i] = in[i] * rms_inv * weight[i];
}

/// Matmul (batched row-vector): C[row, col] = sum_k A[row, k] * B[col, k]
/// A: [M x K], B: [N x K], C: [M x N]
fn matmul(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0;
            for (0..k) |ki| sum += a[row * k + ki] * b[col * k + ki];
            c[row * n + col] = sum;
        }
    }
}

/// ScaledAdd: out[i] = a[i] + scale * b[i]
fn scaledAdd(a: []const f32, b: []const f32, out: []f32, scale: f32) void {
    std.debug.assert(a.len == b.len and a.len == out.len);
    for (0..a.len) |i| out[i] = a[i] + scale * b[i];
}

/// SiLU activation: silu(x) = x / (1 + exp(-x))
fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// SiLU-gated product: out[i] = silu(gate[i]) * up[i]
fn siluMul(gate: []const f32, up: []const f32, out: []f32) void {
    std.debug.assert(gate.len == up.len and gate.len == out.len);
    for (0..gate.len) |i| out[i] = silu(gate[i]) * up[i];
}

test "rmsNorm basic" {
    const n = 8;
    const eps: f32 = 1e-5;
    const input = [n]f32{ 1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0 };
    const weight = [n]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var out = std.mem.zeroes([n]f32);

    rmsNorm(&input, &weight, &out, eps);

    // rms = sqrt((1+4+9+16+1+4+9+16)/8 + eps) = sqrt(60/8 + eps) = sqrt(7.5 + eps)
    const rms = @sqrt(@as(f32, 60.0) / @as(f32, n) + eps);
    for (0..n) |i| {
        const expected = input[i] / rms;
        try testing.expectApproxEqRel(expected, out[i], 1e-5);
    }
}

test "rmsNorm with weights" {
    const n = 4;
    const eps: f32 = 1e-6;
    const input = [n]f32{ 2.0, -1.0, 0.5, 3.0 };
    const weight = [n]f32{ 0.5, 2.0, 1.0, 0.25 };
    var out = std.mem.zeroes([n]f32);

    rmsNorm(&input, &weight, &out, eps);

    var sum_sq: f32 = 0;
    for (input) |v| sum_sq += v * v;
    const rms_inv = 1.0 / @sqrt(sum_sq / @as(f32, n) + eps);
    for (0..n) |i| {
        const expected = input[i] * rms_inv * weight[i];
        try testing.expectApproxEqRel(expected, out[i], 1e-5);
    }
}

test "matmul identity" {
    // vec (1x4) @ identity^T (4x4) → out (1x4): output should equal input
    const m: usize = 1;
    const n: usize = 4;
    const k: usize = 4;
    const a = [m * k]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [n * k]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    };
    var c = std.mem.zeroes([m * n]f32);

    matmul(&a, &b, &c, m, n, k);

    try testing.expectApproxEqRel(@as(f32, 1.0), c[0], 1e-6);
    try testing.expectApproxEqRel(@as(f32, 2.0), c[1], 1e-6);
    try testing.expectApproxEqRel(@as(f32, 3.0), c[2], 1e-6);
    try testing.expectApproxEqRel(@as(f32, 4.0), c[3], 1e-6);
}

test "matmul small" {
    const m: usize = 1;
    const n: usize = 3;
    const k: usize = 2;
    const a = [m * k]f32{ 1.0, 2.0 };
    // b[row] = each output row of the weight matrix (transposed)
    const b = [n * k]f32{ 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
    var c = std.mem.zeroes([m * n]f32);

    matmul(&a, &b, &c, m, n, k);

    // c[0] = dot([1,2], [1,0]) = 1
    // c[1] = dot([1,2], [0,1]) = 2
    // c[2] = dot([1,2], [1,1]) = 3
    try testing.expectApproxEqRel(@as(f32, 1.0), c[0], 1e-6);
    try testing.expectApproxEqRel(@as(f32, 2.0), c[1], 1e-6);
    try testing.expectApproxEqRel(@as(f32, 3.0), c[2], 1e-6);
}

test "scaledAdd" {
    const n = 6;
    const a = [n]f32{ 1.0, 2.0, 3.0, -1.0, -2.0, 0.0 };
    const b = [n]f32{ 0.5, 1.0, -1.0, 2.0, 0.0, 3.0 };
    var out = std.mem.zeroes([n]f32);
    const scale: f32 = 0.263;

    scaledAdd(&a, &b, &out, scale);

    for (0..n) |i| {
        const expected = a[i] + scale * b[i];
        try testing.expectApproxEqRel(expected, out[i], 1e-6);
    }
}

test "scaledAdd identity scale=1" {
    const n = 4;
    const a = [n]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [n]f32{ 1.0, 1.0, 1.0, 1.0 };
    var out = std.mem.zeroes([n]f32);

    scaledAdd(&a, &b, &out, 1.0);

    for (0..n) |i| {
        try testing.expectApproxEqRel(a[i] + b[i], out[i], 1e-6);
    }
}

test "siluMul" {
    const n = 4;
    const gate = [n]f32{ 0.0, 1.0, -1.0, 2.0 };
    const up = [n]f32{ 1.0, 2.0, 0.5, -1.0 };
    var out = std.mem.zeroes([n]f32);

    siluMul(&gate, &up, &out);

    for (0..n) |i| {
        const expected = silu(gate[i]) * up[i];
        try testing.expectApproxEqRel(expected, out[i], 1e-5);
    }
}

test "rmsnorm + matmul + scaledAdd transformer residual roundtrip" {
    // Simulate one transformer residual step: out = input + scale * proj(rmsnorm(input))
    // This mirrors what a single attention/FFN sub-layer does in the compute graph.
    const n = 4;
    const scale: f32 = 0.263;
    const eps: f32 = 1e-5;

    const input = [n]f32{ 1.0, -0.5, 2.0, 0.3 };
    const norm_weight = [n]f32{ 1.0, 1.0, 1.0, 1.0 };
    var normed = std.mem.zeroes([n]f32);
    rmsNorm(&input, &norm_weight, &normed, eps);

    // Identity projection: proj_out == normed
    const proj = [n * n]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    };
    var proj_out = std.mem.zeroes([n]f32);
    matmul(&normed, &proj, &proj_out, 1, n, n);

    for (0..n) |i| {
        try testing.expectApproxEqRel(normed[i], proj_out[i], 1e-5);
    }

    // Residual add
    var out = std.mem.zeroes([n]f32);
    scaledAdd(&input, &proj_out, &out, scale);

    for (0..n) |i| {
        const expected = input[i] + scale * normed[i];
        try testing.expectApproxEqRel(expected, out[i], 1e-5);
    }
}

test "rmsnorm + matmul + scaledAdd numerical" {
    // Use non-trivial weights to verify correctness more thoroughly
    const allocator = testing.allocator;
    const n = 8;
    const scale: f32 = 0.263;
    const eps: f32 = 1e-5;

    // Random-ish inputs (deterministic)
    const input = [n]f32{ 0.3, -1.2, 0.7, 2.1, -0.5, 1.8, -0.9, 0.4 };
    const norm_weight = [n]f32{ 1.1, 0.9, 1.2, 0.8, 1.0, 1.3, 0.7, 1.1 };

    const normed = try allocator.alloc(f32, n);
    defer allocator.free(normed);
    rmsNorm(&input, &norm_weight, normed, eps);

    // Verify normed vector has correct RMS after normalization
    var sum_sq: f32 = 0;
    for (input) |v| sum_sq += v * v;
    const rms_inv = 1.0 / @sqrt(sum_sq / @as(f32, n) + eps);
    for (0..n) |i| {
        const expected = input[i] * rms_inv * norm_weight[i];
        try testing.expectApproxEqRel(expected, normed[i], 1e-5);
    }

    // Simple weight matrix: scaling matrix (diag = 2)
    var weight_matrix = try allocator.alloc(f32, n * n);
    defer allocator.free(weight_matrix);
    for (weight_matrix) |*w| w.* = 0;
    for (0..n) |i| weight_matrix[i * n + i] = 2.0;

    const projected = try allocator.alloc(f32, n);
    defer allocator.free(projected);
    matmul(normed, weight_matrix, projected, 1, n, n);

    // With scaling=2 matrix, projected = 2 * normed
    for (0..n) |i| {
        try testing.expectApproxEqRel(normed[i] * 2.0, projected[i], 1e-5);
    }

    // Residual
    const out = try allocator.alloc(f32, n);
    defer allocator.free(out);
    scaledAdd(&input, projected, out, scale);

    for (0..n) |i| {
        const expected = input[i] + scale * (normed[i] * 2.0);
        try testing.expectApproxEqRel(expected, out[i], 1e-5);
    }
}
