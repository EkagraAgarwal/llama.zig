#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n_rows;       // p1: total rows (one workgroup per row)
    uint row_width;    // p2: head_v_dim (RMS norm width per head)
    uint p3;           // p3
    uint eps_bits;     // p4: epsilon as bitcast u32
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer a;     // input  (core SSM output) [row_width, n_v_heads]
    FloatBuffer b;     // gate   (z slice, same shape as a)
    FloatBuffer c;     // rms_norm weight (per head_v_dim row, [row_width])
    FloatBuffer d;     // output (gated norm result) [row_width, n_v_heads]
} pc;
// Note: ssm_gated_norm already had 4 input slots a/b/c/d, so this shader
// didn't need to change.

layout(local_size_x = 64) in;

// ssm_gated_norm: y = silu(z) * rms_norm(x)
// Per-head row: out[row, v] = silu(z[row, v]) * (x[row, v] / sqrt(mean(x[row]^2) + eps))
void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.n_rows) return;
    uint row_width = pc.row_width;
    float eps = uintBitsToFloat(pc.eps_bits);
    uint row_offset = row * row_width;

    if (gl_LocalInvocationID.x != 0u) return;

    float sum_sq = 0.0;
    for (uint i = 0u; i < row_width; ++i) {
        float v = pc.a.data[row_offset + i];
        sum_sq += v * v;
    }
    float inv_rms = 1.0 / sqrt(sum_sq / float(row_width) + eps);

    for (uint i = 0u; i < row_width; ++i) {
        float z = pc.b.data[row_offset + i];
        float silu_z = z / (1.0 + exp(-z));
        float x_norm = pc.a.data[row_offset + i] * inv_rms;
        pc.d.data[row_offset + i] = silu_z * x_norm;
    }
}
