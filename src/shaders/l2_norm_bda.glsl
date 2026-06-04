#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n_rows;     // p1: number of rows (one workgroup per row)
    uint row_width;  // p2: width of each row
    uint eps_bits;   // p3: epsilon as u32 bitcast
    uint p4;
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer a;
    FloatBuffer b;
    FloatBuffer c;
} pc;

layout(local_size_x = 64) in;

// L2-normalize each row independently: out = x / sqrt(sum(x^2) + eps).
// Single-thread reduction per row (row_width <= 256 typically).
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
    float inv_norm = 1.0 / sqrt(sum_sq + eps);

    for (uint i = 0u; i < row_width; ++i) {
        pc.c.data[row_offset + i] = pc.a.data[row_offset + i] * inv_norm;
    }
}
