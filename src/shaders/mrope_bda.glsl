#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n_heads;        // p1
    uint head_dim;       // p2
    uint pos;            // p3
    uint rope_theta_bits; // p4
    uint p5_byte_off;    // p5: byte offset into a/c (divided by 4 inside)
    uint sec0;           // p6
    uint sec1;           // p7
    uint sec2;           // p8
    FloatBuffer a;       // offset 32 bytes (pc.a in Zig)
    FloatBuffer b;       // offset 40 bytes (pc.b in Zig, unused)
    FloatBuffer c;       // offset 48 bytes (pc.c in Zig)
} pc;

layout(local_size_x = 64) in;

// Multi-axis RoPE. For each head, rotates pairs (idx0, idx1) using a per-pair
// angle theta = pos * inv_freq where inv_freq depends on which section the
// pair belongs to (sec0..sec3).
//
// Push-constant layout matches the rope shader's BDA + offset convention.
// a, c are at offset `p5_byte_off / 4` floats from the start of the
// activation tensor.
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n_heads * (pc.head_dim / 2u)) return;

    float rope_base = uintBitsToFloat(pc.rope_theta_bits);
    uint head_idx = i / (pc.head_dim / 2u);
    uint pair_idx = i % (pc.head_dim / 2u);

    uint s0 = pc.sec0;
    uint s1 = s0 + pc.sec1;
    uint s2 = s1 + pc.sec2;
    uint section_size;
    uint within_pair;
    uint dim_idx = 2u * pair_idx;
    if (dim_idx < s0) {
        section_size = pc.sec0;
        within_pair = pair_idx;
    } else if (dim_idx < s1) {
        section_size = pc.sec1;
        within_pair = pair_idx - s0 / 2u;
    } else if (dim_idx < s2) {
        section_size = pc.sec2;
        within_pair = pair_idx - s1 / 2u;
    } else {
        section_size = pc.head_dim - pc.sec0 - pc.sec1 - pc.sec2;
        within_pair = pair_idx - s2 / 2u;
    }
    if (section_size == 0u) return;

    // NeoX-style (split-half) indexing: (pair_idx, pair_idx + head_dim/2)
    uint idx0 = head_idx * pc.head_dim + pair_idx;
    uint idx1 = idx0 + (pc.head_dim / 2u);

    // Frequencies are relative to section start, but divisor is full head_dim
    float inv_freq = pow(rope_base, -2.0 * float(within_pair) / float(pc.head_dim));
    float theta = float(pc.pos) * inv_freq;
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float v0 = pc.a.data[idx0];
    float v1 = pc.a.data[idx1];
    pc.c.data[idx0] = v0 * cos_theta - v1 * sin_theta;
    pc.c.data[idx1] = v0 * sin_theta + v1 * cos_theta;
}
