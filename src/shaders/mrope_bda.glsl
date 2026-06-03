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
    uint sec0;           // p7
    uint sec1;           // p8
    uint p9;             // p9: sec2 (we use first 8 push slots; sec3 in p10 if needed)
    uint p10;            // p10: sec3
    FloatBuffer a;
    FloatBuffer b;
    FloatBuffer c;
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
    uint byte_off = pc.p5_byte_off;
    uint float_off = byte_off / 4u;
    uint head_idx = i / (pc.head_dim / 2u);
    uint pair_idx = i % (pc.head_dim / 2u);

    uint s0 = pc.sec0;
    uint s1 = s0 + pc.sec1;
    uint s2 = s1 + pc.p9;
    uint section_size;
    uint within;
    if (pair_idx < s0) {
        section_size = pc.sec0;
        within = pair_idx;
    } else if (pair_idx < s1) {
        section_size = pc.sec1;
        within = pair_idx - s0;
    } else if (pair_idx < s2) {
        section_size = pc.p9;
        within = pair_idx - s1;
    } else {
        section_size = pc.p10;
        within = pair_idx - s2;
    }
    if (section_size == 0u) return;

    uint idx0 = float_off + head_idx * pc.head_dim + 2u * pair_idx;
    uint idx1 = idx0 + 1u;

    float inv_freq = pow(rope_base, -2.0 * float(within) / float(section_size));
    float theta = float(pc.pos) * inv_freq;
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float v0 = pc.a.data[idx0];
    float v1 = pc.a.data[idx1];
    pc.c.data[idx0] = v0 * cos_theta - v1 * sin_theta;
    pc.c.data[idx1] = v0 * sin_theta + v1 * cos_theta;
}
