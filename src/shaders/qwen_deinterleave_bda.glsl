#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n;        // p1: total_elements (n_tokens * n_heads * head_dim)
    uint d;        // p2: n_heads
    uint p3;       // p3: head_dim
    uint p4;
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer a; // src: qg (interleaved, size = n_tokens * n_heads * head_dim * 2)
    FloatBuffer b; // dst1: q (de-interleaved, size = n_tokens * n_heads * head_dim)
    FloatBuffer c; // dst2: gate (de-interleaved, size = n_tokens * n_heads * head_dim)
} pc;

layout(local_size_x = 64) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.n) {
        uint token_idx = i / (pc.d * pc.p3);
        uint temp = i % (pc.d * pc.p3);
        uint head_idx = temp / pc.p3;
        uint elem_idx = temp % pc.p3;

        uint qg_offset = token_idx * (2u * pc.d * pc.p3) + head_idx * (2u * pc.p3) + elem_idx;

        pc.b.data[i] = pc.a.data[qg_offset];
        pc.c.data[i] = pc.a.data[qg_offset + pc.p3];
    }
}
