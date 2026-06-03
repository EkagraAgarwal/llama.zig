#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n;        // p1: element count
    uint d;        // p2
    uint p3;
    uint p4;
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer a; // attn output
    FloatBuffer b; // gate
    FloatBuffer c; // out
} pc;

layout(local_size_x = 64) in;

// Qwen3Next/Qwen3.5 attention gating: out = sigmoid(gate) * attn.
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.n) {
        float attn = pc.a.data[i];
        float gate = pc.b.data[i];
        pc.c.data[i] = attn / (1.0 + exp(-gate));
    }
}
