#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n;
    uint d;
    uint p3;
    uint p4;
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer a; // gate (to be gelu'd)
    FloatBuffer b; // up (to be multiplied)
    FloatBuffer c; // output
} pc;

layout(local_size_x = 64) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.n) {
        float x = pc.a.data[i];
        float w = pc.b.data[i];
        // gelu(x) * w
        // tanh approximation: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float SQRT_2_PI = 0.7978845608;
        float x3 = x * x * x;
        pc.c.data[i] = 0.5 * x * (1.0 + tanh(SQRT_2_PI * (x + 0.044715 * x3))) * w;
    }
}
