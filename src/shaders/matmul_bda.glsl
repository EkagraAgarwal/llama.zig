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
    FloatBuffer a;
    FloatBuffer b;
    FloatBuffer c;
} pc;

layout(local_size_x = 16, local_size_y = 16) in;

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint M = pc.n;
    uint N = pc.d;
    uint K = pc.p3;

    if (row < M && col < N) {
        float sum = 0.0;
        for (uint k = 0; k < K; k++) {
            sum += pc.a.data[row * K + k] * pc.b.data[col * K + k];
        }
        pc.c.data[row * N + col] = sum;
    }
}
