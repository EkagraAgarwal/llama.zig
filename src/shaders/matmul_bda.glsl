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
    FloatBuffer a;
    FloatBuffer b;
    FloatBuffer c;
} pc;

layout(local_size_x = 16, local_size_y = 16) in;
shared float a_tile[16][16];
shared float b_tile[16][16];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint M = pc.n;
    uint N = pc.d;
    uint K = pc.p3;

    uint lx = gl_LocalInvocationID.x;
    uint ly = gl_LocalInvocationID.y;

    float sum = 0.0;
    uint tiles = (K + 15u) / 16u;
    for (uint t = 0; t < tiles; ++t) {
        uint kx = t * 16u + lx;
        uint ky = t * 16u + ly;

        a_tile[ly][lx] = (row < M && kx < K) ? pc.a.data[row * K + kx] : 0.0;
        b_tile[ly][lx] = (col < N && ky < K) ? pc.b.data[col * K + ky] : 0.0;

        barrier();
        for (uint k = 0; k < 16u; ++k) {
            sum += a_tile[ly][k] * b_tile[k][lx];
        }
        barrier();
    }

    if (row < M && col < N) {
        pc.c.data[row * N + col] = sum;
    }
}
