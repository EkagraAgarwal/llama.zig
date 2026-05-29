#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer { float data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer UIntBuffer { uint data[]; };

layout(push_constant) uniform PC {
    uint m;
    uint n;
    uint k;
    uint p4;
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer a;
    UIntBuffer b;
    FloatBuffer c;
} pc;

layout(local_size_x = 16, local_size_y = 16) in;
shared float a_tile[16][16];
shared float b_tile[16][16];

uint getByte(UIntBuffer buf, uint idx) {
    uint w = buf.data[idx >> 2];
    return (w >> ((idx & 3u) * 8u)) & 0xFFu;
}

uint getU16(UIntBuffer buf, uint idx) {
    return getByte(buf, idx) | (getByte(buf, idx + 1u) << 8u);
}

float f16Weight(uint row, uint kidx) {
    uint base = (row * pc.k + kidx) * 2u;
    return unpackHalf2x16(getU16(pc.b, base)).x;
}

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    if (row >= pc.m || col >= pc.n) return;
    uint lx = gl_LocalInvocationID.x;
    uint ly = gl_LocalInvocationID.y;
    float sum = 0.0;
    uint tiles = (pc.k + 15u) / 16u;
    for (uint t = 0u; t < tiles; ++t) {
        uint kx = t * 16u + lx;
        uint ky = t * 16u + ly;
        a_tile[ly][lx] = (row < pc.m && kx < pc.k) ? pc.a.data[row * pc.k + kx] : 0.0;
        b_tile[ly][lx] = (col < pc.n && ky < pc.k) ? f16Weight(col, ky) : 0.0;
        barrier();
        for (uint kk = 0u; kk < 16u; ++kk) {
            sum += a_tile[ly][kk] * b_tile[kk][lx];
        }
        barrier();
    }
    pc.c.data[row * pc.n + col] = sum;
}
