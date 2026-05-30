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

float f16ToF32(uint h) {
    return unpackHalf2x16(h).x;
}

uint getUintUnaligned(UIntBuffer buf, uint byte_idx) {
    uint word_idx = byte_idx >> 2u;
    uint shift = (byte_idx & 3u) * 8u;
    uint w0 = buf.data[word_idx];
    if (shift == 0u) return w0;
    uint w1 = buf.data[word_idx + 1u];
    return (w0 >> shift) | (w1 << (32u - shift));
}

float q6kWeight(uint row, uint kidx) {
    const uint QK = 256u;
    const uint BS = 210u;
    uint blocks = (pc.k + QK - 1u) / QK;
    uint b = kidx / QK;
    uint i = kidx % QK;
    uint base = (row * blocks + b) * BS;

    uint hsel = i / 128u;
    uint local = i % 128u;
    uint l = local % 32u;
    uint quarter = local / 32u;
    uint group = (l / 16u);
    uint scale_index = hsel * 8u + group + quarter * 2u;
    int sc = int(int(getByte(pc.b, base + 192u + scale_index) << 24u) >> 24u);

    uint qh = getByte(pc.b, base + 128u + hsel * 32u + l);
    uint ql0 = getByte(pc.b, base + hsel * 64u + l);
    uint ql1 = getByte(pc.b, base + hsel * 64u + l + 32u);
    int qv = 0;
    if (quarter == 0u) qv = int((ql0 & 0xFu) | (((qh >> 0u) & 0x3u) << 4u)) - 32;
    if (quarter == 1u) qv = int((ql1 & 0xFu) | (((qh >> 2u) & 0x3u) << 4u)) - 32;
    if (quarter == 2u) qv = int(((ql0 >> 4u) & 0xFu) | (((qh >> 4u) & 0x3u) << 4u)) - 32;
    if (quarter == 3u) qv = int(((ql1 >> 4u) & 0xFu) | (((qh >> 6u) & 0x3u) << 4u)) - 32;

    float d = f16ToF32(getUintUnaligned(pc.b, base + 208u) & 0xFFFFu);
    return d * float(sc) * float(qv);
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
        b_tile[ly][lx] = (col < pc.n && ky < pc.k) ? q6kWeight(col, ky) : 0.0;
        barrier();
        for (uint kk = 0u; kk < 16u; ++kk) {
            sum += a_tile[ly][kk] * b_tile[kk][lx];
        }
        barrier();
    }
    pc.c.data[row * pc.n + col] = sum;
}