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

float q4kWeight(uint row, uint kidx) {
    const uint QK = 256u;
    const uint BS = 144u;
    uint blocks = (pc.k + QK - 1u) / QK;
    uint b = kidx / QK;
    uint i = kidx % QK;
    uint base = (row * blocks + b) * BS;
    uint w_base = base / 4u;

    uint d_dmin = pc.b.data[w_base];
    float d = f16ToF32(d_dmin & 0xFFFFu);
    float min = f16ToF32(d_dmin >> 16u);

    uint s_word_idx = 1u + (i / 128u);
    uint s_word = pc.b.data[w_base + s_word_idx];
    uint m_word = pc.b.data[w_base + s_word_idx + 1u];
    
    uint is = (i / 64u);
    uint sc, m;
    if (is == 0u) {
        sc = s_word & 0x3Fu;
        m  = m_word & 0x3Fu;
    } else if (is == 1u) {
        sc = (s_word >> 8u) & 0x3Fu;
        m  = (m_word >> 8u) & 0x3Fu;
    } else if (is == 2u) {
        sc = (pc.b.data[w_base + 3u] & 0xFu) | ((s_word >> 2u) & 0x30u);
        m  = ((pc.b.data[w_base + 3u] >> 4u) & 0xFu) | ((m_word >> 2u) & 0x30u);
    } else {
        sc = ((pc.b.data[w_base + 3u] >> 16u) & 0xFu) | ((s_word >> 18u) & 0x30u);
        m  = ((pc.b.data[w_base + 3u] >> 20u) & 0xFu) | ((m_word >> 18u) & 0x30u);
    }

    uint q_byte_idx = 16u + (i / 64u) * 32u + (i & 31u);
    uint qb = (pc.b.data[w_base + q_byte_idx / 4u] >> ((q_byte_idx & 3u) * 8u)) & 0xFFu;

    if ((i & 32u) == 0u) {
        return (d * float(sc)) * float(qb & 0xFu) - (min * float(m));
    } else {
        return (d * float(sc)) * float(qb >> 4u) - (min * float(m));
    }
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
        b_tile[ly][lx] = (col < pc.n && ky < pc.k) ? q4kWeight(col, ky) : 0.0;
        barrier();
        for (uint kk = 0u; kk < 16u; ++kk) {
            sum += a_tile[ly][kk] * b_tile[kk][lx];
        }
        barrier();
    }
    pc.c.data[row * pc.n + col] = sum;
}