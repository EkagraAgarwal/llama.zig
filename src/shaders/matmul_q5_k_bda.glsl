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
    uint w = buf.data[idx >> 2u];
    return (w >> ((idx & 3u) * 8u)) & 0xFFu;
}

float f16ToF32(uint h) {
    return unpackHalf2x16(h).x;
}

int signExtend5(uint v) {
    if ((v & 0x10u) != 0u) return int(v | 0xFFFFFFE0u);
    return int(v);
}

float q5kWeight(uint row, uint kidx) {
    const uint QK = 256u;
    const uint BS = 176u;
    uint blocks = (pc.k + QK - 1u) / QK;
    uint b = kidx / QK;
    uint i = kidx % QK;
    uint base = (row * blocks + b) * BS;
    uint w_base = base / 4u;

    uint d_dmin = pc.b.data[w_base];
    float d = f16ToF32(d_dmin & 0xFFFFu);
    float dmin = f16ToF32(d_dmin >> 16u);

    uint sw0 = pc.b.data[w_base + 1u];
    uint sw1 = pc.b.data[w_base + 2u];
    uint sw2 = pc.b.data[w_base + 3u];

    uint sb = i / 32u;
    uint is = i % 32u;

    uint sc, mn;
    if (sb == 0u) {
        sc = sw0 & 0x3Fu;
        mn = sw1 & 0x3Fu;
    } else if (sb == 1u) {
        sc = (sw0 >> 8u) & 0x3Fu;
        mn = (sw1 >> 8u) & 0x3Fu;
    } else if (sb == 2u) {
        sc = (sw0 >> 16u) & 0x3Fu;
        mn = (sw1 >> 16u) & 0x3Fu;
    } else if (sb == 3u) {
        sc = (sw0 >> 24u) & 0x3Fu;
        mn = (sw1 >> 24u) & 0x3Fu;
    } else if (sb == 4u) {
        sc = (sw2 & 0x0Fu) | ((sw0 >> 2u) & 0x30u);
        mn = ((sw2 >> 4u) & 0x0Fu) | ((sw1 >> 2u) & 0x30u);
    } else if (sb == 5u) {
        sc = ((sw2 >> 8u) & 0x0Fu) | ((sw0 >> 10u) & 0x30u);
        mn = ((sw2 >> 12u) & 0x0Fu) | ((sw1 >> 10u) & 0x30u);
    } else if (sb == 6u) {
        sc = ((sw2 >> 16u) & 0x0Fu) | ((sw0 >> 18u) & 0x30u);
        mn = ((sw2 >> 20u) & 0x0Fu) | ((sw1 >> 18u) & 0x30u);
    } else {
        sc = ((sw2 >> 24u) & 0x0Fu) | ((sw0 >> 26u) & 0x30u);
        mn = ((sw2 >> 28u) & 0x0Fu) | ((sw1 >> 26u) & 0x30u);
    }

    uint qs_word = pc.b.data[w_base + 12u + (sb >> 1u) * 8u + (is >> 2u)];
    uint qs_byte = (qs_word >> ((is & 3u) * 8u)) & 0xFFu;

    uint qh_word = pc.b.data[w_base + 4u + (is >> 2u)];
    uint qh_byte = (qh_word >> ((is & 3u) * 8u)) & 0xFFu;

    uint qs_shift = (sb & 1u) * 4u;
    uint qb = (qs_byte >> qs_shift) & 0xFu;
    uint qh_bit = (qh_byte >> sb) & 1u;

    uint raw5 = qb | (qh_bit << 4u);

    return d * float(sc) * float(uint(raw5)) - dmin * float(mn);
}

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    if (row >= pc.m || col >= pc.n) return;
    uint lx = gl_LocalInvocationID.x;
    uint ly = gl_LocalInvocationID.y;
    float sum = 0.0f;
    uint tiles = (pc.k + 15u) / 16u;
    for (uint t = 0u; t < tiles; ++t) {
        uint kx = t * 16u + lx;
        uint ky = t * 16u + ly;
        a_tile[ly][lx] = (row < pc.m && kx < pc.k) ? pc.a.data[row * pc.k + kx] : 0.0f;
        b_tile[ly][lx] = (col < pc.n && ky < pc.k) ? q5kWeight(col, ky) : 0.0f;
        barrier();
        for (uint kk = 0u; kk < 16u; ++kk) {
            sum += a_tile[ly][kk] * b_tile[kk][lx];
        }
        barrier();
    }
    pc.c.data[row * pc.n + col] = sum;
}
