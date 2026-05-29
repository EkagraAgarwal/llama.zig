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
    uint qtype;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer a;
    UIntBuffer b;
    FloatBuffer c;
} pc;

layout(local_size_x = 256) in;

uint getByte(UIntBuffer buf, uint idx) {
    uint w = buf.data[idx >> 2];
    return (w >> ((idx & 3u) * 8u)) & 0xFFu;
}

uint getU16(UIntBuffer buf, uint idx) {
    return getByte(buf, idx) | (getByte(buf, idx + 1u) << 8u);
}

float f16ToF32(uint h) {
    uint s = (h >> 15u) & 1u;
    uint e = (h >> 10u) & 0x1Fu;
    uint m = h & 0x3FFu;
    if (e == 0u) {
        if (m == 0u) return uintBitsToFloat(s << 31u);
        return uintBitsToFloat(s << 31u) + float(m) * exp2(-24.0);
    }
    if (e == 31u) return uintBitsToFloat((s << 31u) | 0x7F800000u | (m << 13u));
    return uintBitsToFloat((s << 31u) | ((e + 112u) << 23u) | (m << 13u));
}

float q4kWeight(uint row, uint kidx) {
    const uint QK = 256u;
    const uint BS = 144u;
    uint blocks = (pc.k + QK - 1u) / QK;
    uint b = kidx / QK;
    uint i = kidx % QK;
    uint base = (row * blocks + b) * BS;

    float d = f16ToF32(getU16(pc.b, base + 0u));
    float min = f16ToF32(getU16(pc.b, base + 2u));

    uint sbase = base + 4u;
    uint is = (i / 64u) * 2u;
    uint sc0, m0, sc1, m1;
    if (is + 0u < 4u) {
        sc0 = getByte(pc.b, sbase + is + 0u) & 0x3Fu;
        m0 = getByte(pc.b, sbase + is + 4u) & 0x3Fu;
    } else {
        sc0 = (getByte(pc.b, sbase + is + 4u) & 0xFu) | ((getByte(pc.b, sbase + is - 4u) >> 6u) << 4u);
        m0 = (getByte(pc.b, sbase + is + 4u) >> 4u) | ((getByte(pc.b, sbase + is - 0u) >> 6u) << 4u);
    }
    if (is + 1u < 4u) {
        sc1 = getByte(pc.b, sbase + is + 1u) & 0x3Fu;
        m1 = getByte(pc.b, sbase + is + 5u) & 0x3Fu;
    } else {
        sc1 = (getByte(pc.b, sbase + is + 5u) & 0xFu) | ((getByte(pc.b, sbase + is - 3u) >> 6u) << 4u);
        m1 = (getByte(pc.b, sbase + is + 5u) >> 4u) | ((getByte(pc.b, sbase + is + 1u) >> 6u) << 4u);
    }

    float d1 = d * float(sc0);
    float m1v = min * float(m0);
    float d2 = d * float(sc1);
    float m2v = min * float(m1);

    uint qbase = base + 16u + (i / 64u) * 32u + (i & 31u);
    uint qb = getByte(pc.b, qbase);
    uint q4l = qb & 0xFu;
    uint q4h = qb >> 4u;

    if ((i & 32u) == 0u) {
        return d1 * float(q4l) - m1v;
    } else {
        return d2 * float(q4h) - m2v;
    }
}

void main() {
    uint col = gl_GlobalInvocationID.x;
    if (col >= pc.n) return;
    float sum = 0.0;
    uint blocks = (pc.k + 255u) / 256u;
    for (uint bi = 0u; bi < blocks; ++bi) {
        uint k_base = bi * 256u;
        for (uint i = 0u; i < 256u; ++i) {
            if (k_base + i < pc.k) {
                sum += pc.a.data[k_base + i] * q4kWeight(col, k_base + i);
            }
        }
    }
    pc.c.data[col] = sum;
}