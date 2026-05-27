#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) buffer UIntBuffer {
    uint data[];
};

layout(push_constant) uniform PC {
    uint n;
    uint k;
    uint p3;
    uint p4;
    uint p5;
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

void dequantQ6KBlock(uint row, uint block, out float vals[256]) {
    const uint BS = 210u;
    uint blocks = (pc.k + 255u) / 256u;
    uint base = (row * blocks + block) * BS;

    float d = f16ToF32(getU16(pc.b, base + 208u));

    for (uint n = 0u; n < 256u; n += 128u) {
        for (uint l = 0u; l < 32u; ++l) {
            uint qh_v = getByte(pc.b, base + 128u + (n / 4u) + l);
            uint ql0 = getByte(pc.b, base + (n / 2u) + l);
            uint ql1 = getByte(pc.b, base + (n / 2u) + l + 32u);
            uint is = l / 16u;

            int sc1 = int(int(getByte(pc.b, base + 192u + is + 0u) << 24) >> 24);
            int sc2 = int(int(getByte(pc.b, base + 192u + is + 2u) << 24) >> 24);
            int sc3 = int(int(getByte(pc.b, base + 192u + is + 4u) << 24) >> 24);
            int sc4 = int(int(getByte(pc.b, base + 192u + is + 6u) << 24) >> 24);

            int q1 = int((ql0 & 0xFu) | (((qh_v >> 0u) & 0x3u) << 4u)) - 32;
            int q2 = int((ql1 & 0xFu) | (((qh_v >> 2u) & 0x3u) << 4u)) - 32;
            int q3 = int(((ql0 >> 4u) & 0xFu) | (((qh_v >> 4u) & 0x3u) << 4u)) - 32;
            int q4 = int(((ql1 >> 4u) & 0xFu) | (((qh_v >> 6u) & 0x3u) << 4u)) - 32;

            vals[n + l] = d * float(sc1) * float(q1);
            vals[n + l + 32u] = d * float(sc2) * float(q2);
            vals[n + l + 64u] = d * float(sc3) * float(q3);
            vals[n + l + 96u] = d * float(sc4) * float(q4);
        }
    }
}

void main() {
    uint col = gl_GlobalInvocationID.x;
    if (col >= pc.n) return;

    float sum = 0.0;
    uint nblocks = (pc.k + 255u) / 256u;
    for (uint bi = 0u; bi < nblocks; ++bi) {
        float wblk[256];
        dequantQ6KBlock(col, bi, wblk);
        uint k0 = bi * 256u;
        uint kend = min(k0 + 256u, pc.k);
        for (uint kk = k0; kk < kend; ++kk) {
            sum += pc.a.data[kk] * wblk[kk - k0];
        }
    }
    pc.c.data[col] = sum;
}
