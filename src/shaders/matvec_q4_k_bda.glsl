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

void dequantQ4KBlock(uint row, uint block, out float vals[256]) {
    const uint BS = 144u;
    uint blocks = (pc.k + 255u) / 256u;
    uint base = (row * blocks + block) * BS;

    float d = f16ToF32(getU16(pc.b, base + 0u));
    float dmin = f16ToF32(getU16(pc.b, base + 2u));

    for (uint is = 0u; is < 4u; ++is) {
        uint sc = getByte(pc.b, base + 4u + is);
        float dl = d * float(int(sc & 0xFu) - 8);
        float ml = dmin * float(int((sc >> 4u) & 0xFu) - 8);
        for (uint j = 0u; j < 32u; ++j) {
            uint q_idx = is * 32u + j;
            uint qbyte = getByte(pc.b, base + 16u + q_idx);
            vals[is * 64u + j] = dl * float(qbyte & 0xFu) - ml;
            vals[is * 64u + j + 32u] = dl * float(qbyte >> 4u) - ml;
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
        dequantQ4KBlock(col, bi, wblk);
        uint k0 = bi * 256u;
        uint kend = min(k0 + 256u, pc.k);
        for (uint kk = k0; kk < kend; ++kk) {
            sum += pc.a.data[kk] * wblk[kk - k0];
        }
    }
    pc.c.data[col] = sum;
}
