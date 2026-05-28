#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer { float data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer UIntBuffer { uint data[]; };

layout(push_constant) uniform PC {
    uint m; // p1
    uint n; // p2
    uint k; // p3
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

float q80BlockDot(uint row, uint block, uint k_rem) {
    uint blocks = (pc.k + 31u) / 32u;
    uint base = (row * blocks + block) * 34u;
    float d = f16ToF32(getU16(pc.b, base + 0u));
    float sum = 0.0;
    uint k0 = block * 32u;
    for (uint i = 0u; i < 32u; ++i) {
        if (k0 + i < pc.k) {
            int qv = int(int(getByte(pc.b, base + 2u + i) << 24u) >> 24u);
            sum += pc.a.data[k0 + i] * (d * float(qv));
        }
    }
    return sum;
}

void main() {
    uint col = gl_GlobalInvocationID.x;
    if (col >= pc.n) return;
    float sum = 0.0;
    uint nblocks = (pc.k + 31u) / 32u;
    for (uint bi = 0u; bi < nblocks; ++bi) {
        sum += q80BlockDot(col, bi, pc.k);
    }
    pc.c.data[col] = sum;
}
