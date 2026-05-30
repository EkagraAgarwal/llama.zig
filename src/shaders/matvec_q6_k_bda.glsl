#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_KHR_shader_subgroup_arithmetic : require

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

    float d = f16ToF32(getU16(pc.b, base + 208u));
    return d * float(sc) * float(qv);
}

void main() {
    const uint SUBGROUP_SIZE = 32u;
    const uint COLS_PER_WG = 8u;
    uint lane = gl_LocalInvocationID.x;
    uint subgroup_id = lane / SUBGROUP_SIZE;
    uint col = gl_WorkGroupID.x * COLS_PER_WG + subgroup_id;
    uint base_lane = lane % SUBGROUP_SIZE;

    if (col >= pc.n) return;

    float partial_sum = 0.0;
    uint blocks = (pc.k + 255u) / 256u;
    for (uint bi = 0u; bi < blocks; ++bi) {
        uint k_base = bi * 256u;
        for (uint i = base_lane; i < 256u; i += SUBGROUP_SIZE) {
            if (k_base + i < pc.k) {
                partial_sum += pc.a.data[k_base + i] * q6kWeight(col, k_base + i);
            }
        }
    }

    float sum = subgroupAdd(partial_sum);

    if (base_lane == 0) {
        pc.c.data[col] = sum;
    }
}