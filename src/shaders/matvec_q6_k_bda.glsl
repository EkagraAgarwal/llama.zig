#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require

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

uint getUintUnaligned(UIntBuffer buf, uint byte_idx) {
    uint word_idx = byte_idx >> 2u;
    uint shift = (byte_idx & 3u) * 8u;
    uint w0 = buf.data[word_idx];
    if (shift == 0u) return w0;
    uint w1 = buf.data[word_idx + 1u];
    return (w0 >> shift) | (w1 << (32u - shift));
}

float f16ToF32(uint h) {
    return unpackHalf2x16(h).x;
}

void main() {
    const uint LOGICAL_SG_SIZE = 32u;
    const uint COLS_PER_WG = 8u;
    uint lane = gl_LocalInvocationID.x;
    uint subgroup_id = lane / LOGICAL_SG_SIZE;
    uint col = gl_WorkGroupID.x * COLS_PER_WG + subgroup_id;
    uint base_lane = lane % LOGICAL_SG_SIZE;

    if (col >= pc.n) return;

    float sum = 0.0;
    uint blocks = (pc.k + 255u) / 256u;

    for (uint bi = 0u; bi < blocks; ++bi) {
        uint k_base = bi * 256u;
        uint base = (col * blocks + bi) * 210u;

        float d = f16ToF32(getUintUnaligned(pc.b, base + 208u) & 0xFFFFu);

        uint sw0 = getUintUnaligned(pc.b, base + 192u);
        uint sw1 = getUintUnaligned(pc.b, base + 196u);
        uint sw2 = getUintUnaligned(pc.b, base + 200u);
        uint sw3 = getUintUnaligned(pc.b, base + 204u);

        for (uint hsel = 0u; hsel < 2u; ++hsel) {
            uint qh_base = base + 128u + hsel * 32u;
            uint ql0_base = base + hsel * 64u;
            uint ql1_base = base + hsel * 64u + 32u;

            uint l = base_lane;
            uint group = l / 16u;

            uint qh = getByte(pc.b, qh_base + l);
            uint ql0 = getByte(pc.b, ql0_base + l);
            uint ql1 = getByte(pc.b, ql1_base + l);

            int qv0 = int((ql0 & 0xFu) | (((qh >> 0u) & 0x3u) << 4u)) - 32;
            int qv1 = int((ql1 & 0xFu) | (((qh >> 2u) & 0x3u) << 4u)) - 32;
            int qv2 = int(((ql0 >> 4u) & 0xFu) | (((qh >> 4u) & 0x3u) << 4u)) - 32;
            int qv3 = int(((ql1 >> 4u) & 0xFu) | (((qh >> 6u) & 0x3u) << 4u)) - 32;

            int sc0, sc1, sc2, sc3;
            if (hsel == 0u) {
                sc0 = int(int((sw0 >> (group * 8u)) << 24u) >> 24u);
                sc1 = int(int((sw0 >> ((group + 2u) * 8u)) << 24u) >> 24u);
                sc2 = int(int((sw1 >> (group * 8u)) << 24u) >> 24u);
                sc3 = int(int((sw1 >> ((group + 2u) * 8u)) << 24u) >> 24u);
            } else {
                sc0 = int(int((sw2 >> (group * 8u)) << 24u) >> 24u);
                sc1 = int(int((sw2 >> ((group + 2u) * 8u)) << 24u) >> 24u);
                sc2 = int(int((sw3 >> (group * 8u)) << 24u) >> 24u);
                sc3 = int(int((sw3 >> ((group + 2u) * 8u)) << 24u) >> 24u);
            }

            uint k0 = k_base + hsel * 128u + 0u * 32u + l;
            uint k1 = k_base + hsel * 128u + 1u * 32u + l;
            uint k2 = k_base + hsel * 128u + 2u * 32u + l;
            uint k3 = k_base + hsel * 128u + 3u * 32u + l;

            if (k0 < pc.k) sum += pc.a.data[k0] * (d * float(sc0) * float(qv0));
            if (k1 < pc.k) sum += pc.a.data[k1] * (d * float(sc1) * float(qv1));
            if (k2 < pc.k) sum += pc.a.data[k2] * (d * float(sc2) * float(qv2));
            if (k3 < pc.k) sum += pc.a.data[k3] * (d * float(sc3) * float(qv3));
        }
    }

    float final_sum = subgroupClusteredAdd(sum, 32u);
    if (base_lane == 0u) {
        pc.c.data[col] = final_sum;
    }
}
