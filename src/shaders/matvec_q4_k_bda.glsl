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
        uint base = (col * blocks + bi) * 144u;

        float d = f16ToF32(getU16(pc.b, base + 0u));
        float min = f16ToF32(getU16(pc.b, base + 2u));
        uint sbase = base + 4u;
        uint qbase = base + 16u;

        for (uint sb = 0u; sb < 4u; ++sb) {
            uint is = sb * 2u;
            uint sc0, m0, sc1, m1;

            if (is < 4u) {
                sc0 = getByte(pc.b, sbase + is + 0u) & 0x3Fu;
                m0  = getByte(pc.b, sbase + is + 4u) & 0x3Fu;
                sc1 = getByte(pc.b, sbase + is + 1u) & 0x3Fu;
                m1  = getByte(pc.b, sbase + is + 5u) & 0x3Fu;
            } else {
                sc0 = (getByte(pc.b, sbase + is + 4u) & 0xFu) | ((getByte(pc.b, sbase + is - 4u) >> 6u) << 4u);
                m0  = (getByte(pc.b, sbase + is + 4u) >> 4u) | ((getByte(pc.b, sbase + is - 0u) >> 6u) << 4u);
                sc1 = (getByte(pc.b, sbase + is + 5u) & 0xFu) | ((getByte(pc.b, sbase + is - 3u) >> 6u) << 4u);
                m1  = (getByte(pc.b, sbase + is + 5u) >> 4u) | ((getByte(pc.b, sbase + is + 1u) >> 6u) << 4u);
            }

            float d1 = d * float(sc0);
            float m1v = min * float(m0);
            float d2 = d * float(sc1);
            float m2v = min * float(m1);

            uint sub_qbase = qbase + sb * 32u;

            uint qb = getByte(pc.b, sub_qbase + base_lane);

            uint k_idx0 = k_base + sb * 64u + base_lane;
            if (k_idx0 < pc.k) {
                float w0 = d1 * float(qb & 0xFu) - m1v;
                sum += pc.a.data[k_idx0] * w0;
            }

            uint k_idx1 = k_base + sb * 64u + base_lane + 32u;
            if (k_idx1 < pc.k) {
                float w1 = d2 * float(qb >> 4u) - m2v;
                sum += pc.a.data[k_idx1] * w1;
            }
        }
    }

    float final_sum = subgroupClusteredAdd(sum, 32u);
    if (base_lane == 0u) {
        pc.c.data[col] = final_sum;
    }
}
