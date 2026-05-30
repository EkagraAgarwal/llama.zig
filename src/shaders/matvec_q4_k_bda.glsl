#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_ballot : require

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

shared float shared_a[256];

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

    float sum = 0.0;
    uint blocks = (pc.k + 255u) / 256u;

    for (uint bi = 0u; bi < blocks; ++bi) {
        uint k_base = bi * 256u;
        
        if (k_base + lane < pc.k) {
            shared_a[lane] = pc.a.data[k_base + lane];
        } else {
            shared_a[lane] = 0.0;
        }
        barrier();

        if (col < pc.n) {
            uint base = (col * blocks + bi) * 144u;
            uint w_base = base / 4u;

            uint d_dmin = pc.b.data[w_base];
            float d = f16ToF32(d_dmin & 0xFFFFu);
            float min = f16ToF32(d_dmin >> 16u);

            uint sw0 = pc.b.data[w_base + 1u];
            uint sw1 = pc.b.data[w_base + 2u];
            uint sw2 = pc.b.data[w_base + 3u];

            for (uint sb = 0u; sb < 4u; ++sb) {
                uint sc0, m0, sc1, m1;
                if (sb == 0u) {
                    sc0 = sw0 & 0x3Fu;
                    sc1 = (sw0 >> 8u) & 0x3Fu;
                    m0  = sw1 & 0x3Fu;
                    m1  = (sw1 >> 8u) & 0x3Fu;
                } else if (sb == 1u) {
                    sc0 = (sw0 >> 16u) & 0x3Fu;
                    sc1 = (sw0 >> 24u) & 0x3Fu;
                    m0  = (sw1 >> 16u) & 0x3Fu;
                    m1  = (sw1 >> 24u) & 0x3Fu;
                } else if (sb == 2u) {
                    sc0 = (sw2 & 0xFu) | ((sw0 >> 2u) & 0x30u);
                    sc1 = ((sw2 >> 8u) & 0xFu) | ((sw0 >> 10u) & 0x30u);
                    m0  = ((sw2 >> 4u) & 0xFu) | ((sw1 >> 2u) & 0x30u);
                    m1  = ((sw2 >> 12u) & 0xFu) | ((sw1 >> 10u) & 0x30u);
                } else {
                    sc0 = ((sw2 >> 16u) & 0xFu) | ((sw0 >> 18u) & 0x30u);
                    sc1 = ((sw2 >> 24u) & 0xFu) | ((sw0 >> 26u) & 0x30u);
                    m0  = ((sw2 >> 20u) & 0xFu) | ((sw1 >> 18u) & 0x30u);
                    m1  = ((sw2 >> 28u) & 0xFu) | ((sw1 >> 26u) & 0x30u);
                }

                float d1 = d * float(sc0);
                float m1v = min * float(m0);
                float d2 = d * float(sc1);
                float m2v = min * float(m1);

                uint q_word = pc.b.data[w_base + 4u + sb * 8u + (base_lane / 4u)];
                uint qb = (q_word >> ((base_lane & 3u) * 8u)) & 0xFFu;

                uint a_idx = sb * 64u + base_lane;
                sum += shared_a[a_idx] * (d1 * float(qb & 0xFu) - m1v);
                sum += shared_a[a_idx + 32u] * (d2 * float(qb >> 4u) - m2v);
            }
        }
        barrier();
    }

    if (col < pc.n) {
        float final_sum = subgroupClusteredAdd(sum, 32u);
        if (base_lane == 0u) {
            pc.c.data[col] = final_sum;
        }
    }
}
