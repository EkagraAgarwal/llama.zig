#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_shuffle : require

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

float f16ToF32(uint h) {
    return unpackHalf2x16(h).x;
}

int signExtend5(uint v) {
    if ((v & 0x10u) != 0u) return int(v | 0xFFFFFFE0u);
    return int(v);
}

uint getByte(UIntBuffer buf, uint idx) {
    uint w = buf.data[idx >> 2u];
    return (w >> ((idx & 3u) * 8u)) & 0xFFu;
}

void main() {
    const uint LOGICAL_SG_SIZE = 32u;
    const uint COLS_PER_WG = 8u;
    uint lane = gl_LocalInvocationID.x;
    uint subgroup_id = lane / LOGICAL_SG_SIZE;
    uint col = gl_WorkGroupID.x * COLS_PER_WG + subgroup_id;
    uint base_lane = lane % LOGICAL_SG_SIZE;

    float sum = 0.0f;
    uint blocks = (pc.k + 255u) / 256u;

    for (uint bi = 0u; bi < blocks; ++bi) {
        uint k_base = bi * 256u;

        if (k_base + lane < pc.k) {
            shared_a[lane] = pc.a.data[k_base + lane];
        } else {
            shared_a[lane] = 0.0f;
        }
        barrier();

        if (col < pc.n) {
            uint base = (col * blocks + bi) * 176u;
            uint w_base = base / 4u;

            uint d_dmin = pc.b.data[w_base];
            float d = f16ToF32(d_dmin & 0xFFFFu);
            float dmin = f16ToF32(d_dmin >> 16u);

            uint sw0 = pc.b.data[w_base + 1u];
            uint sw1 = pc.b.data[w_base + 2u];
            uint sw2 = pc.b.data[w_base + 3u];

            for (uint sb = 0u; sb < 8u; ++sb) {
                uint sc, mn;
                if (sb == 0u) {
                    sc = sw0 & 0x3Fu; mn = sw1 & 0x3Fu;
                } else if (sb == 1u) {
                    sc = (sw0 >> 8u) & 0x3Fu; mn = (sw1 >> 8u) & 0x3Fu;
                } else if (sb == 2u) {
                    sc = (sw0 >> 16u) & 0x3Fu; mn = (sw1 >> 16u) & 0x3Fu;
                } else if (sb == 3u) {
                    sc = (sw0 >> 24u) & 0x3Fu; mn = (sw1 >> 24u) & 0x3Fu;
                } else if (sb == 4u) {
                    sc = (sw2 & 0x0Fu) | ((sw0 >> 2u) & 0x30u); mn = ((sw2 >> 4u) & 0x0Fu) | ((sw1 >> 2u) & 0x30u);
                } else if (sb == 5u) {
                    sc = ((sw2 >> 8u) & 0x0Fu) | ((sw0 >> 10u) & 0x30u); mn = ((sw2 >> 12u) & 0x0Fu) | ((sw1 >> 10u) & 0x30u);
                } else if (sb == 6u) {
                    sc = ((sw2 >> 16u) & 0x0Fu) | ((sw0 >> 18u) & 0x30u); mn = ((sw2 >> 20u) & 0x0Fu) | ((sw1 >> 18u) & 0x30u);
                } else {
                    sc = ((sw2 >> 24u) & 0x0Fu) | ((sw0 >> 26u) & 0x30u); mn = ((sw2 >> 28u) & 0x0Fu) | ((sw1 >> 26u) & 0x30u);
                }

                float scf = d * float(sc);
                float mnf = dmin * float(mn);

                uint qs_word = pc.b.data[w_base + 12u + ((sb >> 1) * 8u) + ((base_lane >> 2) & 7u)];
                uint qs_byte = (qs_word >> ((base_lane & 3u) * 8u)) & 0xFFu;

                uint qh_word = pc.b.data[w_base + 4u + ((sb >> 1) * 8u) + ((base_lane >> 2) & 7u)];
                uint qh_byte = (qh_word >> ((base_lane & 3u) * 8u)) & 0xFFu;

                uint qs_shift = (sb & 1u) * 4u;
                uint qb = (qs_byte >> qs_shift) & 0xFu;
                uint qh_bit = (qh_byte >> ((sb & 6u) | (sb & 1u))) & 1u;

                uint raw5 = qb | (qh_bit << 4u);

                uint a_idx = sb * 32u + base_lane;
                sum += shared_a[a_idx] * (scf * float(uint(raw5)) - mnf);
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
