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

shared float s_sum[256];

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
    const uint LOGICAL_SG_SIZE = 32u;
    const uint COLS_PER_WG = 8u;

    uint lane = gl_LocalInvocationID.x;
    uint sg_id = lane / LOGICAL_SG_SIZE;
    uint sg_lane = lane % LOGICAL_SG_SIZE;
    uint col = gl_WorkGroupID.x * COLS_PER_WG + sg_id;

    float sum = 0.0;

    if (col < pc.n) {
        uint blocks = (pc.k + 255u) / 256u;
        for (uint bi = 0u; bi < blocks; ++bi) {
            uint k_base = bi * 256u;
            for (uint i = sg_lane; i < 256u; i += LOGICAL_SG_SIZE) {
                if (k_base + i < pc.k) {
                    sum += pc.a.data[k_base + i] * q6kWeight(col, k_base + i);
                }
            }
        }
    }

    s_sum[lane] = sum;
    barrier();

    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        if (sg_lane < offset) {
            s_sum[lane] += s_sum[lane + offset];
        }
        barrier();
    }

    if (sg_lane == 0u && col < pc.n) {
        pc.c.data[col] = s_sum[lane];
    }
}
