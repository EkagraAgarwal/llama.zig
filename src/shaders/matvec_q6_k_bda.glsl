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
    return unpackHalf2x16(h).x;
}

float q6kBlockDot(uint row, uint block) {
    const uint QK = 256u;
    const uint BS = 210u;
    uint blocks = (pc.k + QK - 1u) / QK;
    uint base = (row * blocks + block) * BS;
    float d = f16ToF32(getU16(pc.b, base + 208u));
    float sum = 0.0;
    
    for (uint i = 0u; i < 256u; ++i) {
        uint kidx = block * QK + i;
        if (kidx >= pc.k) break;

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

        sum += pc.a.data[kidx] * (d * float(sc) * float(qv));
    }
    return sum;
}

void main() {
    uint col = gl_GlobalInvocationID.x;
    if (col >= pc.n) return;

    float sum = 0.0;
    uint blocks = (pc.k + 255u) / 256u;
    for (uint bi = 0u; bi < blocks; ++bi) {
        sum += q6kBlockDot(col, bi);
    }

    pc.c.data[col] = sum;
}
