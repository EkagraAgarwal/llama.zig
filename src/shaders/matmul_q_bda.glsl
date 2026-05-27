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

layout(local_size_x = 64) in;

uint getByte(UIntBuffer buf, uint idx) {
    uint w = buf.data[idx >> 2];
    uint sh = (idx & 3u) * 8u;
    return (w >> sh) & 0xFFu;
}

uint getU16(UIntBuffer buf, uint idx) {
    return getByte(buf, idx) | (getByte(buf, idx + 1u) << 8u);
}

float f16ToF32(uint h) {
    uint s = (h >> 15u) & 1u;
    uint e = (h >> 10u) & 0x1Fu;
    uint m = h & 0x3FFu;
    uint out_bits;
    if (e == 0u) {
        if (m == 0u) {
            out_bits = s << 31u;
        } else {
            float v = uintBitsToFloat(s << 31u) + float(m) * exp2(-24.0);
            return v;
        }
    } else if (e == 31u) {
        out_bits = (s << 31u) | 0x7F800000u | (m << 13u);
    } else {
        out_bits = (s << 31u) | ((e + 112u) << 23u) | (m << 13u);
    }
    return uintBitsToFloat(out_bits);
}

float q4kWeight(uint row, uint kidx) {
    const uint QK = 256u;
    const uint BS = 144u;
    uint blocks = (pc.k + QK - 1u) / QK;
    uint b = kidx / QK;
    uint i = kidx % QK;
    uint base = (row * blocks + b) * BS;

    float d = f16ToF32(getU16(pc.b, base + 0u));
    float dmin = f16ToF32(getU16(pc.b, base + 2u));
    uint sc = getByte(pc.b, base + 4u + (i / 64u));
    float dl = d * float(int(sc & 0xFu) - 8);
    float ml = dmin * float(int((sc >> 4u) & 0xFu) - 8);

    uint lane = i % 64u;
    uint qbyte = getByte(pc.b, base + 16u + (i / 64u) * 32u + (lane % 32u));
    uint qv = (lane < 32u) ? (qbyte & 0xFu) : ((qbyte >> 4u) & 0xFu);
    return dl * float(qv) - ml;
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
    int sc = int(int(getByte(pc.b, base + 192u + scale_index) << 24) >> 24);

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
    uint col = gl_GlobalInvocationID.x;
    uint row = gl_GlobalInvocationID.y;
    if (row >= pc.m || col >= pc.n) return;

    float sum = 0.0;
    for (uint kk = 0u; kk < pc.k; ++kk) {
        float w = (pc.qtype == 14u) ? q6kWeight(col, kk) : q4kWeight(col, kk);
        sum += pc.a.data[row * pc.k + kk] * w;
    }
    pc.c.data[row * pc.n + col] = sum;
}
