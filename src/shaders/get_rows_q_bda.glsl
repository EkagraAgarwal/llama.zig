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
    uint n_embd;
    uint n_tokens;
    uint qtype;
    uint scale_bits;
    uint row_bytes;
    uint p6;
    uint p7;
    uint p8;
    UIntBuffer indices;
    UIntBuffer weights;
    FloatBuffer out_buf;
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

float bf16ToF32(uint h) {
    return uintBitsToFloat(h << 16u);
}

float q4kAt(uint row_base, uint kidx) {
    const uint BS = 144u;
    uint b = kidx / 256u;
    uint i = kidx % 256u;
    uint base = row_base + b * BS;

    float d = f16ToF32(getU16(pc.weights, base + 0u));
    float dmin = f16ToF32(getU16(pc.weights, base + 2u));
    uint is = i / 64u;
    uint sub = i % 64u;
    uint j = sub % 32u;
    uint hi = sub / 32u;
    uint sc = getByte(pc.weights, base + 4u + is);
    float dl = d * float(int(sc & 0xFu) - 8);
    float ml = dmin * float(int((sc >> 4u) & 0xFu) - 8);
    uint qbyte = getByte(pc.weights, base + 16u + is * 32u + j);
    uint qv = (hi == 0u) ? (qbyte & 0xFu) : (qbyte >> 4u);
    return dl * float(qv) - ml;
}

float q6kAt(uint row_base, uint kidx) {
    const uint BS = 210u;
    uint b = kidx / 256u;
    uint i = kidx % 256u;
    uint base = row_base + b * BS;

    uint hsel = i / 128u;
    uint local = i % 128u;
    uint l = local % 32u;
    uint quarter = local / 32u;
    uint group = l / 16u;
    uint scale_index = hsel * 8u + group + quarter * 2u;
    int sc = int(int(getByte(pc.weights, base + 192u + scale_index) << 24) >> 24);

    uint qh = getByte(pc.weights, base + 128u + hsel * 32u + l);
    uint ql0 = getByte(pc.weights, base + hsel * 64u + l);
    uint ql1 = getByte(pc.weights, base + hsel * 64u + l + 32u);
    int qv = 0;
    if (quarter == 0u) qv = int((ql0 & 0xFu) | (((qh >> 0u) & 0x3u) << 4u)) - 32;
    if (quarter == 1u) qv = int((ql1 & 0xFu) | (((qh >> 2u) & 0x3u) << 4u)) - 32;
    if (quarter == 2u) qv = int(((ql0 >> 4u) & 0xFu) | (((qh >> 4u) & 0x3u) << 4u)) - 32;
    if (quarter == 3u) qv = int(((ql1 >> 4u) & 0xFu) | (((qh >> 6u) & 0x3u) << 4u)) - 32;

    float d = f16ToF32(getU16(pc.weights, base + 208u));
    return d * float(sc) * float(qv);
}

float q40At(uint row_base, uint kidx) {
    uint b = kidx / 32u;
    uint i = kidx % 32u;
    uint base = row_base + b * 18u;
    float d = f16ToF32(getU16(pc.weights, base + 0u));
    uint qbyte = getByte(pc.weights, base + 2u + (i / 2u));
    int qv = int((i & 1u) == 0u ? (qbyte & 0xFu) : (qbyte >> 4u)) - 8;
    return d * float(qv);
}

float q80At(uint row_base, uint kidx) {
    uint b = kidx / 32u;
    uint i = kidx % 32u;
    uint base = row_base + b * 34u;
    float d = f16ToF32(getU16(pc.weights, base + 0u));
    int qv = int(int(getByte(pc.weights, base + 2u + i) << 24u) >> 24u);
    return d * float(qv);
}

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= pc.n_embd) return;

    uint token = pc.indices.data[0];
    uint row_base = token * pc.row_bytes;

    float v = 0.0;
    if (pc.qtype == 2u) {
        v = q40At(row_base, tid);
    } else if (pc.qtype == 8u) {
        v = q80At(row_base, tid);
    } else if (pc.qtype == 12u) {
        v = q4kAt(row_base, tid);
    } else if (pc.qtype == 14u) {
        v = q6kAt(row_base, tid);
    } else {
        uint byte_idx = row_base + tid * 4u;
        uint w = pc.weights.data[byte_idx >> 2];
        uint sh = (byte_idx & 3u) * 8u;
        v = uintBitsToFloat((w >> sh) & 0xFFFFFFFFu);
    }

    float scale = (pc.scale_bits != 0u) ? uintBitsToFloat(pc.scale_bits) : 1.0;
    pc.out_buf.data[tid] = v * scale;
}
