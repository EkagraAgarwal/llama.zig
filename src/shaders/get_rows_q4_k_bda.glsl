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
    return unpackHalf2x16(h).x;
}

float q4kAt(uint row_base, uint kidx) {
    const uint QK = 256u;
    const uint BS = 144u;
    uint b = kidx / QK;
    uint i = kidx % QK;
    uint base = row_base + b * BS;

    float d = f16ToF32(getU16(pc.weights, base + 0u));
    float min = f16ToF32(getU16(pc.weights, base + 2u));

    uint sbase = base + 4u;
    uint is = (i / 64u) * 2u;
    uint sc0, m0, sc1, m1;
    if (is + 0u < 4u) {
        sc0 = getByte(pc.weights, sbase + is + 0u) & 0x3Fu;
        m0 = getByte(pc.weights, sbase + is + 4u) & 0x3Fu;
    } else {
        sc0 = (getByte(pc.weights, sbase + is + 4u) & 0xFu) | ((getByte(pc.weights, sbase + is - 4u) >> 6u) << 4u);
        m0 = (getByte(pc.weights, sbase + is + 4u) >> 4u) | ((getByte(pc.weights, sbase + is - 0u) >> 6u) << 4u);
    }
    if (is + 1u < 4u) {
        sc1 = getByte(pc.weights, sbase + is + 1u) & 0x3Fu;
        m1 = getByte(pc.weights, sbase + is + 5u) & 0x3Fu;
    } else {
        sc1 = (getByte(pc.weights, sbase + is + 5u) & 0xFu) | ((getByte(pc.weights, sbase + is - 3u) >> 6u) << 4u);
        m1 = (getByte(pc.weights, sbase + is + 5u) >> 4u) | ((getByte(pc.weights, sbase + is + 1u) >> 6u) << 4u);
    }

    float d1 = d * float(sc0);
    float m1v = min * float(m0);
    float d2 = d * float(sc1);
    float m2v = min * float(m1);

    uint qbase = base + 16u + (i / 64u) * 32u + (i & 31u);
    uint qb = getByte(pc.weights, qbase);
    uint q4l = qb & 0xFu;
    uint q4h = qb >> 4u;

    float result;
    if ((i & 32u) == 0u) {
        result = d1 * float(q4l) - m1v;
    } else {
        result = d2 * float(q4h) - m2v;
    }
    return result;
}

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= pc.n_embd) return;

    uint token = pc.indices.data[0];
    uint row_base = token * pc.row_bytes;

    float v = q4kAt(row_base, tid);

    float scale = (pc.scale_bits != 0u) ? uintBitsToFloat(pc.scale_bits) : 1.0;
    pc.out_buf.data[tid] = v * scale;
}