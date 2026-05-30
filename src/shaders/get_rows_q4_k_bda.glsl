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
    uint w_base = base / 4u;

    uint d_dmin = pc.weights.data[w_base];
    float d = f16ToF32(d_dmin & 0xFFFFu);
    float min = f16ToF32(d_dmin >> 16u);

    uint s_word_idx = 1u + (i / 128u);
    uint s_word = pc.weights.data[w_base + s_word_idx];
    uint m_word = pc.weights.data[w_base + s_word_idx + 1u];
    
    uint is = (i / 64u);
    uint sc, m;
    if (is == 0u) {
        sc = s_word & 0x3Fu;
        m  = m_word & 0x3Fu;
    } else if (is == 1u) {
        sc = (s_word >> 8u) & 0x3Fu;
        m  = (m_word >> 8u) & 0x3Fu;
    } else if (is == 2u) {
        sc = (pc.weights.data[w_base + 3u] & 0xFu) | ((s_word >> 2u) & 0x30u);
        m  = ((pc.weights.data[w_base + 3u] >> 4u) & 0xFu) | ((m_word >> 2u) & 0x30u);
    } else {
        sc = ((pc.weights.data[w_base + 3u] >> 16u) & 0xFu) | ((s_word >> 18u) & 0x30u);
        m  = ((pc.weights.data[w_base + 3u] >> 20u) & 0xFu) | ((m_word >> 18u) & 0x30u);
    }

    uint q_byte_idx = 16u + (i / 64u) * 32u + (i & 31u);
    uint qb = (pc.weights.data[w_base + q_byte_idx / 4u] >> ((q_byte_idx & 3u) * 8u)) & 0xFFu;

    if ((i & 32u) == 0u) {
        return (d * float(sc)) * float(qb & 0xFu) - (min * float(m));
    } else {
        return (d * float(sc)) * float(qb >> 4u) - (min * float(m));
    }
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