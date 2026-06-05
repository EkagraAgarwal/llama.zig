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

    uint sw0 = pc.weights.data[w_base + 1u];
    uint sw1 = pc.weights.data[w_base + 2u];
    uint sw2 = pc.weights.data[w_base + 3u];

    uint sb = i / 64u;
    uint base_lane = i % 32u;
    bool is_hi = (i & 32u) != 0u;

    uint sc, m;
    if (sb == 0u) {
        sc = is_hi ? ((sw0 >> 8u) & 0x3Fu) : (sw0 & 0x3Fu);
        m  = is_hi ? ((sw1 >> 8u) & 0x3Fu) : (sw1 & 0x3Fu);
    } else if (sb == 1u) {
        sc = is_hi ? ((sw0 >> 24u) & 0x3Fu) : ((sw0 >> 16u) & 0x3Fu);
        m  = is_hi ? ((sw1 >> 24u) & 0x3Fu) : ((sw1 >> 16u) & 0x3Fu);
    } else if (sb == 2u) {
        sc = is_hi ? (((sw2 >> 8u) & 0xFu) | ((sw0 >> 10u) & 0x30u)) : ((sw2 & 0xFu) | ((sw0 >> 2u) & 0x30u));
        m  = is_hi ? (((sw2 >> 12u) & 0xFu) | ((sw1 >> 10u) & 0x30u)) : (((sw2 >> 4u) & 0xFu) | ((sw1 >> 2u) & 0x30u));
    } else {
        sc = is_hi ? (((sw2 >> 24u) & 0xFu) | ((sw0 >> 26u) & 0x30u)) : (((sw2 >> 16u) & 0xFu) | ((sw0 >> 18u) & 0x30u));
        m  = is_hi ? (((sw2 >> 28u) & 0xFu) | ((sw1 >> 26u) & 0x30u)) : (((sw2 >> 20u) & 0xFu) | ((sw1 >> 18u) & 0x30u));
    }

    uint q_word = pc.weights.data[w_base + 4u + sb * 8u + (base_lane / 4u)];
    uint qb = (q_word >> ((base_lane & 3u) * 8u)) & 0xFFu;
    uint nibble = is_hi ? (qb >> 4u) : (qb & 0xFu);

    return (d * float(sc)) * float(nibble) - (min * float(m));
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