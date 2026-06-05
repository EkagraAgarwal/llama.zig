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
    uint w = buf.data[idx >> 2u];
    return (w >> ((idx & 3u) * 8u)) & 0xFFu;
}

float f16ToF32(uint h) {
    return unpackHalf2x16(h).x;
}

int signExtend5(uint v) {
    if ((v & 0x10u) != 0u) return int(v | 0xFFFFFFE0u);
    return int(v);
}

float q5kAt(uint row_base, uint kidx) {
    const uint QK = 256u;
    const uint BS = 176u;
    uint b = kidx / QK;
    uint i = kidx % QK;
    uint base = row_base + b * BS;
    uint w_base = base / 4u;

    uint d_dmin = pc.weights.data[w_base];
    float d = f16ToF32(d_dmin & 0xFFFFu);
    float dmin = f16ToF32(d_dmin >> 16u);

    uint sw0 = pc.weights.data[w_base + 1u];
    uint sw1 = pc.weights.data[w_base + 2u];
    uint sw2 = pc.weights.data[w_base + 3u];

    uint sb = i / 32u;
    uint is = i % 32u;
    bool is_hi = (i & 16u) != 0u;

    uint sc, mn;
    if (sb == 0u) {
        sc = is_hi ? ((sw0 >> 8u) & 0x3Fu) : (sw0 & 0x3Fu);
        mn = is_hi ? ((sw1 >> 8u) & 0x3Fu) : (sw1 & 0x3Fu);
    } else if (sb == 1u) {
        sc = is_hi ? ((sw0 >> 24u) & 0x3Fu) : ((sw0 >> 16u) & 0x3Fu);
        mn = is_hi ? ((sw1 >> 24u) & 0x3Fu) : ((sw1 >> 16u) & 0x3Fu);
    } else if (sb == 2u) {
        sc = is_hi ? (((sw2 >> 8u) & 0xFu) | ((sw0 >> 10u) & 0x30u)) : ((sw2 & 0xFu) | ((sw0 >> 2u) & 0x30u));
        mn = is_hi ? (((sw2 >> 12u) & 0xFu) | ((sw1 >> 10u) & 0x30u)) : (((sw2 >> 4u) & 0xFu) | ((sw1 >> 2u) & 0x30u));
    } else if (sb == 3u) {
        sc = is_hi ? (((sw2 >> 24u) & 0xFu) | ((sw0 >> 26u) & 0x30u)) : (((sw2 >> 16u) & 0xFu) | ((sw0 >> 18u) & 0x30u));
        mn = is_hi ? (((sw2 >> 28u) & 0xFu) | ((sw1 >> 26u) & 0x30u)) : (((sw2 >> 20u) & 0xFu) | ((sw1 >> 18u) & 0x30u));
    } else if (sb == 4u) {
        // Wait! Q5_K superblock logic for is >= 4... 
        // Actually I'll use the logic from weights.zig
        uint idx = (i / 16u);
        sc = (idx < 4u) ? ((sw0 >> (idx * 8u)) & 0x3Fu) : 
             (idx < 8u) ? (((sw0 >> ((idx-4u)*8u+16u)) & 0x3Fu)) : 
             (idx < 12u) ? (((sw2 >> ((idx-8u)*8u)) & 0xFu) | ((sw0 >> ((idx-8u)*8u+2u)) & 0x30u)) :
             (((sw2 >> ((idx-12u)*8u+16u)) & 0xFu) | ((sw0 >> ((idx-12u)*8u+18u)) & 0x30u));
        mn = (idx < 4u) ? ((sw1 >> (idx * 8u)) & 0x3Fu) : 
             (idx < 8u) ? (((sw1 >> ((idx-4u)*8u+16u)) & 0x3Fu)) : 
             (idx < 12u) ? (((sw2 >> ((idx-8u)*8u+4u)) & 0xFu) | ((sw1 >> ((idx-8u)*8u+2u)) & 0x30u)) :
             (((sw2 >> ((idx-12u)*8u+20u)) & 0xFu) | ((sw1 >> ((idx-12u)*8u+18u)) & 0x30u));
    } else {
        // Reuse the logic above
        uint idx = (i / 16u);
        sc = (idx < 4u) ? ((sw0 >> (idx * 8u)) & 0x3Fu) : 
             (idx < 8u) ? (((sw0 >> ((idx-4u)*8u+16u)) & 0x3Fu)) : 
             (idx < 12u) ? (((sw2 >> ((idx-8u)*8u)) & 0xFu) | ((sw0 >> ((idx-8u)*8u+2u)) & 0x30u)) :
             (((sw2 >> ((idx-12u)*8u+16u)) & 0xFu) | ((sw0 >> ((idx-12u)*8u+18u)) & 0x30u));
        mn = (idx < 4u) ? ((sw1 >> (idx * 8u)) & 0x3Fu) : 
             (idx < 8u) ? (((sw1 >> ((idx-4u)*8u+16u)) & 0x3Fu)) : 
             (idx < 12u) ? (((sw2 >> ((idx-8u)*8u+4u)) & 0xFu) | ((sw1 >> ((idx-8u)*8u+2u)) & 0x30u)) :
             (((sw2 >> ((idx-12u)*8u+20u)) & 0xFu) | ((sw1 >> ((idx-12u)*8u+18u)) & 0x30u));
    }

    uint qh_bit_pos = i / 32u;
    uint qh_byte = getByte(pc.weights, base + 16u + (i % 32u));
    uint qh_bit = (qh_byte >> qh_bit_pos) & 1u;

    uint qs_byte = getByte(pc.weights, base + 48u + i / 2u);
    uint qb = (i & 1u) != 0u ? (qs_byte >> 4u) : (qs_byte & 0xFu);
    uint raw5 = qb | (qh_bit << 4u);

    return d * float(sc) * float(raw5) - dmin * float(mn);
}

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= pc.n_embd) return;

    uint token = pc.indices.data[0];
    uint row_base = token * pc.row_bytes;

    float v = q5kAt(row_base, tid);

    float sc = (pc.scale_bits != 0u) ? uintBitsToFloat(pc.scale_bits) : 1.0;
    pc.out_buf.data[tid] = v * sc;
}
