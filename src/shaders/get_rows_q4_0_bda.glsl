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

float bf16ToF32(uint h) {
    return uintBitsToFloat(h << 16u);
}

int extendSign4(uint v) {
    return int(v) - 8;
}

float q40At(uint row_base, uint kidx) {
    uint b = kidx / 32u;
    uint i = kidx % 32u;
    uint base = row_base + b * 18u;
    float d = f16ToF32(getU16(pc.weights, base + 0u));
    uint byte_idx = i % 16u;
    uint qb = getByte(pc.weights, base + 2u + byte_idx);
    int qv;
    if (i < 16u) {
        qv = extendSign4(qb & 0x0Fu);
    } else {
        qv = extendSign4(qb >> 4);
    }
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
    } else {
        uint byte_idx = row_base + tid * 4u;
        uint w = pc.weights.data[byte_idx >> 2];
        uint sh = (byte_idx & 3u) * 8u;
        v = uintBitsToFloat((w >> sh) & 0xFFFFFFFFu);
    }

    float scale = (pc.scale_bits != 0u) ? uintBitsToFloat(pc.scale_bits) : 1.0;
    pc.out_buf.data[tid] = v * scale;
}
