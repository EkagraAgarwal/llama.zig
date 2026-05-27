#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n_heads;
    uint head_dim_packed;
    uint max_ctx;
    uint pos;
    uint attn_scale_bits;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer q;
    FloatBuffer kv;
    FloatBuffer attn_out;
} pc;

layout(local_size_x = 64) in;

void main() {
    uint head = gl_GlobalInvocationID.x;
    uint hd = pc.head_dim_packed & 0xFFFFu;
    uint n_kv_heads = pc.head_dim_packed >> 16;
    if (head >= pc.n_heads) return;

    uint n_rep = pc.n_heads / n_kv_heads;
    uint kv_head = head / n_rep;
    uint seq_len = pc.pos + 1;
    uint kv_stride = n_kv_heads * hd;
    uint v_base = pc.max_ctx * kv_stride;

    float scale = (pc.attn_scale_bits != 0u) ? uintBitsToFloat(pc.attn_scale_bits) : (1.0 / sqrt(float(hd)));

    float max_s = -1e30;
    for (uint t = 0; t < seq_len; t++) {
        float dot = 0.0;
        for (uint d = 0; d < hd; d++) {
            dot += pc.q.data[head * hd + d] * pc.kv.data[t * kv_stride + kv_head * hd + d];
        }
        max_s = max(max_s, dot * scale);
    }

    float sum_e = 0.0;
    for (uint t = 0; t < seq_len; t++) {
        float dot = 0.0;
        for (uint d = 0; d < hd; d++) {
            dot += pc.q.data[head * hd + d] * pc.kv.data[t * kv_stride + kv_head * hd + d];
        }
        sum_e += exp(dot * scale - max_s);
    }

    for (uint d = 0; d < hd; d++) {
        float acc = 0.0;
        for (uint t = 0; t < seq_len; t++) {
            float dot = 0.0;
            for (uint k = 0; k < hd; k++) {
                dot += pc.q.data[head * hd + k] * pc.kv.data[t * kv_stride + kv_head * hd + k];
            }
            float w = exp(dot * scale - max_s) / sum_e;
            acc += w * pc.kv.data[v_base + t * kv_stride + kv_head * hd + d];
        }
        pc.attn_out.data[head * hd + d] = acc;
    }
}
