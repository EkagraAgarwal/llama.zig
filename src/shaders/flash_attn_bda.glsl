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
    uint tile_size;
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
    uint seq_len = pc.pos + 1u;
    uint kv_stride = n_kv_heads * hd;
    uint v_base = pc.max_ctx * kv_stride;
    uint tile_sz = (pc.tile_size != 0u) ? pc.tile_size : 64u;

    float scale = (pc.attn_scale_bits != 0u) ? uintBitsToFloat(pc.attn_scale_bits) : (1.0 / sqrt(float(hd)));
    if (hd > 256u) return;

    float acc[256];
    for (uint d = 0u; d < hd; ++d) acc[d] = 0.0;

    float m = -1e30;
    float l = 0.0;

    for (uint t0 = 0u; t0 < seq_len; t0 += tile_sz) {
        uint t1 = min(t0 + tile_sz, seq_len);
        for (uint t = t0; t < t1; ++t) {
            float dot = 0.0;
            for (uint d = 0u; d < hd; ++d) {
                dot += pc.q.data[head * hd + d] * pc.kv.data[t * kv_stride + kv_head * hd + d];
            }
            float s = dot * scale;
            float m_new = max(m, s);
            float alpha = exp(m - m_new);
            float beta = exp(s - m_new);
            for (uint d = 0u; d < hd; ++d) {
                float v = pc.kv.data[v_base + t * kv_stride + kv_head * hd + d];
                acc[d] = acc[d] * alpha + beta * v;
            }
            l = l * alpha + beta;
            m = m_new;
        }
    }

    float inv_l = (l > 0.0) ? (1.0 / l) : 0.0;
    for (uint d = 0u; d < hd; ++d) {
        pc.attn_out.data[head * hd + d] = acc[d] * inv_l;
    }
}
