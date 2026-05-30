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
    uint n_heads;
    uint head_dim_packed;
    uint max_ctx;
    uint pos;
    uint attn_scale_bits;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer q;
    UIntBuffer kv;
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
    
    uint hd_pairs = hd / 2u;
    uint kv_stride_pairs = n_kv_heads * hd_pairs;
    uint v_base_pairs = pc.max_ctx * kv_stride_pairs;

    float scale = (pc.attn_scale_bits != 0u) ? uintBitsToFloat(pc.attn_scale_bits) : (1.0 / sqrt(float(hd)));

    const uint MAX_HD = 256u;
    if (hd > MAX_HD) return;

    float acc[MAX_HD];
    for (uint d = 0u; d < hd; ++d) acc[d] = 0.0;

    float m = -1e30;
    float l = 0.0;
    for (uint t = 0u; t < seq_len; ++t) {
        float dot = 0.0;
        for (uint d = 0u; d < hd_pairs; ++d) {
            vec2 k_vec = unpackHalf2x16(pc.kv.data[t * kv_stride_pairs + kv_head * hd_pairs + d]);
            dot += pc.q.data[head * hd + d * 2u] * k_vec.x;
            dot += pc.q.data[head * hd + d * 2u + 1u] * k_vec.y;
        }
        float s = dot * scale;
        float m_new = max(m, s);
        float alpha = exp(m - m_new);
        float beta = exp(s - m_new);
        for (uint d = 0u; d < hd_pairs; ++d) {
            vec2 v_vec = unpackHalf2x16(pc.kv.data[v_base_pairs + t * kv_stride_pairs + kv_head * hd_pairs + d]);
            acc[d * 2u] = acc[d * 2u] * alpha + beta * v_vec.x;
            acc[d * 2u + 1u] = acc[d * 2u + 1u] * alpha + beta * v_vec.y;
        }
        l = l * alpha + beta;
        m = m_new;
    }

    float inv_l = (l > 0.0) ? (1.0 / l) : 0.0;
    for (uint d = 0u; d < hd; ++d) {
        pc.attn_out.data[head * hd + d] = acc[d] * inv_l;
    }
}
