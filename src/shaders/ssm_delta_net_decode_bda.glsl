#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n_v_heads;     // p1: num_v_heads
    uint head_v_dim;    // p2: S_v
    uint head_k_dim;    // p3: S_k (head_k_dim)
    uint n_k_heads;     // p4: num_k_heads
    uint scale_bits;    // p5: scale (1.0 / sqrt(S_v)) as bitcast u32
    uint kda_flag;      // p6: 0 = scalar g, 1 = per-element g (KDA)
    uint p7;            // reserved
    uint p8;            // reserved
    FloatBuffer a;      // ssm_state (read+write) [head_v_dim, head_v_dim, n_v_heads]
    FloatBuffer b;      // q_in  [head_k_dim, n_k_heads]
    FloatBuffer c;      // k_in  [head_k_dim, n_k_heads]
    FloatBuffer d;      // v_in  [head_v_dim, n_v_heads]
    FloatBuffer e;      // g_in  [1] scalar OR [head_v_dim, n_v_heads] (KDA)
    FloatBuffer f;      // beta_in [n_v_heads] per-head scalar
    FloatBuffer g;      // e_out [head_v_dim, n_v_heads] — output
} pc;

layout(local_size_x = 1) in;

// GPU reference implementation of the Gated Delta Net recurrence for PREFILL.
// Algorithm (per head h):
//   a. S = S * exp(g)                 (decay)
//   b. sk[v] = sum_k S[k,v] * k[k]
//   c. d[v] = (v[v] - sk[v]) * beta
//   d. S[k,v] = S[k,v] + k[k] * d[v]
//   e. o[v] = sum_k S[k,v] * q[k] * scale
//
// The decode-step GDN is computed on CPU (see ssm_state.zig); this shader
// is used during prefill where multiple tokens are processed in one go.
void main() {
    uint h = gl_GlobalInvocationID.x;
    if (h >= pc.n_v_heads) return;

    uint H_v = pc.head_v_dim;
    uint H_k = pc.head_k_dim;
    uint n_rep = max(pc.n_v_heads / max(pc.n_k_heads, 1u), 1u);
    uint hk = h / n_rep;
    uint state_off = h * H_v * H_v;
    bool kda = pc.kda_flag != 0u;

    float scale = uintBitsToFloat(pc.scale_bits);
    float g_val = (kda ? pc.e.data[h * H_v] : pc.e.data[0]);
    float decay = exp(clamp(g_val, -30.0, 30.0));

    // a. Decay
    for (uint v = 0u; v < H_v; ++v) {
        for (uint k = 0u; k < H_v; ++k) {
            pc.a.data[state_off + v * H_v + k] *= decay;
        }
    }

    // b. sk[v] = sum_k S[k,v] * k[k]  (head_k_dim rows of S, padded if H_k < H_v)
    // d. S[k,v] += k[k] * (v[v] - sk[v]) * beta
    // e. o[v] = sum_k S[k,v] * q[k] * scale
    for (uint v = 0u; v < H_v; ++v) {
        float sk = 0.0;
        for (uint kk = 0u; kk < H_k; ++kk) {
            if (kk < H_v) {
                sk += pc.a.data[state_off + kk * H_v + v] * pc.c.data[hk * H_k + kk];
            }
        }
        float vv = pc.d.data[h * H_v + v];
        float beta = pc.f.data[h];
        float delta_val = (vv - sk) * beta;
        // d. outer-product update: S[k,v] += k[k] * delta
        for (uint kk = 0u; kk < H_k; ++kk) {
            if (kk < H_v) {
                pc.a.data[state_off + kk * H_v + v] += pc.c.data[hk * H_k + kk] * delta_val;
            }
        }
        // e. output: o[v] = sum_k S[k,v] * q[k] * scale
        float ov = 0.0;
        for (uint kk = 0u; kk < H_k; ++kk) {
            if (kk < H_v) {
                ov += pc.a.data[state_off + kk * H_v + v] * pc.b.data[hk * H_k + kk];
            }
        }
        pc.g.data[h * H_v + v] = ov * scale;
    }
}
