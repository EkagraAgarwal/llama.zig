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
    FloatBuffer a;      // ssm_state (read+write) [n_v_heads, head_k_dim, head_v_dim]
    FloatBuffer b;      // q_in  [n_k_heads, head_k_dim]
    FloatBuffer c;      // k_in  [n_k_heads, head_k_dim]
    FloatBuffer d;      // v_in  [n_v_heads, head_v_dim]
    FloatBuffer e;      // g_in  [n_v_heads] scalar OR [n_v_heads, head_v_dim] (KDA)
    FloatBuffer f;      // beta_in [n_v_heads] per-head scalar
    FloatBuffer g;      // e_out [n_v_heads, head_v_dim] — output
} pc;

layout(local_size_x = 256) in;

// GPU implementation of the Gated Delta Net recurrence for DECODE (N=1).
// Algorithm (per head h):
//   a. S = S * exp(g)             (decay)
//   b. sk[v] = sum_k S[k,v] * k[k]
//   c. d[v] = (v[v] - sk[v]) * beta
//   d. S[k,v] = S[k,v] + k[k] * d[v]
//   e. o[v] = sum_k S[k,v] * q[k] * scale
void main() {
    uint h = gl_WorkGroupID.x;
    if (h >= pc.n_v_heads) return;
    uint v = gl_LocalInvocationID.x;
    if (v >= pc.head_v_dim) return;

    uint H_v = pc.head_v_dim;
    uint H_k = pc.head_k_dim;
    uint n_rep = max(pc.n_v_heads / max(pc.n_k_heads, 1u), 1u);
    uint hk = h / n_rep;
    uint state_off = h * H_k * H_v;
    bool kda = pc.kda_flag != 0u;

    float scale = uintBitsToFloat(pc.scale_bits);
    float g_val = (kda ? pc.e.data[h * H_v + v] : pc.e.data[h]);
    float decay = exp(clamp(g_val, -30.0, 30.0));

    // a. Decay and b. sk[v]
    float sk = 0.0;
    // S is row-major [H_k, H_v], so column v is at indices: state_off + k*H_v + v
    for (uint k = 0u; k < H_k; ++k) {
        float s_val = pc.a.data[state_off + k * H_v + v] * decay;
        pc.a.data[state_off + k * H_v + v] = s_val;
        sk += s_val * pc.c.data[hk * H_k + k];
    }

    // c. d[v]
    float vv = pc.d.data[h * H_v + v];
    float beta = pc.f.data[h];
    float delta_val = (vv - sk) * beta;

    // d. S[k,v] update and e. o[v]
    float ov = 0.0;
    for (uint k = 0u; k < H_k; ++k) {
        float s_val = pc.a.data[state_off + k * H_v + v] + pc.c.data[hk * H_k + k] * delta_val;
        pc.a.data[state_off + k * H_v + v] = s_val;
        ov += s_val * pc.b.data[hk * H_k + k];
    }
    pc.g.data[h * H_v + v] = ov * scale;
}
