#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n_heads;
    uint head_v_dim;
    uint head_k_dim;
    uint num_k_heads;
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    uint p9;
    uint p10;
    FloatBuffer state;
    FloatBuffer q_in;
    FloatBuffer k_in;
    FloatBuffer v_in;
    FloatBuffer e;       // [head_v_dim, num_v_heads] — output
} pc;

layout(local_size_x = 1) in;

// ssm_delta_net_prefill: chunked Gated Delta Net for the prefill path.
// For the initial port (token-by-token prefill), this shader is unused.
// It is provided so the OpType and pipeline name are resolvable; the
// current implementation is a placeholder that emits v_in unchanged.
void main() {
    uint h = gl_GlobalInvocationID.x;
    if (h >= pc.n_heads) return;
    uint H_v = pc.head_v_dim;
    for (uint v = 0u; v < H_v; ++v) {
        pc.e.data[h * H_v + v] = pc.v_in.data[h * H_v + v];
    }
}
