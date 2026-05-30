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
    uint n_kv_heads;
    uint head_dim;
    uint max_ctx;
    uint pos;
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer k_new;
    FloatBuffer v_new;
    UIntBuffer kv_cache;
} pc;

layout(local_size_x = 64) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint hd_pairs = pc.head_dim / 2u;
    uint n = pc.n_kv_heads * hd_pairs;
    if (i >= n) return;

    uint kv_head = i / hd_pairs;
    uint d_pair = i % hd_pairs;
    uint d = d_pair * 2u;
    uint src_idx = kv_head * pc.head_dim + d;

    uint kv_stride_pairs = pc.n_kv_heads * hd_pairs;
    uint k_off = pc.pos * kv_stride_pairs + i;
    uint v_off = pc.max_ctx * kv_stride_pairs + pc.pos * kv_stride_pairs + i;

    vec2 k_vec = vec2(pc.k_new.data[src_idx], pc.k_new.data[src_idx + 1u]);
    vec2 v_vec = vec2(pc.v_new.data[src_idx], pc.v_new.data[src_idx + 1u]);

    pc.kv_cache.data[k_off] = packHalf2x16(k_vec);
    pc.kv_cache.data[v_off] = packHalf2x16(v_vec);
}
