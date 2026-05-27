#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
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
    FloatBuffer kv_cache;
} pc;

layout(local_size_x = 64) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint n = pc.n_kv_heads * pc.head_dim;
    if (i >= n) return;

    uint kv_stride = pc.n_kv_heads * pc.head_dim;
    uint k_off = pc.pos * kv_stride + i;
    uint v_off = pc.max_ctx * kv_stride + pc.pos * kv_stride + i;
    pc.kv_cache.data[k_off] = pc.k_new.data[i];
    pc.kv_cache.data[v_off] = pc.v_new.data[i];
}
