#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n;
    uint d;
    uint p3;
    uint p4;
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer a;
    FloatBuffer b;
    FloatBuffer c;
} pc;

layout(local_size_x = 256) in;

shared float s_max;
shared float s_sum;
shared float sg_results[8]; // 256 / 32 = 8

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint n = pc.n;

    float local_max = -1e30;
    for (uint i = tid; i < n; i += 256) {
        local_max = max(local_max, pc.a.data[i]);
    }

    float sg_max = subgroupMax(local_max);
    if (gl_SubgroupInvocationID == 0) {
        sg_results[gl_SubgroupID] = sg_max;
    }
    barrier();
    if (tid < 8) {
        float m = sg_results[tid];
        float max_val_sg = subgroupMax(m);
        if (tid == 0) s_max = max_val_sg;
    }
    barrier();
    float max_val = s_max;

    float local_sum = 0.0;
    for (uint i = tid; i < n; i += 256) {
        float e = exp(pc.a.data[i] - max_val);
        pc.c.data[i] = e;
        local_sum += e;
    }

    float sg_sum = subgroupAdd(local_sum);
    if (gl_SubgroupInvocationID == 0) {
        sg_results[gl_SubgroupID] = sg_sum;
    }
    barrier();
    if (tid < 8) {
        float s = sg_results[tid];
        float total_sum = subgroupAdd(s);
        if (tid == 0) s_sum = total_sum;
    }
    barrier();
    float sum_val = s_sum;

    for (uint i = tid; i < n; i += 256) {
        pc.c.data[i] /= sum_val;
    }
}
