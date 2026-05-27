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
    uint vocab_size;
    uint top_k;
    uint logit_scale_bits;
    uint p4;
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer logits;
    UIntBuffer out_indices;
    FloatBuffer out_values;
} pc;

layout(local_size_x = 256) in;

shared float s_val[256];
shared uint s_idx[256];

void main() {
    uint lid = gl_LocalInvocationID.x;

    float local_best = -1e30;
    uint local_idx = 0u;
    for (uint i = lid; i < pc.vocab_size; i += 256u) {
        float v = pc.logits.data[i];
        if (pc.logit_scale_bits != 0u) {
            v *= 1.0 / uintBitsToFloat(pc.logit_scale_bits);
        }
        if (v > local_best) {
            local_best = v;
            local_idx = i;
        }
    }

    s_val[lid] = local_best;
    s_idx[lid] = local_idx;
    barrier();

    for (uint off = 128u; off > 0u; off >>= 1u) {
        if (lid < off) {
            if (s_val[lid + off] > s_val[lid]) {
                s_val[lid] = s_val[lid + off];
                s_idx[lid] = s_idx[lid + off];
            }
        }
        barrier();
    }

    if (lid == 0u) {
        pc.out_indices.data[0] = s_idx[0];
        pc.out_values.data[0] = s_val[0];
    }
}
