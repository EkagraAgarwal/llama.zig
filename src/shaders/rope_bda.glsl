#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n_heads;
    uint head_dim;
    uint pos;
    uint padding;
    FloatBuffer a; // input/output
    FloatBuffer unused1;
    FloatBuffer unused2;
} pc;

layout(local_size_x = 64) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.n_heads * (pc.head_dim / 2)) {
        uint head_idx = i / (pc.head_dim / 2);
        uint pair_idx = i % (pc.head_dim / 2);
        
        uint idx0 = head_idx * pc.head_dim + pair_idx;
        uint idx1 = idx0 + pc.head_dim / 2;
        
        float theta = float(pc.pos) * pow(10000.0, -2.0 * float(pair_idx) / float(pc.head_dim));
        float cos_theta = cos(theta);
        float sin_theta = sin(theta);
        
        float v0 = pc.a.data[idx0];
        float v1 = pc.a.data[idx1];
        
        pc.a.data[idx0] = v0 * cos_theta - v1 * sin_theta;
        pc.a.data[idx1] = v0 * sin_theta + v1 * cos_theta;
    }
}
