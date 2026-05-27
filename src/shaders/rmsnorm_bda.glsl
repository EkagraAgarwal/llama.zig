#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n;
    uint d;
    uint eps_bits;
    uint p4;
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer a;
    FloatBuffer b;
    FloatBuffer c;
} pc;

layout(local_size_x = 64) in;

void main() {
    uint row_idx = gl_GlobalInvocationID.x;
    uint num_rows = pc.n / pc.d;
    float eps = uintBitsToFloat(pc.eps_bits);

    if (row_idx < num_rows) {
        uint row_offset = row_idx * pc.d;

        float sum_sq = 0.0;
        for (uint i = 0; i < pc.d; i++) {
            float val = pc.a.data[row_offset + i];
            sum_sq += val * val;
        }

        float rms_scale = 1.0 / sqrt(sum_sq / float(pc.d) + eps);

        for (uint j = 0; j < pc.d; j++) {
            float val = pc.a.data[row_offset + j];
            float weight = pc.b.data[j];
            pc.c.data[row_offset + j] = val * rms_scale * weight;
        }
    }
}
