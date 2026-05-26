#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n;
    uint d;
    uint p3;
    uint p4;
    FloatBuffer a;
    FloatBuffer b;
    FloatBuffer c;
} pc;

layout(local_size_x = 64) in;

void main() {
    uint row_idx = gl_GlobalInvocationID.x;
    uint num_rows = pc.n / pc.d;
    
    if (row_idx < num_rows) {
        uint row_offset = row_idx * pc.d;
        
        float max_val = -1e30;
        for (uint i = 0; i < pc.d; i++) {
            float val = pc.a.data[row_offset + i];
            if (val > max_val) max_val = val;
        }
        
        float sum_exp = 0.0;
        for (uint j = 0; j < pc.d; j++) {
            float val = pc.a.data[row_offset + j];
            sum_exp += exp(val - max_val);
        }
        
        for (uint k = 0; k < pc.d; k++) {
            float val = pc.a.data[row_offset + k];
            pc.a.data[row_offset + k] = exp(val - max_val) / sum_exp;
        }
    }
}
