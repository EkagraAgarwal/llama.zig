#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n; // count
    uint d; // unused
    uint p3; // src_offset (elements)
    uint p4; // dst_offset (elements)
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer a; // src
    FloatBuffer c; // dst
} pc;

layout(local_size_x = 64) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.n) {
        pc.c.data[i + pc.p4] = pc.a.data[i + pc.p3];
    }
}
