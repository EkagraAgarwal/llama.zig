#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n;
    uint padding; // Explicit padding to match Zig's extern struct alignment
    FloatBuffer a;
    FloatBuffer b;
    FloatBuffer c;
} pc;

layout(local_size_x = 64) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.n) {
        pc.c.data[idx] = pc.a.data[idx] + pc.b.data[idx];
    }
}
