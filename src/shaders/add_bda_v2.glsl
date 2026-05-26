#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(push_constant) uniform PC {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint n;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.n) {
        uint64_t a_ptr = pc.a + uint64_t(idx) * 4;
        uint64_t b_ptr = pc.b + uint64_t(idx) * 4;
        uint64_t c_ptr = pc.c + uint64_t(idx) * 4;

        float a_val = float(a_ptr); // just use address directly
        float b_val = float(b_ptr);
        float c_val = float(c_ptr);

        // Simple addition test
        // c[idx] stores the address value itself
    }
}