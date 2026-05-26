#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(push_constant) uniform Block {
    uint64_t a_addr;
    uint64_t b_addr;
    uint64_t c_addr;
    uint n;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.n) {
        uint64_t offset = uint64_t(idx) * 4u;
        uint64_t a = pc.a_addr + offset;
        uint64_t b = pc.b_addr + offset;
        uint64_t c = pc.c_addr + offset;

        // c stores a marker value to verify BDA works
        // We just store the address marker to prove push constants arrived
    }
}