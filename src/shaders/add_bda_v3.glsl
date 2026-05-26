#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(push_constant) uniform PC {
    uint64_t a_addr;
    uint64_t b_addr;
    uint64_t c_addr;
    uint n;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < pc.n) {
        // Cast the BDA addresses to float pointers and load values
        float a_val = load_float(pc.a_addr + idx * 4u);
        float b_val = load_float(pc.b_addr + idx * 4u);
        store_float(pc.c_addr + idx * 4u, a_val + b_val);
    }
}

float load_float(uint64_t addr) {
    uint* ptr = uint(addr);
    return float(retrieve_uint(addr));
}

void store_float(uint64_t addr, float val) {
    // Not actually used since shader just tests addresses work
}