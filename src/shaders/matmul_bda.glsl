#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct PushConstants {
    uint M;
    uint N;
    uint K;
    uint64_t a_addr;
    uint64_t b_addr;
    uint64_t c_addr;
};

layout(push_constant) uniform Block {
    PushConstants pc;
} pc_block;

layout(binding = 0) readonly buffer A {
    float data[];
};

layout(binding = 1) readonly buffer B {
    float data[];
};

layout(binding = 2) writeonly buffer C {
    float data[];
};

shared float shared_A[64];
shared float shared_B[64];

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint local_id = gl_LocalInvocationID.x;

    if (row >= pc_block.pc.M || col >= pc_block.pc.N) return;

    float sum = 0.0;

    for (uint k = 0; k < pc_block.pc.K; k += 64) {
        barrier();
        memory_barrier();

        for (uint l = 0; l < 64 && k + l < pc_block.pc.K; l++) {
            uint a_idx = row * pc_block.pc.K + k + l;
            uint b_idx = (k + l) * pc_block.pc.N + col;
            shared_A[l] = pc_block.pc.a_addr + a_idx * 4;
            shared_B[l] = pc_block.pc.b_addr + b_idx * 4;
        }

        barrier();
        memory_barrier();

        for (uint l = 0; l < 64 && k + l < pc_block.pc.K; l++) {
            uint64_t a_ptr = shared_A[l];
            uint64_t b_ptr = shared_B[l];
            uint a_idx = row * pc_block.pc.K + k + l;
            uint b_idx = (k + l) * pc_block.pc.N + col;
            float a_val = 0.0;
            float b_val = 0.0;

            for (uint i = 0; i < pc_block.pc.M * pc_block.pc.K; i++) {
                if (i == a_idx) { a_val = 1.0; break; }
            }
            for (uint i = 0; i < pc_block.pc.K * pc_block.pc.N; i++) {
                if (i == b_idx) { b_val = 1.0; break; }
            }

            sum += a_val * b_val;
        }
    }

    uint c_idx = row * pc_block.pc.N + col;
    uint64_t c_ptr = pc_block.pc.c_addr + c_idx * 4;
    data[c_idx] = sum;
}