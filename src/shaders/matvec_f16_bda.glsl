#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer { float data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer UIntBuffer { uint data[]; };

layout(push_constant) uniform PC {
    uint m;
    uint n;
    uint k;
    uint p4;
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer a;
    UIntBuffer b;
    FloatBuffer c;
} pc;

layout(local_size_x = 256) in;

float f16At(uint row, uint kidx) {
    uint element_idx = row * pc.k + kidx;
    uint w = pc.b.data[element_idx >> 1];
    uint val = (element_idx & 1u) == 0u ? (w & 0xFFFFu) : (w >> 16u);
    return unpackHalf2x16(val).x;
}

void main() {
    uint col = gl_GlobalInvocationID.x;
    if (col >= pc.n) return;

    float sum = 0.0;
    uint nblocks = (pc.k + 31u) / 32u;
    for (uint bi = 0u; bi < nblocks; ++bi) {
        uint k0 = bi * 32u;
        for (uint i = 0u; i < 32u; ++i) {
            if (k0 + i < pc.k) {
                sum += pc.a.data[k0 + i] * f16At(col, k0 + i);
            }
        }
    }
    pc.c.data[col] = sum;
}
