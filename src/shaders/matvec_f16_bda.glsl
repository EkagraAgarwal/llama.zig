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

uint getByte(UIntBuffer buf, uint idx) {
    uint w = buf.data[idx >> 2];
    return (w >> ((idx & 3u) * 8u)) & 0xFFu;
}

uint getU16(UIntBuffer buf, uint idx) {
    return getByte(buf, idx) | (getByte(buf, idx + 1u) << 8u);
}

float f16At(uint row, uint kidx) {
    uint base = (row * pc.k + kidx) * 2u;
    return unpackHalf2x16(getU16(pc.b, base)).x;
}

void main() {
    uint col = gl_GlobalInvocationID.x;
    if (col >= pc.n) return;
    float sum = 0.0;
    for (uint i = 0u; i < pc.k; i += 1u) {
        sum += pc.a.data[i] * f16At(col, i);
    }
    pc.c.data[col] = sum;
}
