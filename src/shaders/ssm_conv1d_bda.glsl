#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    uint n;            // p1: number of channels (dispatch = n)
    uint channels;     // p2: conv_channels
    uint d_conv;       // p3: kernel size
    uint p4;
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer a;     // conv state (read+write) — layout: (d_conv-1) * channels
    FloatBuffer b;     // new chunk (channels floats, the current input)
    FloatBuffer c;     // conv kernel (d_conv * channels floats)
    FloatBuffer d;     // output (channels floats)
} pc;

layout(local_size_x = 64) in;

// 1D causal convolution for the Gated Delta Net. The state buffer holds
// the previous (d_conv - 1) chunks, channel-major. The kernel is laid out
// as kernel[c * d_conv + k] for channel c, tap k (0 = oldest, d_conv-1 = newest).
//
// Per-channel: out[c] = sum_k kernel[c,k] * x_k
//   where x_k = state[k*channels + c] for k < d_conv-1
//         x_k = new_chunk[c]          for k == d_conv-1
void main() {
    uint c = gl_GlobalInvocationID.x;
    uint channels = pc.channels;
    if (c >= channels) return;

    float acc = 0.0;
    for (uint k = 0u; k < pc.d_conv; ++k) {
        float x_k;
        if (k < pc.d_conv - 1u) {
            x_k = pc.a.data[k * channels + c];
        } else {
            x_k = pc.b.data[c];
        }
        acc += pc.c.data[c * pc.d_conv + k] * x_k;
    }
    pc.d.data[c] = acc;

    // Shift state: oldest entry is at row 0, newest at row d_conv-2.
    // Drop the oldest, append the new chunk at the end.
    if (pc.d_conv >= 2u) {
        for (uint k = 0u; k + 1u < pc.d_conv - 1u; ++k) {
            pc.a.data[k * channels + c] = pc.a.data[(k + 1u) * channels + c];
        }
        pc.a.data[(pc.d_conv - 2u) * channels + c] = pc.b.data[c];
    }
}
