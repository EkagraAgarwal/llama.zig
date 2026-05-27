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
    uint p5;
    uint p6;
    uint p7;
    uint p8;
    FloatBuffer a;
    FloatBuffer b;
    FloatBuffer c;
} pc;

layout(local_size_x = 256) in;

shared float shared_max[256];
shared float shared_sum[256];

void main() {
  uint tid = gl_LocalInvocationID.x;
  uint n = pc.n;

  float local_max = -1e30;
  for (uint i = tid; i < n; i += 256) {
    local_max = max(local_max, pc.a.data[i]);
  }
  shared_max[tid] = local_max;
  barrier();

  for (uint s = 128; s > 0; s >>= 1) {
    if (tid < s) shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
    barrier();
  }
  float max_val = shared_max[0];

  float local_sum = 0.0;
  for (uint i = tid; i < n; i += 256) {
    float e = exp(pc.a.data[i] - max_val);
    pc.c.data[i] = e;
    local_sum += e;
  }
  shared_sum[tid] = local_sum;
  barrier();

  for (uint s = 128; s > 0; s >>= 1) {
    if (tid < s) shared_sum[tid] += shared_sum[tid + s];
    barrier();
  }
  float sum_val = shared_sum[0];

  for (uint i = tid; i < n; i += 256) {
    pc.c.data[i] /= sum_val;
  }
}
