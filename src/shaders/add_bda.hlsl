struct PushConstants {
    uint n;
    uint padding;
    uint64_t a;
    uint64_t b;
    uint64_t c;
};

[[vk::push_constant]] PushConstants pc;

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x < pc.n) {
        float a_val = vk::RawBufferLoad<float>(pc.a + gid.x * 4);
        float b_val = vk::RawBufferLoad<float>(pc.b + gid.x * 4);
        vk::RawBufferStore<float>(pc.c + gid.x * 4, a_val + b_val);
    }
}
