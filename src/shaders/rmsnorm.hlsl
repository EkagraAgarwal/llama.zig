struct PushConstants {
    uint n;
    uint d;
    uint64_t a;
    uint64_t b;
    uint64_t c;
};

[[vk::push_constant]] PushConstants pc;

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint row_idx = gid.x;
    uint num_rows = pc.n / pc.d;
    if (row_idx < num_rows) {
        uint row_offset = row_idx * pc.d;
        
        float sum_sq = 0.0f;
        for (uint i = 0; i < pc.d; i++) {
            float val = vk::RawBufferLoad<float>(pc.a + (row_offset + i) * 4);
            sum_sq += val * val;
        }
        
        float mean_sq = sum_sq / (float)pc.d;
        float rms_scale = 1.0f / sqrt(mean_sq + 1e-5f);
        
        for (uint j = 0; j < pc.d; j++) {
            float val = vk::RawBufferLoad<float>(pc.a + (row_offset + j) * 4);
            float weight = vk::RawBufferLoad<float>(pc.b + j * 4);
            vk::RawBufferStore<float>(pc.c + (row_offset + j) * 4, val * rms_scale * weight);
        }
    }
}
