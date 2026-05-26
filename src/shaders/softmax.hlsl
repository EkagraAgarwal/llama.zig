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
        
        float max_val = -1e30f;
        for (uint i = 0; i < pc.d; i++) {
            float val = vk::RawBufferLoad<float>(pc.a + (row_offset + i) * 4);
            if (val > max_val) {
                max_val = val;
            }
        }
        
        float sum_exp = 0.0f;
        for (uint j = 0; j < pc.d; j++) {
            float val = vk::RawBufferLoad<float>(pc.a + (row_offset + j) * 4);
            sum_exp += exp(val - max_val);
        }
        
        for (uint k = 0; k < pc.d; k++) {
            float val = vk::RawBufferLoad<float>(pc.a + (row_offset + k) * 4);
            float prob = exp(val - max_val) / sum_exp;
            vk::RawBufferStore<float>(pc.c + (row_offset + k) * 4, prob);
        }
    }
}
