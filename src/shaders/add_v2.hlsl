RWStructuredBuffer<float> output : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint64_t a_addr = 0;
    uint64_t b_addr = 0;
    uint64_t c_addr = 0;
    uint n = 0;
    
    c[gid.x] = a[gid.x] + b[gid.x];
}