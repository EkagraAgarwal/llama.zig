RWStructuredBuffer<float> a : register(u0);
RWStructuredBuffer<float> b : register(u1);
RWStructuredBuffer<float> c : register(u2);

[numthreads(64, 1, 1)]
void addmain(uint3 gid : SV_DispatchThreadID) {
    c[gid.x] = a[gid.x] + b[gid.x];
}

[numthreads(64, 1, 1)]
void mulmain(uint3 gid : SV_DispatchThreadID) {
    c[gid.x] = a[gid.x] * b[gid.x];
}