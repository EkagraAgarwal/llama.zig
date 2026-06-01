const kernels_data = @import("kernels_data");
const vulkan = @import("vulkan_backend.zig");

pub fn registerDefaultKernels(registry: *vulkan.PipelineRegistry, vk_ctx: *vulkan.Context) !void {
    try registry.register(vk_ctx, "add", kernels_data.kernels_add_spv, "main");
    try registry.register(vk_ctx, "mul", kernels_data.kernels_mul_spv, "main");
    try registry.register(vk_ctx, "rms_norm", kernels_data.kernels_rmsnorm_spv, "main");
    try registry.register(vk_ctx, "softmax", kernels_data.kernels_softmax_spv, "main");
    try registry.register(vk_ctx, "matmul", kernels_data.kernels_matmul_spv, "main");
    try registry.register(vk_ctx, "rope", kernels_data.kernels_rope_spv, "main");
    try registry.register(vk_ctx, "silu_mul", kernels_data.kernels_silu_mul_spv, "main");
    try registry.register(vk_ctx, "attention", kernels_data.kernels_attention_spv, "main");
    try registry.register(vk_ctx, "kv_write", kernels_data.kernels_kv_write_spv, "main");
    try registry.register(vk_ctx, "scaled_add", kernels_data.kernels_scaled_add_spv, "main");
    try registry.register(vk_ctx, "matmul_q8_0", kernels_data.kernels_matmul_q8_0_spv, "main");
    try registry.register(vk_ctx, "matvec_q8_0", kernels_data.kernels_matvec_q8_0_spv, "main");
    try registry.register(vk_ctx, "matmul_f16", kernels_data.kernels_matmul_f16_spv, "main");
    try registry.register(vk_ctx, "matvec_f16", kernels_data.kernels_matvec_f16_spv, "main");
    try registry.register(vk_ctx, "get_rows_q", kernels_data.kernels_get_rows_q_spv, "main");
    try registry.register(vk_ctx, "matmul_q4_0", kernels_data.kernels_matmul_q4_0_spv, "main");
    try registry.register(vk_ctx, "matvec_q4_0", kernels_data.kernels_matvec_q4_0_spv, "main");
    try registry.register(vk_ctx, "get_rows_q4_0", kernels_data.kernels_get_rows_q4_0_spv, "main");
    try registry.register(vk_ctx, "matvec_q4_1", kernels_data.kernels_matvec_q4_1_spv, "main");
    try registry.register(vk_ctx, "matmul_q4_1", kernels_data.kernels_matmul_q4_1_spv, "main");
    try registry.register(vk_ctx, "get_rows_q4_1", kernels_data.kernels_get_rows_q4_1_spv, "main");
    try registry.register(vk_ctx, "get_rows_q4_k", kernels_data.kernels_get_rows_q4_k_spv, "main");
    try registry.register(vk_ctx, "matvec_q4_k", kernels_data.kernels_matvec_q4_k_spv, "main");
    try registry.register(vk_ctx, "matmul_q4_k", kernels_data.kernels_matmul_q4_k_spv, "main");
    try registry.register(vk_ctx, "get_rows_q6_k", kernels_data.kernels_get_rows_q6_k_spv, "main");
    try registry.register(vk_ctx, "matvec_q6_k", kernels_data.kernels_matvec_q6_k_spv, "main");
    try registry.register(vk_ctx, "matmul_q6_k", kernels_data.kernels_matmul_q6_k_spv, "main");
    try registry.register(vk_ctx, "topk", kernels_data.kernels_topk_spv, "main");
    try registry.register(vk_ctx, "attention_flash", kernels_data.kernels_flash_attn_spv, "main");
    try registry.register(vk_ctx, "gelu_mul", kernels_data.kernels_gelu_mul_spv, "main");
    try registry.register(vk_ctx, "copy", kernels_data.kernels_copy_spv, "main");
}
