const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const vk_xml_path = b.dependency("vulkan_headers", .{}).path("registry/vk.xml");

    const vulkan_dep = b.dependency("vulkan", .{
        .registry = vk_xml_path,
    });

    const vulkan_mod = vulkan_dep.module("vulkan-zig");

    const kernels_data_mod = b.createModule(.{
        .root_source_file = b.path("kernels_data.zig"),
    });

    const exe = b.addExecutable(.{
        .name = "llama.zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
                .{ .name = "kernels_data", .module = kernels_data_mod },
            },
        }),
    });

    b.installArtifact(exe);

    if (compileShaders(b)) |shader_step| {
        exe.step.dependOn(shader_step);
    }

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const ops_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ops_test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_ops_tests = b.addRunArtifact(ops_tests);
    const root_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
                .{ .name = "kernels_data", .module = kernels_data_mod },
            },
        }),
    });
    const run_root_tests = b.addRunArtifact(root_tests);

    const ssm_state_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ssm_state_test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_ssm_state_tests = b.addRunArtifact(ssm_state_tests);

    const qwen35_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/qwen35_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
                .{ .name = "kernels_data", .module = kernels_data_mod },
            },
        }),
    });
    const run_qwen35_tests = b.addRunArtifact(qwen35_tests);

    const mmap_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mmap_test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_mmap_tests = b.addRunArtifact(mmap_tests);
    const test_mmap_step = b.step("test-mmap", "Run mmap unit tests");
    test_mmap_step.dependOn(&run_mmap_tests.step);

    const integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/integration_test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_integration_tests = b.addRunArtifact(integration_tests);
    const test_integration_step = b.step("test-integration", "Run model loading integration tests");
    test_integration_step.dependOn(&run_integration_tests.step);

const test_step = b.step("test", "Run CPU and parity-focused unit tests");
    test_step.dependOn(&run_ops_tests.step);
    test_step.dependOn(&run_root_tests.step);
    test_step.dependOn(&run_ssm_state_tests.step);
    test_step.dependOn(&run_qwen35_tests.step);
    test_step.dependOn(&run_mmap_tests.step);

    const clean_step = b.step("clean", "Remove all build artifacts and caches");
    const clean_cmd = b.addSystemCommand(&.{
        switch (builtin.os.tag) {
            .windows => "cmd",
            else => "rm",
        },
    });
    if (builtin.os.tag == .windows) {
        clean_cmd.addArgs(&.{"/c", "rmdir /s /q .zig-cache zig-pkg 2>nul & del /q *.spv 2>nul & exit /b 0"});
    } else {
        clean_cmd.addArgs(&.{"-rf", ".zig-cache", "zig-pkg", "*.spv"});
    }
    clean_step.dependOn(&clean_cmd.step);
}

fn compileShaders(b: *std.Build) ?*std.Build.Step {
    const shaders = [_]struct { src: []const u8, out: []const u8 }{
        .{ .src = "src/shaders/add_bda.glsl", .out = "add_bda.spv" },
        .{ .src = "src/shaders/mul_bda.glsl", .out = "mul_bda.spv" },
        .{ .src = "src/shaders/matmul_bda.glsl", .out = "matmul_bda.spv" },
        .{ .src = "src/shaders/matmul_q8_0_bda.glsl", .out = "matmul_q8_0_bda.spv" },
        .{ .src = "src/shaders/matvec_q8_0_bda.glsl", .out = "matvec_q8_0_bda.spv" },
        .{ .src = "src/shaders/matmul_f16_bda.glsl", .out = "matmul_f16_bda.spv" },
        .{ .src = "src/shaders/matvec_f16_bda.glsl", .out = "matvec_f16_bda.spv" },
        .{ .src = "src/shaders/get_rows_q_bda.glsl", .out = "get_rows_q_bda.spv" },
        .{ .src = "src/shaders/matmul_q4_0_bda.glsl", .out = "matmul_q4_0_bda.spv" },
        .{ .src = "src/shaders/matvec_q4_0_bda.glsl", .out = "matvec_q4_0_bda.spv" },
        .{ .src = "src/shaders/get_rows_q4_0_bda.glsl", .out = "get_rows_q4_0_bda.spv" },
        .{ .src = "src/shaders/matvec_q4_1_bda.glsl", .out = "matvec_q4_1_bda.spv" },
        .{ .src = "src/shaders/matmul_q4_1_bda.glsl", .out = "matmul_q4_1_bda.spv" },
        .{ .src = "src/shaders/get_rows_q4_1_bda.glsl", .out = "get_rows_q4_1_bda.spv" },
        .{ .src = "src/shaders/get_rows_q4_k_bda.glsl", .out = "get_rows_q4_k_bda.spv" },
        .{ .src = "src/shaders/get_rows_q5_k_bda.glsl", .out = "get_rows_q5_k_bda.spv" },
        .{ .src = "src/shaders/matvec_q4_k_bda.glsl", .out = "matvec_q4_k_bda.spv" },
        .{ .src = "src/shaders/matvec_q5_k_bda.glsl", .out = "matvec_q5_k_bda.spv" },
        .{ .src = "src/shaders/matmul_q4_k_bda.glsl", .out = "matmul_q4_k_bda.spv" },
        .{ .src = "src/shaders/matmul_q5_k_bda.glsl", .out = "matmul_q5_k_bda.spv" },
        .{ .src = "src/shaders/get_rows_q6_k_bda.glsl", .out = "get_rows_q6_k_bda.spv" },
        .{ .src = "src/shaders/matvec_q6_k_bda.glsl", .out = "matvec_q6_k_bda.spv" },
        .{ .src = "src/shaders/matmul_q6_k_bda.glsl", .out = "matmul_q6_k_bda.spv" },
        .{ .src = "src/shaders/topk_bda.glsl", .out = "topk_bda.spv" },
        .{ .src = "src/shaders/flash_attn_bda.glsl", .out = "flash_attn_bda.spv" },
        .{ .src = "src/shaders/rmsnorm_bda.glsl", .out = "rmsnorm_bda.spv" },
        .{ .src = "src/shaders/softmax_bda.glsl", .out = "softmax_bda.spv" },
        .{ .src = "src/shaders/rope_bda.glsl", .out = "rope_bda.spv" },
        .{ .src = "src/shaders/silu_mul_bda.glsl", .out = "silu_mul_bda.spv" },
        .{ .src = "src/shaders/attention_bda.glsl", .out = "attention_bda.spv" },
        .{ .src = "src/shaders/kv_write_bda.glsl", .out = "kv_write_bda.spv" },
        .{ .src = "src/shaders/scaled_add_bda.glsl", .out = "scaled_add_bda.spv" },
        .{ .src = "src/shaders/gelu_mul_bda.glsl", .out = "gelu_mul_bda.spv" },
        .{ .src = "src/shaders/copy_bda.glsl", .out = "copy_bda.spv" },
        .{ .src = "src/shaders/softplus_bda.glsl", .out = "softplus_bda.spv" },
        .{ .src = "src/shaders/sigmoid_bda.glsl", .out = "sigmoid_bda.spv" },
        .{ .src = "src/shaders/silu_bda.glsl", .out = "silu_bda.spv" },
        .{ .src = "src/shaders/l2_norm_bda.glsl", .out = "l2_norm_bda.spv" },
        .{ .src = "src/shaders/mrope_bda.glsl", .out = "mrope_bda.spv" },
        .{ .src = "src/shaders/attn_gate_mul_bda.glsl", .out = "attn_gate_mul_bda.spv" },
        .{ .src = "src/shaders/ssm_conv1d_bda.glsl", .out = "ssm_conv1d_bda.spv" },
        .{ .src = "src/shaders/ssm_delta_net_decode_bda.glsl", .out = "ssm_delta_net_decode_bda.spv" },
        .{ .src = "src/shaders/ssm_gated_norm_bda.glsl", .out = "ssm_gated_norm_bda.spv" },
    };

    const step = b.step("shaders", "Compile GLSL compute shaders to SPIR-V");
    for (shaders) |s| {
        const cmd = b.addSystemCommand(&.{ "glslangValidator", "-V", "-S", "comp", "--target-env", "vulkan1.2", "-o", s.out, s.src });
        step.dependOn(&cmd.step);
    }
    return step;
}
