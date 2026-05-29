const std = @import("std");

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

    const test_step = b.step("test", "Run CPU and parity-focused unit tests");
    test_step.dependOn(&run_ops_tests.step);
    test_step.dependOn(&run_root_tests.step);
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
        .{ .src = "src/shaders/get_rows_q6_k_bda.glsl", .out = "get_rows_q6_k_bda.spv" },
        .{ .src = "src/shaders/matvec_q6_k_bda.glsl", .out = "matvec_q6_k_bda.spv" },
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
    };

    const step = b.step("shaders", "Compile GLSL compute shaders to SPIR-V");
    for (shaders) |s| {
        const cmd = b.addSystemCommand(&.{ "glslangValidator", "-V", "-S", "comp", "--target-env", "vulkan1.2", "-o", s.out, s.src });
        step.dependOn(&cmd.step);
    }
    return step;
}
