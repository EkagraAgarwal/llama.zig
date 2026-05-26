const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Vulkan code generation
    const vk_xml_path = b.dependency("vulkan_headers", .{}).path("registry/vk.xml");

    const vulkan_dep = b.dependency("vulkan", .{
        .registry = vk_xml_path,
    });

    const vulkan_mod = vulkan_dep.module("vulkan-zig");

    // SPIR-V Kernel Compilation via System Command
    // We use raw spirv64-vulkan which seems to produce more compatible output
    const kernels_compile = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "build-obj",
        b.pathFromRoot("src/shaders/kernels.zig"),
        "-target", "spirv64-vulkan",
        "-ofmt=spirv",
    });
    const kernels_spv = kernels_compile.addPrefixedOutputFileArg("-femit-bin=", "kernels.spv");

    const wf = b.addWriteFile("kernels_data.zig", "pub const data = @embedFile(\"kernels.spv\");\n");
    _ = wf.addCopyFile(kernels_spv, "kernels.spv");
    
    const kernels_data_mod = b.createModule(.{
        .root_source_file = wf.getDirectory().path(b, "kernels_data.zig"),
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

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_tests = b.addTest(.{
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

    const run_exe_tests = b.addRunArtifact(exe_tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_exe_tests.step);
}
