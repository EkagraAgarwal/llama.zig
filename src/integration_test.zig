const std = @import("std");
const gguf = @import("gguf.zig");
const testing = std.testing;

test "mmap load matches file I/O load" {
    const allocator = testing.allocator;
    const model_path = "D:\\llama.zig\\models\\granite-4.0-350m-BF16.gguf";

    // Skip test if model file is not present
    const file_exists = blk: {
        const cwd = std.Io.Dir.cwd();
        const io = std.Io.Threaded.global_single_threaded.io();
        const f = cwd.openFile(io, model_path, .{ .mode = .read_only }) catch {
            break :blk false;
        };
        f.close(io);
        break :blk true;
    };
    if (!file_exists) return;

    // Load via standard file I/O
    var ctx_file = try gguf.loadModel(allocator, model_path);
    defer ctx_file.deinit();

    // Load via mmap
    var ctx_mmap = try gguf.loadModelMmap(allocator, model_path);
    defer ctx_mmap.deinit();

    // Verify metadata matches
    try testing.expectEqual(ctx_file.version, ctx_mmap.version);
    try testing.expectEqual(ctx_file.tensor_count, ctx_mmap.tensor_count);
    try testing.expectEqual(ctx_file.kv_count, ctx_mmap.kv_count);
    try testing.expectEqual(ctx_file.data_offset, ctx_mmap.data_offset);

    // Verify KVs match
    var kv_it = ctx_file.kvs.iterator();
    while (kv_it.next()) |entry| {
        const key = entry.key_ptr.*;
        const val_file = entry.value_ptr.*;
        const val_mmap = ctx_mmap.kvs.get(key) orelse {
            std.debug.print("Key '{s}' missing in mmap load\n", .{key});
            return error.TestExpectedKeyMissing;
        };
        try testing.expectEqual(@as(gguf.ValueType, val_file), @as(gguf.ValueType, val_mmap));
    }

    // Verify tensors match
    var tensor_it = ctx_file.tensors.iterator();
    while (tensor_it.next()) |entry| {
        const name = entry.key_ptr.*;
        const t_file = entry.value_ptr.*;
        const t_mmap = ctx_mmap.tensors.get(name) orelse {
            std.debug.print("Tensor '{s}' missing in mmap load\n", .{name});
            return error.TestExpectedTensorMissing;
        };

        try testing.expectEqual(t_file.type, t_mmap.type);
        try testing.expectEqual(t_file.offset, t_mmap.offset);
        try testing.expectEqual(t_file.n_dims, t_mmap.n_dims);
        try testing.expectEqual(t_file.size(), t_mmap.size());

        // Compare data read from both contexts
        const t_size = t_file.size();
        const compare_len = @min(t_size, 1024);
        if (compare_len > 0) {
            const buf_file = try allocator.alloc(u8, compare_len);
            defer allocator.free(buf_file);
            const buf_mmap = try allocator.alloc(u8, compare_len);
            defer allocator.free(buf_mmap);

            try ctx_file.readTensorData(t_file, buf_file);
            try ctx_mmap.readTensorData(t_mmap, buf_mmap);

            try testing.expectEqualStrings(buf_file, buf_mmap);

            // Also check getTensorSlice directamente
            const mmap_slice = try ctx_mmap.getTensorSlice(t_mmap);
            try testing.expectEqualStrings(buf_file, mmap_slice[0..compare_len]);
        }
    }
}

test "Qwen3.5 4B GGUF parses and has expected architecture + SSM dims" {
    const t = std.testing;
    const allocator = t.allocator;

    const model_path = "D:\\llama.zig\\models\\Qwen3.5-4B-Q4_K_M.gguf";
    if (!std.fs.path.isAbsolute(model_path)) return;
    const file_exists = blk: {
        const cwd = std.Io.Dir.cwd();
        const io = std.Io.Threaded.global_single_threaded.io();
        const f = cwd.openFile(io, model_path, .{ .mode = .read_only }) catch {
            break :blk false;
        };
        f.close(io);
        break :blk true;
    };
    if (!file_exists) return;

    var ctx = try gguf.loadModelMmap(allocator, model_path);
    defer ctx.deinit();

    // Architecture must be qwen35
    const arch_val = ctx.kvs.get("general.architecture") orelse return error.MissingArchitecture;
    try t.expectEqualStrings("qwen35", arch_val.string);

    // Qwen 3.5 SSM parameters must be populated
    const ssm_d_conv = ctx.kvs.get("qwen35.ssm.conv_kernel") orelse return error.MissingSSMConv;
    try t.expectEqual(@as(u32, 4), ssm_d_conv.u32);
    const ssm_d_state = ctx.kvs.get("qwen35.ssm.state_size") orelse return error.MissingSSMState;
    try t.expectEqual(@as(u32, 128), ssm_d_state.u32);
    const ssm_dt_rank = ctx.kvs.get("qwen35.ssm.time_step_rank") orelse return error.MissingSSMDtRank;
    try t.expectEqual(@as(u32, 32), ssm_dt_rank.u32);
    const ssm_n_group = ctx.kvs.get("qwen35.ssm.group_count") orelse return error.MissingSSMGroup;
    try t.expectEqual(@as(u32, 16), ssm_n_group.u32);

    // Dense 4B has no NextN
    if (ctx.kvs.get("qwen35.nextn_predict_layers")) |nextn| {
        try t.expectEqual(@as(u32, 0), nextn.u32);
    }

    // Critical tensor names must be present
    try t.expect(ctx.tensors.contains("blk.0.attn_qkv.weight"));
    try t.expect(ctx.tensors.contains("blk.0.attn_gate.weight"));
    try t.expect(ctx.tensors.contains("blk.0.ssm_conv1d.weight"));
    try t.expect(ctx.tensors.contains("blk.0.ssm_dt.bias"));
    try t.expect(ctx.tensors.contains("blk.0.ffn_down.weight"));
    try t.expect(ctx.tensors.contains("token_embd.weight"));
    try t.expect(ctx.tensors.contains("output_norm.weight"));

    // Architecture-shape sanity checks
    const n_embd_kv = ctx.kvs.get("qwen35.embedding_length") orelse return error.MissingEmbDim;
    try t.expectEqual(@as(u32, 2560), n_embd_kv.u32);
    const n_layer_kv = ctx.kvs.get("qwen35.block_count") orelse return error.MissingBlockCount;
    try t.expectEqual(@as(u32, 32), n_layer_kv.u32);
    const head_dim_kv = ctx.kvs.get("qwen35.attention.key_length") orelse return error.MissingKeyLength;
    try t.expectEqual(@as(u32, 256), head_dim_kv.u32);
}

test "Qwen3.5 9B GGUF parses and has expected architecture + SSM dims" {
    const t = std.testing;
    const allocator = t.allocator;

    const model_path = "D:\\llama.zig\\models\\Qwen3.5-9B-Q4_K_M.gguf";
    const file_exists = blk: {
        const cwd = std.Io.Dir.cwd();
        const io = std.Io.Threaded.global_single_threaded.io();
        const f = cwd.openFile(io, model_path, .{ .mode = .read_only }) catch {
            break :blk false;
        };
        f.close(io);
        break :blk true;
    };
    if (!file_exists) return;

    var ctx = try gguf.loadModelMmap(allocator, model_path);
    defer ctx.deinit();

    const arch_val = ctx.kvs.get("general.architecture") orelse return error.MissingArchitecture;
    try t.expectEqualStrings("qwen35", arch_val.string);

    // 9B has 4096 hidden_dim
    const n_embd_kv = ctx.kvs.get("qwen35.embedding_length") orelse return error.MissingEmbDim;
    try t.expectEqual(@as(u32, 4096), n_embd_kv.u32);
    const n_layer_kv = ctx.kvs.get("qwen35.block_count") orelse return error.MissingBlockCount;
    try t.expectEqual(@as(u32, 32), n_layer_kv.u32);

    // 9B has a larger head_dim (256 vs 160)
    const head_dim_kv = ctx.kvs.get("qwen35.attention.key_length") orelse return error.MissingKeyLength;
    try t.expectEqual(@as(u32, 256), head_dim_kv.u32);

    // SSM parameters consistent with 4B
    const ssm_d_conv = ctx.kvs.get("qwen35.ssm.conv_kernel") orelse return error.MissingSSMConv;
    try t.expectEqual(@as(u32, 4), ssm_d_conv.u32);
    const ssm_d_state = ctx.kvs.get("qwen35.ssm.state_size") orelse return error.MissingSSMState;
    try t.expectEqual(@as(u32, 128), ssm_d_state.u32);
}
