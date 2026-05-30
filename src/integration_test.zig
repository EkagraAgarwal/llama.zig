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
            
            // Also check getTensorSlice directly
            const mmap_slice = try ctx_mmap.getTensorSlice(t_mmap);
            try testing.expectEqualStrings(buf_file, mmap_slice[0..compare_len]);
        }
    }
}
