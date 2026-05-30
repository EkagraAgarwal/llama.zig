const std = @import("std");
const mmap = @import("mmap.zig");
const testing = std.testing;

test "mmap opens and reads a known file" {
    const cwd = std.Io.Dir.cwd();
    const io = std.Io.Threaded.global_single_threaded.io();
    const test_filename = "mmap_test_temp_file.txt";

    // Create file
    const file = try cwd.createFile(io, test_filename, .{});
    defer {
        cwd.deleteFile(io, test_filename) catch {};
    }
    const expected_content = "Hello, memory mapped files in Zig! This is a test string.";
    try file.writePositionalAll(io, expected_content, 0);
    file.close(io);

    // Map it
    var mapped = try mmap.MappedFile.init(test_filename);
    defer mapped.deinit();

    // Verify content
    try testing.expectEqual(expected_content.len, mapped.data.len);
    try testing.expectEqualStrings(expected_content, mapped.data);
}

test "mmap slice returns correct subrange" {
    const cwd = std.Io.Dir.cwd();
    const io = std.Io.Threaded.global_single_threaded.io();
    const test_filename = "mmap_test_temp_slice.txt";
    const file = try cwd.createFile(io, test_filename, .{});
    defer {
        cwd.deleteFile(io, test_filename) catch {};
    }
    const expected_content = "0123456789abcdef";
    try file.writePositionalAll(io, expected_content, 0);
    file.close(io);

    var mapped = try mmap.MappedFile.init(test_filename);
    defer mapped.deinit();

    const sub = mapped.slice(4, 6);
    try testing.expectEqualStrings("456789", sub);
}

test "mmap empty file returns error" {
    const cwd = std.Io.Dir.cwd();
    const io = std.Io.Threaded.global_single_threaded.io();
    const test_filename = "mmap_test_temp_empty.txt";
    const file = try cwd.createFile(io, test_filename, .{});
    defer {
        cwd.deleteFile(io, test_filename) catch {};
    }
    file.close(io);

    const result = mmap.MappedFile.init(test_filename);
    try testing.expectError(error.EmptyFile, result);
}

test "mmap non-existent file returns error" {
    const result = mmap.MappedFile.init("non_existent_file_xyz_123.txt");
    try testing.expectError(error.FileNotFound, result);
}
