const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Type = @import("tensor.zig").Type;

pub const GGUFMagic = 0x46554747; // "GGUF" in little-endian

pub const ValueType = enum(u32) {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,
};

pub const MetadataValue = union(ValueType) {
    u8: u8,
    i8: i8,
    u16: u16,
    i16: i16,
    u32: u32,
    i32: i32,
    f32: f32,
    bool: bool,
    string: []const u8,
    array: []MetadataValue,
    u64: u64,
    i64: i64,
    f64: f64,
};

pub const KV = struct {
    key: []const u8,
    value: MetadataValue,
};

pub const GGUFContext = struct {
    allocator: std.mem.Allocator,
    file: std.Io.File,
    version: u32,
    tensor_count: u64,
    kv_count: u64,
    kvs: std.StringHashMap(MetadataValue),
    tensors: std.StringHashMap(*Tensor),
    data_offset: u64,
    
    pub fn init(allocator: std.mem.Allocator, file: std.Io.File) GGUFContext {
        return GGUFContext{
            .allocator = allocator,
            .file = file,
            .version = 0,
            .tensor_count = 0,
            .kv_count = 0,
            .kvs = std.StringHashMap(MetadataValue).init(allocator),
            .tensors = std.StringHashMap(*Tensor).init(allocator),
            .data_offset = 0,
        };
    }

    pub fn deinit(self: *GGUFContext) void {
        // Free keys and values
        var it = self.kvs.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            freeMetadataValue(self.allocator, entry.value_ptr.*);
        }
        self.kvs.deinit();

        // Free tensors
        var tensor_it = self.tensors.iterator();
        while (tensor_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit(self.allocator);
        }
        self.tensors.deinit();

        const io = std.Io.Threaded.global_single_threaded.io();
        self.file.close(io);
    }
};

fn freeMetadataValue(allocator: std.mem.Allocator, value: MetadataValue) void {
    switch (value) {
        .string => |s| allocator.free(s),
        .array => |arr| {
            for (arr) |item| {
                freeMetadataValue(allocator, item);
            }
            allocator.free(arr);
        },
        else => {},
    }
}

pub fn loadModel(allocator: std.mem.Allocator, path: []const u8) !GGUFContext {
    // Open file using std.Io
    const cwd = std.Io.Dir.cwd();
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try cwd.openFile(io, path, .{ .mode = .read_only });
    
    var ctx = GGUFContext.init(allocator, file);
    errdefer ctx.deinit();

    var read_buffer: [8192]u8 = undefined;
    var file_reader = file.readerStreaming(io, &read_buffer);
    const reader = &file_reader.interface;
    
    // Read Magic
    const magic = try readInt(u32, reader);
    if (magic != GGUFMagic) {
        return error.InvalidMagic;
    }

    // Read Version
    ctx.version = try readInt(u32, reader);
    
    // Read counts
    ctx.tensor_count = try readInt(u64, reader);
    ctx.kv_count = try readInt(u64, reader);

    // Parse KV pairs
    for (0..ctx.kv_count) |_| {
        const key = try readString(allocator, reader);
        errdefer allocator.free(key);
        
        const val_type_int = try readInt(u32, reader);
        const val_type = @as(ValueType, @enumFromInt(val_type_int));
        
        const val = try readMetadataValue(allocator, reader, val_type);
        try ctx.kvs.put(key, val);
    }

    // Parse Tensor Info
    for (0..ctx.tensor_count) |_| {
        const name = try readString(allocator, reader);
        errdefer allocator.free(name);

        const n_dims = try readInt(u32, reader);
        var dims = try allocator.alloc(u64, n_dims);
        defer allocator.free(dims);

        for (0..n_dims) |i| {
            dims[i] = try readInt(u64, reader);
        }

        const tensor_type_int = try readInt(u32, reader);
        const tensor_type = @as(Type, @enumFromInt(tensor_type_int));
        const offset = try readInt(u64, reader);

        var tensor = try Tensor.init(allocator, name, tensor_type, dims);
        tensor.offset = offset;
        
        try ctx.tensors.put(name, tensor);
    }

    // The data offset requires aligning to a multiple
    const alignment = getAlignment(ctx.kvs);
    const current_pos = file_reader.logicalPos();
    
    const remainder = current_pos % alignment;
    if (remainder != 0) {
        ctx.data_offset = current_pos + (alignment - remainder);
    } else {
        ctx.data_offset = current_pos;
    }

    return ctx;
}

fn getAlignment(kvs: std.StringHashMap(MetadataValue)) u64 {
    if (kvs.get("general.alignment")) |val| {
        return switch (val) {
            .u8 => |v| @as(u64, v),
            .u16 => |v| @as(u64, v),
            .u32 => |v| @as(u64, v),
            .u64 => |v| v,
            .i8 => |v| @as(u64, @intCast(v)),
            .i16 => |v| @as(u64, @intCast(v)),
            .i32 => |v| @as(u64, @intCast(v)),
            .i64 => |v| @as(u64, @intCast(v)),
            else => 32,
        };
    }
    return 32; // default GGUF alignment
}

fn readString(allocator: std.mem.Allocator, reader: *std.Io.Reader) ![]const u8 {
    const len = try readInt(u64, reader);
    const buf = try allocator.alloc(u8, len);
    errdefer allocator.free(buf);
    try reader.readSliceAll(buf);
    return buf;
}

fn readInt(comptime T: type, reader: *std.Io.Reader) !T {
    var bytes: [@sizeOf(T)]u8 = undefined;
    try reader.readSliceAll(&bytes);
    return std.mem.readInt(T, &bytes, .little);
}

fn readMetadataValue(allocator: std.mem.Allocator, reader: *std.Io.Reader, val_type: ValueType) anyerror!MetadataValue {
    return switch (val_type) {
        .u8 => .{ .u8 = (try readInt(u8, reader)) },
        .i8 => .{ .i8 = @bitCast(try readInt(u8, reader)) },
        .u16 => .{ .u16 = try readInt(u16, reader) },
        .i16 => .{ .i16 = @bitCast(try readInt(i16, reader)) },
        .u32 => .{ .u32 = try readInt(u32, reader) },
        .i32 => .{ .i32 = @bitCast(try readInt(i32, reader)) },
        .f32 => .{ .f32 = @bitCast(try readInt(u32, reader)) },
        .f64 => .{ .f64 = @bitCast(try readInt(u64, reader)) },
        .u64 => .{ .u64 = try readInt(u64, reader) },
        .i64 => .{ .i64 = @bitCast(try readInt(i64, reader)) },
        .bool => .{ .bool = (try readInt(u8, reader)) != 0 },
        .string => .{ .string = try readString(allocator, reader) },
        .array => {
            const arr_val_type_int = try readInt(u32, reader);
            const arr_val_type = @as(ValueType, @enumFromInt(arr_val_type_int));
            const arr_len = try readInt(u64, reader);
            
            var arr = try allocator.alloc(MetadataValue, arr_len);
            errdefer allocator.free(arr);
            
            for (0..arr_len) |i| {
                arr[i] = try readMetadataValue(allocator, reader, arr_val_type);
            }
            return .{ .array = arr };
        },
    };
}
