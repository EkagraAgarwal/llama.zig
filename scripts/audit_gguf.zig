const std = @import("std");
const gguf = @import("../src/gguf.zig");
const tensor = @import("../src/tensor.zig");

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next();
    const model_path = args.next() orelse {
        std.debug.print("Usage: zig run scripts/audit_gguf.zig -- <model.gguf>\n", .{});
        return;
    };

    var ctx = try gguf.loadModel(allocator, model_path);
    defer ctx.deinit();

    var counts = std.AutoHashMap(tensor.Type, u64).init(allocator);
    defer counts.deinit();

    var it = ctx.tensors.iterator();
    while (it.next()) |entry| {
        const t = entry.value_ptr.*.*;
        const prev = counts.get(t.type) orelse 0;
        try counts.put(t.type, prev + 1);
    }

    std.debug.print("Model: {s}\nTensor count: {}\n\n", .{ model_path, ctx.tensor_count });
    std.debug.print("Type counts:\n", .{});

    var kvs = std.ArrayList(struct { tt: tensor.Type, count: u64 }).empty;
    defer kvs.deinit(allocator);
    var c_it = counts.iterator();
    while (c_it.next()) |entry| {
        try kvs.append(allocator, .{ .tt = entry.key_ptr.*, .count = entry.value_ptr.* });
    }

    std.mem.sort(@TypeOf(kvs.items[0]), kvs.items, {}, struct {
        fn lessThan(_: void, a: @TypeOf(kvs.items[0]), b: @TypeOf(kvs.items[0])) bool {
            return @intFromEnum(a.tt) < @intFromEnum(b.tt);
        }
    }.lessThan);

    for (kvs.items) |item| {
        std.debug.print("  {s:>5} ({:>2})  {:>5}\n", .{
            @tagName(item.tt),
            @intFromEnum(item.tt),
            item.count,
        });
    }

    std.debug.print("\nUnsupported types for current dequant path:\n", .{});
    for (kvs.items) |item| {
        if (!@import("../src/weights.zig").isSupportedType(item.tt)) {
            std.debug.print("  - {s} ({})\n", .{ @tagName(item.tt), @intFromEnum(item.tt) });
        }
    }
}
