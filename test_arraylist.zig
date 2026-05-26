const std = @import("std");
pub const GraphNode = struct { x: u32 };
pub fn main() void {
    var list: std.ArrayListUnmanaged(GraphNode) = .{ .items = &[_]GraphNode{}, .capacity = 0 };
    list.append(null, .{ .x = 1 });
    std.debug.print("len={d}\n", .{list.items.len});
}