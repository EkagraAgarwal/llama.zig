const std = @import("std");
const kernels_data = @import("kernels_data");
pub fn main() void {
    std.debug.print("kernels_data size: {d} bytes\n", .{kernels_data.data.len});
}