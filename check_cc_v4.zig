const std = @import("std");

pub fn main() !void {
    inline for (std.meta.fieldNames(std.builtin.CallingConvention)) |name| {
        std.debug.print("{s}\n", .{name});
    }
}
