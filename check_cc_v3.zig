const std = @import("std");

pub fn main() !void {
    const info = @typeInfo(std.builtin.CallingConvention);
    inline for (info.Enum.fields) |field| {
        std.debug.print("{s}\n", .{field.name});
    }
}
