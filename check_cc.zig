const std = @import("std");
const builtin = @import("builtin");

pub fn main() void {
    const info = @typeInfo(builtin.CallingConvention);
    inline for (info.union.fields) |field| {
        std.debug.print("{s}\n", .{field.name});
    }
}
