const std = @import("std");
const builtin = @import("builtin");

pub fn main() !void {
    const info = @typeInfo(builtin.CallingConvention);
    inline for (info.Enum.fields) |field| {
        std.debug.print("{s}\n", .{field.name});
    }
}
// If it's not an enum, try union
// pub fn main() !void {
//     const info = @typeInfo(builtin.CallingConvention);
//     inline for (info.Union.fields) |field| {
//         std.debug.print("{s}\n", .{field.name});
//     }
// }
