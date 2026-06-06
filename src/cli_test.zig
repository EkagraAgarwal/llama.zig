const std = @import("std");
const cli = @import("cli.zig");

test "CliConfig.parse basic" {
    const allocator = std.testing.allocator;
    const args = [_][]const u8{ "llama.zig", "--model", "test.gguf", "--prompt", "hello", "--max-tokens", "100" };
    
    // Create a mock iterator. Since std.process.Args.Iterator is complex to mock, 
    // we can implement a simple one that has .next().
    const MockArgs = struct {
        args: []const []const u8,
        pos: usize = 0,
        pub fn next(self: *@this()) ?[]const u8 {
            if (self.pos >= self.args.len) return null;
            const res = self.args[self.pos];
            self.pos += 1;
            return res;
        }
    };
    var mock = MockArgs{ .args = &args };
    
    const cfg = try cli.CliConfig.parse(&mock);
    try std.testing.expectEqualStrings("test.gguf", cfg.model_path.?);
    try std.testing.expectEqualStrings("hello", cfg.prompt_text.?);
    try std.testing.expectEqual(@as(u32, 100), cfg.max_tokens);
}

test "CliConfig.parse validation - max_tokens" {
    const args = [_][]const u8{ "llama.zig", "--max-tokens", "0" };
    const MockArgs = struct {
        args: []const []const u8,
        pos: usize = 0,
        pub fn next(self: *@this()) ?[]const u8 {
            if (self.pos >= self.args.len) return null;
            const res = self.args[self.pos];
            self.pos += 1;
            return res;
        }
    };
    var mock = MockArgs{ .args = &args };
    try std.testing.expectError(error.InvalidCommandLineArguments, cli.CliConfig.parse(&mock));
}

test "CliConfig.parse validation - temperature" {
    const args = [_][]const u8{ "llama.zig", "--temperature", "-0.1" };
    const MockArgs = struct {
        args: []const []const u8,
        pos: usize = 0,
        pub fn next(self: *@this()) ?[]const u8 {
            if (self.pos >= self.args.len) return null;
            const res = self.args[self.pos];
            self.pos += 1;
            return res;
        }
    };
    var mock = MockArgs{ .args = &args };
    try std.testing.expectError(error.InvalidCommandLineArguments, cli.CliConfig.parse(&mock));
}
