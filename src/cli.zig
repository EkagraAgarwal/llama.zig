const std = @import("std");

pub const CliConfig = struct {
    model_path: ?[]const u8 = null,
    prompt_text: ?[]const u8 = null,
    max_tokens: u32 = 64,
    temperature: f32 = 0.8,
    seed: u64 = 0,
    top_k: u32 = 0,
    top_p: f32 = 0.9,
    min_p: f32 = 0.0,
    ctx_size_override: ?u32 = null,
    debug_logits: u32 = 0,
    debug_hidden: bool = false,
    chat_mode: bool = false,
    verbose: bool = false,
    prefill_chunk: u32 = 0,
    gpu_embed: bool = true,
    report_json: bool = false,
    debug_trace: bool = false,
    staging_size: u64 = 128 * 1024 * 1024,

    pub fn parse(args_it: anytype) !CliConfig {
        var config = CliConfig{};
        _ = args_it.next(); // Skip executable name

        while (args_it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--model")) {
                config.model_path = args_it.next();
            } else if (std.mem.eql(u8, arg, "--prompt")) {
                config.prompt_text = args_it.next();
            } else if (std.mem.eql(u8, arg, "--max-tokens")) {
                config.max_tokens = std.fmt.parseInt(u32, args_it.next() orelse "64", 10) catch 64;
            } else if (std.mem.eql(u8, arg, "--temperature")) {
                config.temperature = std.fmt.parseFloat(f32, args_it.next() orelse "0.8") catch 0.8;
            } else if (std.mem.eql(u8, arg, "--seed")) {
                config.seed = std.fmt.parseInt(u64, args_it.next() orelse "0", 10) catch 0;
            } else if (std.mem.eql(u8, arg, "--top-k")) {
                config.top_k = std.fmt.parseInt(u32, args_it.next() orelse "0", 10) catch 0;
            } else if (std.mem.eql(u8, arg, "--top-p")) {
                config.top_p = std.fmt.parseFloat(f32, args_it.next() orelse "0.9") catch 0.9;
            } else if (std.mem.eql(u8, arg, "--min-p")) {
                config.min_p = std.fmt.parseFloat(f32, args_it.next() orelse "0.0") catch 0.0;
            } else if (std.mem.eql(u8, arg, "--ctx-size")) {
                config.ctx_size_override = std.fmt.parseInt(u32, args_it.next() orelse "0", 10) catch null;
            } else if (std.mem.eql(u8, arg, "--chat")) {
                config.chat_mode = true;
            } else if (std.mem.eql(u8, arg, "--no-chat")) {
                config.chat_mode = false;
            } else if (std.mem.eql(u8, arg, "--verbose")) {
                config.verbose = true;
            } else if (std.mem.eql(u8, arg, "--debug-logits")) {
                config.debug_logits = std.fmt.parseInt(u32, args_it.next() orelse "10", 10) catch 10;
            } else if (std.mem.eql(u8, arg, "--debug-hidden")) {
                config.debug_hidden = true;
            } else if (std.mem.eql(u8, arg, "--prefill-chunk")) {
                config.prefill_chunk = std.fmt.parseInt(u32, args_it.next() orelse "512", 10) catch 512;
            } else if (std.mem.eql(u8, arg, "--no-gpu-embed")) {
                config.gpu_embed = false;
            } else if (std.mem.eql(u8, arg, "--report-json")) {
                config.report_json = true;
            } else if (std.mem.eql(u8, arg, "--debug-trace")) {
                config.debug_trace = true;
            } else if (std.mem.eql(u8, arg, "--staging-size")) {
                config.staging_size = std.fmt.parseInt(u64, args_it.next() orelse "134217728", 10) catch (128 * 1024 * 1024);
            }
        }

        if (config.max_tokens < 1) return error.InvalidCommandLineArguments;
        if (config.temperature < 0.0) return error.InvalidCommandLineArguments;
        if (config.top_p < 0.0 or config.top_p > 1.0) return error.InvalidCommandLineArguments;
        if (config.min_p < 0.0 or config.min_p > 1.0) return error.InvalidCommandLineArguments;

        return config;
    }
};
