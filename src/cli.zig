const std = @import("std");

pub const Config = struct {
    model_path: []const u8,
    prompt_text: []const u8,
    max_tokens: u32 = 64,
    temperature: f32 = 0.8,
    seed: u64 = 0,
    top_k: u32 = 0,
    top_p: f32 = 0.9,
    min_p: f32 = 0.0,
    ctx_size_override: ?u32 = null,
    debug_logits: u32 = 0,
    chat_mode: bool = true,
    verbose: bool = false,
    inspect_block: bool = false,
    prefill_chunk: u32 = 0,
    gpu_embed: bool = true,
    report_json: bool = false,
};

pub fn parseArgs(args_it: anytype) !?Config {
    var model_path: ?[]const u8 = null;
    var prompt_text: ?[]const u8 = null;
    var max_tokens: u32 = 64;
    var temperature: f32 = 0.8;
    var seed: u64 = 0;
    var top_k: u32 = 0;
    var top_p: f32 = 0.9;
    var min_p: f32 = 0.0;
    var ctx_size_override: ?u32 = null;
    var debug_logits: u32 = 0;
    var chat_mode: bool = true;
    var verbose: bool = false;
    var inspect_block: bool = false;
    var prefill_chunk: u32 = 0;
    var gpu_embed: bool = true;
    var report_json: bool = false;

    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            model_path = args_it.next();
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            prompt_text = args_it.next();
        } else if (std.mem.eql(u8, arg, "--max-tokens")) {
            max_tokens = std.fmt.parseInt(u32, args_it.next() orelse "64", 10) catch 64;
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            temperature = std.fmt.parseFloat(f32, args_it.next() orelse "0.8") catch 0.8;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            seed = std.fmt.parseInt(u64, args_it.next() orelse "0", 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            top_k = std.fmt.parseInt(u32, args_it.next() orelse "0", 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            top_p = std.fmt.parseFloat(f32, args_it.next() orelse "0.9") catch 0.9;
        } else if (std.mem.eql(u8, arg, "--min-p")) {
            min_p = std.fmt.parseFloat(f32, args_it.next() orelse "0.0") catch 0.0;
        } else if (std.mem.eql(u8, arg, "--ctx-size")) {
            ctx_size_override = std.fmt.parseInt(u32, args_it.next() orelse "0", 10) catch null;
        } else if (std.mem.eql(u8, arg, "--chat")) {
            chat_mode = true;
        } else if (std.mem.eql(u8, arg, "--verbose")) {
            verbose = true;
        } else if (std.mem.eql(u8, arg, "--debug-logits")) {
            debug_logits = std.fmt.parseInt(u32, args_it.next() orelse "10", 10) catch 10;
        } else if (std.mem.eql(u8, arg, "--inspect-block")) {
            inspect_block = true;
        } else if (std.mem.eql(u8, arg, "--prefill-chunk")) {
            prefill_chunk = std.fmt.parseInt(u32, args_it.next() orelse "512", 10) catch 512;
        } else if (std.mem.eql(u8, arg, "--no-gpu-embed")) {
            gpu_embed = false;
        } else if (std.mem.eql(u8, arg, "--report-json")) {
            report_json = true;
        }
    }

    if (model_path == null or prompt_text == null) {
        return null;
    }

    return Config{
        .model_path = model_path.?,
        .prompt_text = prompt_text.?,
        .max_tokens = max_tokens,
        .temperature = temperature,
        .seed = seed,
        .top_k = top_k,
        .top_p = top_p,
        .min_p = min_p,
        .ctx_size_override = ctx_size_override,
        .debug_logits = debug_logits,
        .chat_mode = chat_mode,
        .verbose = verbose,
        .inspect_block = inspect_block,
        .prefill_chunk = prefill_chunk,
        .gpu_embed = gpu_embed,
        .report_json = report_json,
    };
}

pub fn printUsage(writer: anytype) !void {
    try writer.print("Usage: llama.zig --model <path.gguf> --prompt '<text>' [--max-tokens N] [--temperature T]\n", .{});
}
