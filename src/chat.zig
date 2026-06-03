const std = @import("std");
const tokenizer = @import("tokenizer.zig");
const model = @import("model.zig");

pub const ChatFormat = enum {
    chatml,
    gemma,
    llama3,
    llama2,
    granite,
    unknown,
};

pub fn detectChatFormat(
    chat_template: ?[]const u8,
    arch: model.Architecture,
    special: *const tokenizer.SpecialTokens,
) ChatFormat {
    _ = arch;
    if (chat_template) |tpl| {
        if (std.mem.indexOf(u8, tpl, "<|im_start|>") != null) return .chatml;
        if (std.mem.indexOf(u8, tpl, "<start_of_turn>") != null) return .gemma;
        if (std.mem.indexOf(u8, tpl, "<|start_header_id|>") != null) return .llama3;
        if (std.mem.indexOf(u8, tpl, "[INST]") != null) return .llama2;
        if (std.mem.indexOf(u8, tpl, "<|start_of_role|>") != null) return .granite;
    }
    if (special.im_start != null) return .chatml;
    if (special.start_of_turn != null) return .gemma;
    if (special.inst_start != null) return .llama2;
    if (special.start_of_role != null) return .granite;
    return .unknown;
}

pub fn buildChatPrompt(
    tok: *const tokenizer.Tokenizer,
    format: ChatFormat,
    user_text: []const u8,
    allocator: std.mem.Allocator,
) ![]tokenizer.TokenID {
    var tokens: std.ArrayList(tokenizer.TokenID) = .empty;
    errdefer tokens.deinit(allocator);

    const addToken = struct {
        fn func(t: *std.ArrayList(tokenizer.TokenID), id: tokenizer.TokenID, a: std.mem.Allocator) !void {
            try t.append(a, id);
        }
    }.func;

    const addText = struct {
        fn func(tok_enc: *const tokenizer.Tokenizer, t: *std.ArrayList(tokenizer.TokenID), text: []const u8, a: std.mem.Allocator) !void {
            const encoded = try tok_enc.encodeEx(text, false, a);
            defer a.free(encoded);
            try t.appendSlice(a, encoded);
        }
    }.func;

    switch (format) {
        .chatml => {
            if (tok.bos_token_id != null and tok.add_bos_token) {
                try addToken(&tokens, tok.bos_token_id.?, allocator);
            }
            if (tok.special.im_start) |id| try addToken(&tokens, id, allocator);
            try addText(tok, &tokens, "user\n", allocator);
            try addText(tok, &tokens, user_text, allocator);
            if (tok.special.im_end) |id| try addToken(&tokens, id, allocator);
            try addText(tok, &tokens, "\n", allocator);
            if (tok.special.im_start) |id| try addToken(&tokens, id, allocator);
            try addText(tok, &tokens, "assistant\n", allocator);
        },
        .gemma => {
            if (tok.special.start_of_turn == null or tok.special.end_of_turn == null) {
                return try tok.encode(user_text, allocator);
            }
            if (tok.bos_token_id != null and tok.add_bos_token) {
                try addToken(&tokens, tok.bos_token_id.?, allocator);
            }
            if (tok.special.start_of_turn) |id| try addToken(&tokens, id, allocator);
            try addText(tok, &tokens, "user\n", allocator);
            try addText(tok, &tokens, user_text, allocator);
            if (tok.special.end_of_turn) |id| try addToken(&tokens, id, allocator);
            try addText(tok, &tokens, "\n", allocator);
            if (tok.special.start_of_turn) |id| try addToken(&tokens, id, allocator);
            try addText(tok, &tokens, "model\n", allocator);
        },
        .llama3 => {
            if (tok.special.begin_of_text == null or tok.special.start_header_id == null or
                tok.special.end_header_id == null or tok.special.eot_id == null) {
                return try tok.encode(user_text, allocator);
            }
            try addToken(&tokens, tok.special.begin_of_text.?, allocator);
            try addToken(&tokens, tok.special.start_header_id.?, allocator);
            try addText(tok, &tokens, "user", allocator);
            try addToken(&tokens, tok.special.end_header_id.?, allocator);
            try addText(tok, &tokens, "\n\n", allocator);
            try addText(tok, &tokens, user_text, allocator);
            try addToken(&tokens, tok.special.eot_id.?, allocator);
            try addToken(&tokens, tok.special.start_header_id.?, allocator);
            try addText(tok, &tokens, "assistant", allocator);
            try addToken(&tokens, tok.special.end_header_id.?, allocator);
            try addText(tok, &tokens, "\n\n", allocator);
        },
        .llama2 => {
            if (tok.special.inst_start == null or tok.special.inst_end == null) {
                return try tok.encode(user_text, allocator);
            }
            if (tok.bos_token_id != null and tok.add_bos_token) {
                try addToken(&tokens, tok.bos_token_id.?, allocator);
            }
            if (tok.special.inst_start) |id| try addToken(&tokens, id, allocator);
            try addText(tok, &tokens, user_text, allocator);
            if (tok.special.inst_end) |id| try addToken(&tokens, id, allocator);
            try addText(tok, &tokens, "\n", allocator);
        },
        .granite => {
            if (tok.special.start_of_role == null or tok.special.end_of_role == null) {
                return try tok.encode(user_text, allocator);
            }
            if (tok.bos_token_id != null and tok.add_bos_token) {
                try addToken(&tokens, tok.bos_token_id.?, allocator);
            }
            if (tok.special.start_of_role) |id| try addToken(&tokens, id, allocator);
            try addText(tok, &tokens, "user", allocator);
            if (tok.special.end_of_role) |id| try addToken(&tokens, id, allocator);
            if (tok.special.end_of_text) |id| try addToken(&tokens, id, allocator);
            try addText(tok, &tokens, user_text, allocator);
            try addText(tok, &tokens, "\n", allocator);
            if (tok.special.start_of_role) |id| try addToken(&tokens, id, allocator);
            try addText(tok, &tokens, "assistant", allocator);
            if (tok.special.end_of_role) |id| try addToken(&tokens, id, allocator);
            try addText(tok, &tokens, "\n", allocator);
        },
        .unknown => {
            return try tok.encode(user_text, allocator);
        },
    }

    return try tokens.toOwnedSlice(allocator);
}
