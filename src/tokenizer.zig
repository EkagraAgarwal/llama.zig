const std = @import("std");
const gguf = @import("gguf.zig");
const GGUFContext = gguf.GGUFContext;

pub const TokenID = u32;

pub const SpecialTokens = struct {
    start_of_role: ?TokenID = null,
    end_of_role: ?TokenID = null,
    end_of_text: ?TokenID = null,
    begin_of_text: ?TokenID = null,
    eot_id: ?TokenID = null,
    start_header_id: ?TokenID = null,
    end_header_id: ?TokenID = null,
};

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMap(TokenID),
    id_to_token: [][]const u8,
    merges: std.StringHashMap(usize),
    bos_token_id: ?TokenID = null,
    eos_token_id: ?TokenID = null,
    pad_token_id: ?TokenID = null,
    unk_token_id: ?TokenID = null,
    add_bos_token: bool = true,
    special: SpecialTokens = .{},
    chat_template: ?[]const u8 = null,
    model_name: []const u8 = "unknown",
    use_byte_to_unicode: bool = false,

    pub fn init(allocator: std.mem.Allocator, ctx: *GGUFContext) !Tokenizer {
        var tokenizer = Tokenizer{
            .allocator = allocator,
            .vocab = std.StringHashMap(TokenID).init(allocator),
            .id_to_token = &[_][]const u8{},
            .merges = std.StringHashMap(usize).init(allocator),
        };
        errdefer tokenizer.deinit();

        // 1. Load Vocabulary
        const tokens_val = ctx.kvs.get("tokenizer.ggml.tokens") orelse return error.TokenizerTokensNotFound;
        const tokens_array = tokens_val.array;
        
        tokenizer.id_to_token = try allocator.alloc([]const u8, tokens_array.len);
        for (tokens_array, 0..) |tok, i| {
            const tok_str = try allocator.dupe(u8, tok.string);
            tokenizer.id_to_token[i] = tok_str;
            try tokenizer.vocab.put(tok_str, @as(TokenID, @intCast(i)));
        }

        // 2. Load Merges (if available)
        if (ctx.kvs.get("tokenizer.ggml.merges")) |merges_val| {
            const merges_array = merges_val.array;
            for (merges_array, 0..) |m, i| {
                const merge_str = try allocator.dupe(u8, m.string);
                try tokenizer.merges.put(merge_str, i);
            }
        }

        // 3. Load Special Tokens
        if (ctx.kvs.get("tokenizer.ggml.bos_token_id")) |val| {
            tokenizer.bos_token_id = getU32Value(val);
        }
        if (ctx.kvs.get("tokenizer.ggml.eos_token_id")) |val| {
            tokenizer.eos_token_id = getU32Value(val);
        }
        if (ctx.kvs.get("tokenizer.ggml.padding_token_id")) |val| {
            tokenizer.pad_token_id = getU32Value(val);
        }
        if (ctx.kvs.get("tokenizer.ggml.unknown_token_id")) |val| {
            tokenizer.unk_token_id = getU32Value(val);
        }
        if (ctx.kvs.get("tokenizer.ggml.add_bos_token")) |val| {
            tokenizer.add_bos_token = switch (val) {
                .bool => |b| b,
                else => true,
            };
        }
        if (ctx.kvs.get("tokenizer.chat_template")) |val| {
            if (val == .string) {
                tokenizer.chat_template = try allocator.dupe(u8, val.string);
            }
        }
        if (ctx.kvs.get("tokenizer.ggml.model")) |val| {
            if (val == .string) {
                tokenizer.model_name = try allocator.dupe(u8, val.string);
                tokenizer.use_byte_to_unicode = std.mem.eql(u8, tokenizer.model_name, "gpt2");
            }
        } else {
            if (ctx.kvs.get("general.architecture")) |arch_val| {
                if (arch_val == .string) {
                    const arch = arch_val.string;
                    tokenizer.use_byte_to_unicode = std.ascii.eqlIgnoreCase(arch, "llama") or
                                                    std.ascii.eqlIgnoreCase(arch, "granite");
                }
            }
        }

        // 4. Load Granite/Chat special tokens by scanning the vocab
        for (tokens_array, 0..) |tok_str, i| {
            if (std.mem.eql(u8, tok_str.string, "<|start_of_role|>")) tokenizer.special.start_of_role = @as(TokenID, @intCast(i));
            if (std.mem.eql(u8, tok_str.string, "<|end_of_role|>")) tokenizer.special.end_of_role = @as(TokenID, @intCast(i));
            if (std.mem.eql(u8, tok_str.string, "<|end_of_text|>")) tokenizer.special.end_of_text = @as(TokenID, @intCast(i));
            if (std.mem.eql(u8, tok_str.string, "<|begin_of_text|>")) tokenizer.special.begin_of_text = @as(TokenID, @intCast(i));
            if (std.mem.eql(u8, tok_str.string, "<|eot_id|>")) tokenizer.special.eot_id = @as(TokenID, @intCast(i));
            if (std.mem.eql(u8, tok_str.string, "<|start_header_id|>")) tokenizer.special.start_header_id = @as(TokenID, @intCast(i));
            if (std.mem.eql(u8, tok_str.string, "<|end_header_id|>")) tokenizer.special.end_header_id = @as(TokenID, @intCast(i));
        }

        return tokenizer;
    }

    pub fn deinit(self: *Tokenizer) void {
        self.vocab.deinit();

        for (self.id_to_token) |tok| {
            self.allocator.free(tok);
        }
        self.allocator.free(self.id_to_token);
        if (self.chat_template) |tpl| {
            self.allocator.free(tpl);
        }
        if (!std.mem.eql(u8, self.model_name, "unknown")) {
            self.allocator.free(self.model_name);
        }

        var merges_it = self.merges.iterator();
        while (merges_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.merges.deinit();
    }

    pub fn encode(self: *const Tokenizer, text: []const u8, out_allocator: std.mem.Allocator) ![]TokenID {
        var arena_alloc = std.heap.ArenaAllocator.init(self.allocator);
        defer arena_alloc.deinit();
        const arena = arena_alloc.allocator();

        var token_ids: std.ArrayList(TokenID) = .empty;
        errdefer token_ids.deinit(out_allocator);

        if (self.add_bos_token) {
            if (self.bos_token_id) |bos| {
                try token_ids.append(out_allocator, bos);
            }
        }

        var cursor: usize = 0;
        while (cursor < text.len) {
            if (findSpecialTokenAt(self, text, cursor)) |m| {
                try token_ids.append(out_allocator, m.id);
                cursor += m.len;
                continue;
            }
            const chunk = if (self.use_byte_to_unicode) blk: {
                var it = PreTokenizerIterator{ .text = text[cursor..] };
                const c = it.next() orelse break;
                cursor += c.len;
                break :blk c;
            } else blk: {
                const c = text[cursor..];
                cursor = text.len;
                break :blk c;
            };

            // GPT-2 style BPE uses byte-to-unicode remap. SentencePiece-style models
            // (gemma/llama/qwen variants) should merge directly on unicode pieces.
            var symbols: std.ArrayList([]const u8) = .empty;
            defer symbols.deinit(arena);
            if (self.use_byte_to_unicode) {
                for (chunk) |byte| {
                    var buf: [4]u8 = undefined;
                    const cp = byteToCodepoint(byte);
                    const len = std.unicode.utf8Encode(@as(u21, @intCast(cp)), &buf) catch unreachable;
                    const utf8_str = try arena.dupe(u8, buf[0..len]);
                    try symbols.append(arena, utf8_str);
                }
            } else {
                var utf8 = try std.unicode.Utf8View.init(chunk);
                var it_chars = utf8.iterator();
                while (it_chars.nextCodepointSlice()) |cp_slice| {
                    const s = try arena.dupe(u8, cp_slice);
                    try symbols.append(arena, s);
                }
            }

            // BPE Merge Loop
            while (symbols.items.len > 1) {
                var best_pair_idx: ?usize = null;
                var best_rank: usize = std.math.maxInt(usize);

                var i: usize = 0;
                while (i < symbols.items.len - 1) : (i += 1) {
                    const left = symbols.items[i];
                    const right = symbols.items[i + 1];
                    const pair_key = try std.fmt.allocPrint(arena, "{s} {s}", .{ left, right });
                    if (self.merges.get(pair_key)) |rank| {
                        if (rank < best_rank) {
                            best_rank = rank;
                            best_pair_idx = i;
                        }
                    }
                }

                if (best_pair_idx) |idx| {
                    const merged = try std.mem.concat(arena, u8, &[_][]const u8{ symbols.items[idx], symbols.items[idx + 1] });
                    symbols.items[idx] = merged;
                    _ = symbols.orderedRemove(idx + 1);
                } else {
                    break;
                }
            }

            // Map symbols to Token IDs. For sentencepiece-style vocabularies, handle
            // leading whitespace as U+2581 marker fallback.
            for (symbols.items, 0..) |sym, idx| {
                if (self.vocab.get(sym)) |id| {
                    try token_ids.append(out_allocator, id);
                } else {
                    if (!self.use_byte_to_unicode and idx == 0 and sym.len == 1 and sym[0] == ' ') {
                        if (self.vocab.get("▁")) |space_id| {
                            try token_ids.append(out_allocator, space_id);
                            continue;
                        }
                    }
                    if (self.unk_token_id) |unk| {
                        try token_ids.append(out_allocator, unk);
                    }
                }
            }
        }

        return try token_ids.toOwnedSlice(out_allocator);
    }

    pub fn decode(self: *const Tokenizer, tokens: []const TokenID, writer: anytype) !void {
        for (tokens) |id| {
            if (id == self.bos_token_id or id == self.eos_token_id or id == self.pad_token_id) {
                continue;
            }
            if (id >= self.id_to_token.len) continue;
            const token_str = self.id_to_token[id];

            if (self.use_byte_to_unicode) {
                // Revert byte-to-unicode mapping
                var it = std.unicode.Utf8View.init(token_str) catch continue;
                var cp_it = it.iterator();
                while (cp_it.nextCodepoint()) |cp| {
                    if (codepointToByte(cp)) |byte| {
                        try writer.writeByte(byte);
                    }
                }
            } else {
                // SentencePiece-style marker for whitespace.
                var it = std.unicode.Utf8View.init(token_str) catch continue;
                var cp_it = it.iterator();
                while (cp_it.nextCodepoint()) |cp| {
                    if (cp == 0x2581) {
                        try writer.writeByte(' ');
                    } else {
                        var buf: [4]u8 = undefined;
                        const len = std.unicode.utf8Encode(@as(u21, @intCast(cp)), &buf) catch continue;
                        try writer.writeAll(buf[0..len]);
                    }
                }
            }
        }
    }
};

fn getU32Value(val: gguf.MetadataValue) u32 {
    return switch (val) {
        .u8 => |v| @as(u32, v),
        .u16 => |v| @as(u32, v),
        .u32 => |v| v,
        .u64 => |v| @as(u32, @intCast(v)),
        .i8 => |v| @as(u32, @intCast(v)),
        .i16 => |v| @as(u32, @intCast(v)),
        .i32 => |v| @as(u32, @intCast(v)),
        .i64 => |v| @as(u32, @intCast(v)),
        else => 0,
    };
}

pub fn byteToCodepoint(b: u8) u32 {
    if ((b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255)) {
        return b;
    }
    var rank: u32 = 0;
    var i: u32 = 0;
    while (i < b) : (i += 1) {
        if (!((i >= 33 and i <= 126) or (i >= 161 and i <= 172) or (i >= 174 and i <= 255))) {
            rank += 1;
        }
    }
    return 256 + rank;
}

pub fn codepointToByte(c: u32) ?u8 {
    if ((c >= 33 and c <= 126) or (c >= 161 and c <= 172) or (c >= 174 and c <= 255)) {
        return @as(u8, @intCast(c));
    }
    if (c >= 256 and c < 324) {
        const rank = c - 256;
        var r: u32 = 0;
        var b: u32 = 0;
        while (b < 256) : (b += 1) {
            if (!((b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255))) {
                if (r == rank) {
                    return @as(u8, @intCast(b));
                }
                r += 1;
            }
        }
    }
    return null;
}

const PreTokenizerIterator = struct {
    text: []const u8,
    index: usize = 0,

    pub fn next(self: *PreTokenizerIterator) ?[]const u8 {
        if (self.index >= self.text.len) return null;
        const start = self.index;
        
        var has_space = false;
        var ptr = start;
        if (self.text[ptr] == ' ') {
            has_space = true;
            ptr += 1;
        }

        if (ptr < self.text.len and std.ascii.isAlphanumeric(self.text[ptr])) {
            var end = ptr + 1;
            while (end < self.text.len and std.ascii.isAlphanumeric(self.text[end])) : (end += 1) {}
            self.index = end;
            return self.text[start..end];
        }

        if (has_space) {
            self.index = start + 1;
            return self.text[start .. start + 1];
        }

        const first = self.text[start];
        if (std.ascii.isWhitespace(first)) {
            var end = start + 1;
            while (end < self.text.len and std.ascii.isWhitespace(self.text[end])) : (end += 1) {}
            self.index = end;
            return self.text[start..end];
        } else {
            const len = std.unicode.utf8ByteSequenceLength(first) catch 1;
            const end = @min(self.text.len, start + len);
            self.index = end;
            return self.text[start..end];
        }
    }
};

const TokenMatch = struct {
    id: TokenID,
    len: usize,
};

fn findSpecialTokenAt(self: *const Tokenizer, text: []const u8, at: usize) ?TokenMatch {
    var best: ?TokenMatch = null;
    for (self.id_to_token, 0..) |tok, i| {
        if (tok.len < 4) continue;
        if (!(tok[0] == '<' and tok[tok.len - 1] == '>')) continue;
        if (at + tok.len > text.len) continue;
        if (std.mem.eql(u8, text[at .. at + tok.len], tok)) {
            if (best == null or tok.len > best.?.len) {
                best = .{ .id = @as(TokenID, @intCast(i)), .len = tok.len };
            }
        }
    }
    return best;
}

test "encode preserves special token spans" {
    const alloc = std.testing.allocator;
    var t = Tokenizer{
        .allocator = alloc,
        .vocab = std.StringHashMap(TokenID).init(alloc),
        .id_to_token = try alloc.alloc([]const u8, 7),
        .merges = std.StringHashMap(usize).init(alloc),
        .bos_token_id = null,
        .eos_token_id = null,
        .pad_token_id = null,
        .unk_token_id = null,
        .add_bos_token = false,
        .special = .{},
    };
    defer t.deinit();

    const toks = [_][]const u8{ "<|begin_of_text|>", "h", "e", "l", "o", " ", "!" };
    for (toks, 0..) |tok, i| {
        const owned = try alloc.dupe(u8, tok);
        t.id_to_token[i] = owned;
        try t.vocab.put(owned, @as(TokenID, @intCast(i)));
    }

    const ids = try t.encode("<|begin_of_text|>hello!", alloc);
    defer alloc.free(ids);
    try std.testing.expect(ids.len >= 2);
    try std.testing.expectEqual(@as(TokenID, 0), ids[0]);
}
