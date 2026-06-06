const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub const SamplerConfig = struct {
    temperature: f32 = 0.8,
    top_k: u32 = 0,
    top_p: f32 = 0.9,
    min_p: f32 = 0.0,
    typical_p: f32 = 1.0,
    seed: u64 = 0,
    repetition_window: u32 = 32,
    repetition_penalty: f32 = 1.1,
};

pub const Sampler = struct {
    cfg: SamplerConfig,
    prng: std.Random.DefaultPrng,

    pub fn init(cfg: SamplerConfig) Sampler {
        return .{
            .cfg = cfg,
            .prng = std.Random.DefaultPrng.init(if (cfg.seed != 0) cfg.seed else 0xDEADBEEF),
        };
    }

    pub fn sample(self: *Sampler, allocator: std.mem.Allocator, logits: []const f32, prev_tokens: []const tokenizer.TokenID) !tokenizer.TokenID {
        return sampleWithRandom(allocator, logits, self.cfg, prev_tokens, self.prng.random());
    }
};

pub fn sampleArgmax(logits: []const f32) tokenizer.TokenID {
    var best: usize = 0;
    var max_v: f32 = logits[0];
    for (logits[1..], 1..) |v, i| {
        if (v > max_v) {
            max_v = v;
            best = i;
        }
    }
    return @intCast(best);
}

fn sampleWithRandom(
    allocator: std.mem.Allocator,
    logits: []const f32,
    cfg: SamplerConfig,
    prev_tokens: []const tokenizer.TokenID,
    random: std.Random,
) !tokenizer.TokenID {
    if (cfg.temperature <= 0.0) return sampleArgmax(logits);

    const n = logits.len;

    // Apply repetition penalty.
    var penalized = try allocator.alloc(f32, n);
    defer allocator.free(penalized);
    @memcpy(penalized, logits);

    if (cfg.repetition_penalty != 1.0 and prev_tokens.len > 0) {
        const window = @min(prev_tokens.len, @as(usize, cfg.repetition_window));
        const start = if (prev_tokens.len == 0) 0 else prev_tokens.len - window;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            for (prev_tokens[start..]) |pt| {
                if (@as(tokenizer.TokenID, @intCast(i)) == pt) {
                    if (penalized[i] > 0) {
                        penalized[i] /= cfg.repetition_penalty;
                    } else {
                        penalized[i] *= cfg.repetition_penalty;
                    }
                    break;
                }
            }
        }
    }

    var probs = try allocator.alloc(f32, n);
    defer allocator.free(probs);

    var max_logit: f32 = penalized[0];
    for (penalized[1..]) |v| max_logit = @max(max_logit, v);

    var sum: f32 = 0;
    for (penalized, 0..) |v, i| {
        const p = std.math.exp((v - max_logit) / cfg.temperature);
        probs[i] = p;
        sum += p;
    }
    if (sum <= 0) return sampleArgmax(logits);
    for (probs) |*p| p.* /= sum;

    // Sort indices by probability descending
    var indices = try allocator.alloc(usize, n);
    defer allocator.free(indices);
    for (0..n) |i| indices[i] = i;
    std.mem.sort(usize, indices, probs, struct {
        fn lessThan(ctx: []const f32, a: usize, b: usize) bool {
            return ctx[a] > ctx[b];
        }
    }.lessThan);

    var selected = try allocator.alloc(bool, n);
    defer allocator.free(selected);
    @memset(selected, true);

    if (cfg.top_k > 0 and cfg.top_k < n) {
        const keep = @as(usize, cfg.top_k);
        for (indices[keep..]) |idx| selected[idx] = false;
    }

    if (cfg.top_p > 0.0 and cfg.top_p < 1.0) {
        var cum_p: f32 = 0.0;
        for (indices) |idx| {
            if (!selected[idx]) continue;
            cum_p += probs[idx];
            if (cum_p > cfg.top_p) selected[idx] = false;
        }
    }

    if (cfg.min_p > 0.0) {
        const max_prob = probs[indices[0]];
        const p_thresh = max_prob * cfg.min_p;
        for (indices) |idx| {
            if (selected[idx] and probs[idx] < p_thresh) selected[idx] = false;
        }
    }

    if (cfg.typical_p > 0.0 and cfg.typical_p < 1.0) {
        var entropy: f32 = 0.0;
        for (probs) |p| {
            if (p > 0.0) entropy -= p * @log(p);
        }
        var dev = try allocator.alloc(f32, n);
        defer allocator.free(dev);
        for (probs, 0..) |p, i| {
            const surprise = if (p > 0.0) -@log(p) else std.math.inf(f32);
            dev[i] = @abs(surprise - entropy);
        }
        var typical_order = try allocator.alloc(usize, n);
        defer allocator.free(typical_order);
        for (0..n) |i| typical_order[i] = i;
        std.mem.sort(usize, typical_order, dev, struct {
            fn lessThan(ctx: []const f32, a: usize, b: usize) bool {
                return ctx[a] < ctx[b];
            }
        }.lessThan);

        var cum_typ: f32 = 0.0;
        for (typical_order) |idx| {
            if (!selected[idx]) continue;
            cum_typ += probs[idx];
            if (cum_typ > cfg.typical_p) selected[idx] = false;
        }
    }

    var total_selected: f32 = 0.0;
    for (indices) |idx| {
        if (selected[idx]) total_selected += probs[idx];
    }
    if (total_selected <= 0.0) return @intCast(indices[0]);

    var r = random.float(f32) * total_selected;
    for (indices) |idx| {
        if (!selected[idx]) continue;
        r -= probs[idx];
        if (r <= 0) return @intCast(idx);
    }
    return @intCast(indices[0]);
}

test "sampler with same seed is deterministic" {
    const alloc = std.testing.allocator;
    const logits = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const cfg = SamplerConfig{
        .temperature = 1.0,
        .top_k = 0,
        .top_p = 1.0,
        .min_p = 0.0,
        .typical_p = 1.0,
        .seed = 1234,
        .repetition_window = 0,
        .repetition_penalty = 1.0,
    };
    var s1 = Sampler.init(cfg);
    var s2 = Sampler.init(cfg);
    var seq1: [8]tokenizer.TokenID = undefined;
    var seq2: [8]tokenizer.TokenID = undefined;
    for (0..seq1.len) |i| {
        seq1[i] = try s1.sample(alloc, &logits, &.{});
        seq2[i] = try s2.sample(alloc, &logits, &.{});
    }
    try std.testing.expectEqualDeep(seq1, seq2);
}

test "top-k=1 always returns argmax" {
    const alloc = std.testing.allocator;
    const logits = [_]f32{ 2.0, 1.9, 1.8 };
    const cfg = SamplerConfig{
        .temperature = 1.0,
        .top_k = 1,
        .top_p = 1.0,
        .min_p = 0.0,
        .typical_p = 1.0,
        .seed = 1,
        .repetition_window = 0,
        .repetition_penalty = 1.0,
    };
    var s = Sampler.init(cfg);
    for (0..16) |_| {
        const id = try s.sample(alloc, &logits, &.{});
        try std.testing.expectEqual(@as(tokenizer.TokenID, 0), id);
    }
}
