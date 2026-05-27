const std = @import("std");
const tokenizer = @import("tokenizer.zig");

pub const SamplerConfig = struct {
    temperature: f32 = 0.8,
    top_p: f32 = 0.9,
    seed: u64 = 0,
    repetition_window: u32 = 32,
    repetition_penalty: f32 = 1.1,
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

pub fn sampleTopP(allocator: std.mem.Allocator, logits: []const f32, cfg: SamplerConfig, prev_tokens: []const tokenizer.TokenID) !tokenizer.TokenID {
    if (cfg.temperature <= 0.0) return sampleArgmax(logits);

    var rng = std.Random.DefaultPrng.init(if (cfg.seed != 0) cfg.seed else 0xDEADBEEF);
    const random = rng.random();

    const n = logits.len;

    // Apply repetition penalty.
    var penalized = try allocator.alloc(f32, n);
    defer allocator.free(penalized);
    @memcpy(penalized, logits);

    if (cfg.repetition_penalty != 1.0 and prev_tokens.len > 0) {
        const window = @min(prev_tokens.len, @as(usize, cfg.repetition_window));
        const start = prev_tokens.len - window;
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

    var cum: f32 = 0;
    var cutoff: usize = n;
    for (indices, 0..) |idx, i| {
        cum += probs[idx];
        if (cum >= cfg.top_p) {
            cutoff = i + 1;
            break;
        }
    }

    var r = random.float(f32) * cum;
    for (indices[0..cutoff]) |idx| {
        r -= probs[idx];
        if (r <= 0) return @intCast(idx);
    }
    return @intCast(indices[0]);
}
