//! CPU-side per-decode step of the Gated Delta Net (GDN) recurrence.
//!
//! In Qwen 3.5 hybrid models, every layer that is NOT a periodic full-attention
//! layer uses a linear attention block called "Gated Delta Net". The conv1d
//! pre-step and gated_norm post-step run on the GPU, but the actual per-token
//! recurrent state update runs on the CPU between decode steps. This file
//! holds that step plus the per-layer state storage.
//!
//! Algorithm (mirrors `Reference/llama.cpp-src/ggml/src/ggml.c:6180`):
//!   a. S = S * exp(g)             (decay — scalar or per-element)
//!   b. s_k[v] = sum_k S[k,v] * k[k]   (length head_v_dim scratch)
//!   c. d[v]   = (v[v] - s_k[v]) * beta  (delta — scalar)
//!   d. S[k,v] = S[k,v] + k[k] * d[v]   (outer-product update)
//!   e. o[v]   = sum_k S[k,v] * q[k] * scale  (output)
//!
//! Per-layer state lives in two flat host-visible buffers. The ZINC team
//! (a similar Zig LLM project) found that keeping the recurrent state on the
//! CPU between decode steps is the only stable option — GPU-resident state
//! grows from ~0.004 to >20,000 over a long decode run, indicating FP
//! drift in the per-token update.

const std = @import("std");
const model = @import("model.zig");

/// Per-layer view of the SSM state.
///
/// `conv` is a rolling window of the previous (d_conv - 1) chunks of the
/// conv1d output, channel-major. `rec` is the per-head recurrent state
/// matrix, head-major, shape (num_v_heads, head_k_dim, head_v_dim).
pub const SsmLayerState = struct {
    conv: []f32,
    rec: []f32,
};

/// Owns the flat per-layer state for the Gated Delta Net recurrence.
///
/// Two big host-visible f32 buffers (one for conv state, one for recurrent
/// state) cover all main layers. The CPU step operates on slices of these
/// buffers per layer. The GPU dispatch path maps the same buffers via
/// Vulkan BDA in `compute_graph.Dispatcher.ssmCacheLayerOffset`.
pub const SsmCpuContext = struct {
    cfg: *const model.ModelConfig,
    /// Per-layer view into `conv_buf`. The slice is non-owning.
    conv_per_layer: u32,
    rec_per_layer: u32,
    n_main: u32,
    conv_buf: []f32,
    rec_buf: []f32,
    allocator: std.mem.Allocator,
    owns_storage: bool,

    pub fn init(allocator: std.mem.Allocator, cfg: *const model.ModelConfig) !SsmCpuContext {
        const n_main = cfg.n_layer -| cfg.nextn_predict_layers;
        const head_v_dim: u32 = if (cfg.ssm_dt_rank > 0) cfg.ssm_d_inner / cfg.ssm_dt_rank else 0;
        const head_k_dim: u32 = cfg.ssm_d_state;
        const d_conv: u32 = if (cfg.ssm_d_conv > 0) cfg.ssm_d_conv else 4;
        const conv_channels: u32 = cfg.ssm_d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state;
        const conv_per_layer: u32 = (d_conv - 1) * conv_channels;
        const rec_per_layer: u32 = head_v_dim * head_k_dim * cfg.ssm_dt_rank;
        const conv_total: u32 = conv_per_layer * n_main;
        const rec_total: u32 = rec_per_layer * n_main;

        const conv_buf = try allocator.alloc(f32, conv_total);
        errdefer allocator.free(conv_buf);
        const rec_buf = try allocator.alloc(f32, rec_total);
        errdefer allocator.free(rec_buf);
        @memset(conv_buf, 0.0);
        @memset(rec_buf, 0.0);

        return .{
            .cfg = cfg,
            .conv_per_layer = conv_per_layer,
            .rec_per_layer = rec_per_layer,
            .n_main = n_main,
            .conv_buf = conv_buf,
            .rec_buf = rec_buf,
            .allocator = allocator,
            .owns_storage = true,
        };
    }

    /// Wrap externally-owned buffers (for testing or for direct GPU mapping).
    pub fn wrap(
        cfg: *const model.ModelConfig,
        conv_buf: []f32,
        rec_buf: []f32,
    ) SsmCpuContext {
        const n_main = cfg.n_layer -| cfg.nextn_predict_layers;
        const head_v_dim: u32 = if (cfg.ssm_dt_rank > 0) cfg.ssm_d_inner / cfg.ssm_dt_rank else 0;
        const head_k_dim: u32 = cfg.ssm_d_state;
        const d_conv: u32 = if (cfg.ssm_d_conv > 0) cfg.ssm_d_conv else 4;
        const conv_channels: u32 = cfg.ssm_d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state;
        return .{
            .cfg = cfg,
            .conv_per_layer = (d_conv - 1) * conv_channels,
            .rec_per_layer = head_v_dim * head_k_dim * cfg.ssm_dt_rank,
            .n_main = n_main,
            .conv_buf = conv_buf,
            .rec_buf = rec_buf,
            .allocator = undefined,
            .owns_storage = false,
        };
    }

    pub fn deinit(self: *SsmCpuContext) void {
        if (self.owns_storage) {
            self.allocator.free(self.conv_buf);
            self.allocator.free(self.rec_buf);
        }
    }

    /// Borrow the state for a single layer. The returned slices point into
    /// `self.conv_buf` / `self.rec_buf` and are invalidated by `deinit`.
    pub fn getLayer(self: *SsmCpuContext, layer: u32) SsmLayerState {
        std.debug.assert(layer < self.n_main);
        return .{
            .conv = self.conv_buf[self.conv_per_layer * layer ..][0..self.conv_per_layer],
            .rec = self.rec_buf[self.rec_per_layer * layer ..][0..self.rec_per_layer],
        };
    }

    /// Append `new_chunk` to the conv rolling window for `layer` and drop the
    /// oldest entry. `new_chunk.len` must equal `conv_channels`.
    ///
    /// Layout: state has `(d_conv - 1)` rows, each of length `conv_channels`,
    /// stored row-major. After this call:
    ///   - state[0] is the second-oldest chunk (oldest was dropped)
    ///   - state[d_conv-2] is the newest chunk in state (just before the
    ///     current one, which the shader reads from a separate buffer)
    pub fn stepConv1d(self: *SsmCpuContext, layer: u32, new_chunk: []const f32) void {
        const s = self.getLayer(layer);
        const conv_channels: u32 = self.cfg.ssm_d_inner + 2 * self.cfg.ssm_n_group * self.cfg.ssm_d_state;
        std.debug.assert(new_chunk.len == conv_channels);
        const d_conv: u32 = if (self.cfg.ssm_d_conv > 0) self.cfg.ssm_d_conv else 4;
        if (d_conv >= 2) {
            // Shift toward lower indices: state[k] = state[k+1] for k=0..d_conv-3.
            // This drops the oldest entry (was state[0]) and makes room for the
            // new chunk at the highest index (state[d_conv-2]).
            var k: u32 = 0;
            while (k < d_conv - 2) : (k += 1) {
                const dst = s.conv[k * conv_channels ..][0..conv_channels];
                const src = s.conv[(k + 1) * conv_channels ..][0..conv_channels];
                @memcpy(dst, src);
            }
            @memcpy(s.conv[(d_conv - 2) * conv_channels ..][0..conv_channels], new_chunk);
        }
    }

    /// One token of the Gated Delta Net recurrence. Updates the per-head
    /// state matrix in-place and writes the output into `out`.
    ///
    /// Inputs (all length `head_v_dim * num_v_heads`; the first `head_v_dim`
    /// elements are head 0, the next are head 1, etc.):
    ///   - `q`: query vector (head_k_dim * num_k_heads)
    ///   - `k`: key vector (head_k_dim * num_k_heads)
    ///   - `v`: value vector (head_v_dim * num_v_heads)
    ///   - `g`: gate — scalar (gated delta net proper) or per-element (KDA)
    ///   - `beta`: per-head scalar
    ///   - `out`: pre-allocated output buffer (head_v_dim * num_v_heads)
    ///   - `scale`: output scaling factor (typically 1.0 / sqrt(head_v_dim))
    pub fn stepDeltaNet(
        self: *SsmCpuContext,
        layer: u32,
        q: []const f32,
        k: []const f32,
        v: []const f32,
        g: []const f32,
        beta: []const f32,
        out: []f32,
        scale: f32,
    ) !void {
        const cfg = self.cfg;
        const s = self.getLayer(layer);
        const head_v_dim: u32 = if (cfg.ssm_dt_rank > 0) cfg.ssm_d_inner / cfg.ssm_dt_rank else return error.InvalidModelConfig;
        const num_v_heads: u32 = cfg.ssm_dt_rank;
        const head_k_dim: u32 = cfg.ssm_d_state;
        const num_k_heads: u32 = cfg.ssm_n_group;
        const rep: u32 = if (num_k_heads > 0) num_v_heads / num_k_heads else 1;
        
        const rec_per_head = head_k_dim * head_v_dim;
        const total_per_layer: u32 = rec_per_head * num_v_heads;
        std.debug.assert(s.rec.len >= total_per_layer);
        std.debug.assert(q.len == head_k_dim * num_k_heads);
        std.debug.assert(k.len == head_k_dim * num_k_heads);
        std.debug.assert(v.len == head_v_dim * num_v_heads);
        std.debug.assert(out.len == head_v_dim * num_v_heads);
        std.debug.assert(beta.len == num_v_heads);

        const kda = g.len == head_v_dim * num_v_heads;

        var sk = try self.allocator.alloc(f32, head_v_dim);
        defer self.allocator.free(sk);

        var h: u32 = 0;
        while (h < num_v_heads) : (h += 1) {
            const hk: u32 = h / rep;
            const s_head_off = h * rec_per_head;
            const S = s.rec[s_head_off..][0..rec_per_head];

            // a. Decay: S *= exp(g) — scalar or per-element.
            //    S shape (H_k, H_v), row-major. S[k, v].
            if (kda) {
                // KDA: column-wise scaling. S[:, c] *= exp(g[c]).
                const g_h = g[h * head_v_dim ..][0..head_v_dim];
                for (0..head_v_dim) |vi| {
                    const decay = @exp(std.math.clamp(g_h[vi], -30.0, 30.0));
                    for (0..head_k_dim) |ki| {
                        S[ki * head_v_dim + vi] *= decay;
                    }
                }
            } else {
                const decay = @exp(std.math.clamp(g[h], -30.0, 30.0));
                for (0..rec_per_head) |i| S[i] *= decay;
            }

            // b. s_k[v] = sum_k S[k,v] * k[k].
            const k_h = k[hk * head_k_dim ..][0..head_k_dim];
            @memset(sk, 0.0);
            for (0..head_k_dim) |ki| {
                const kv = k_h[ki];
                for (0..head_v_dim) |vi| {
                    sk[vi] += S[ki * head_v_dim + vi] * kv;
                }
            }

            // c. d[v] = (v[v] - s_k[v]) * beta_h
            const v_h = v[h * head_v_dim ..][0..head_v_dim];
            const b_h = beta[h];
            // S[k,v] += k[k] * d[v]
            for (0..head_v_dim) |vi| {
                const dv = (v_h[vi] - sk[vi]) * b_h;
                for (0..head_k_dim) |ki| {
                    S[ki * head_v_dim + vi] += k_h[ki] * dv;
                }
            }

            // e. o[v] = sum_k S[k,v] * q[k] * scale.
            const q_h = q[hk * head_k_dim ..][0..head_k_dim];
            const out_h = out[h * head_v_dim ..][0..head_v_dim];
            for (0..head_v_dim) |vi| {
                var acc: f32 = 0.0;
                for (0..head_k_dim) |ki| {
                    acc += S[ki * head_v_dim + vi] * q_h[ki];
                }
                out_h[vi] = acc * scale;
            }
        }
    }
};
