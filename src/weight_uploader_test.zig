const std = @import("std");
const weight_uploader = @import("weight_uploader.zig");

test "getFusedComponentNames mapping" {
    const allocator = std.testing.allocator;
    
    const attn = try weight_uploader.getFusedComponentNames(allocator, "blk.0.attn_qkv.weight");
    defer if (attn) |comps| { for (comps) |c| allocator.free(c); allocator.free(comps); };
    try std.testing.expect(attn != null);
    try std.testing.expectEqualStrings("blk.0.attn_q.weight", attn.?[0]);
    try std.testing.expectEqualStrings("blk.0.attn_k.weight", attn.?[1]);
    try std.testing.expectEqualStrings("blk.0.attn_v.weight", attn.?[2]);
    
    const ffn = try weight_uploader.getFusedComponentNames(allocator, "blk.1.ffn_gate_up.weight");
    defer if (ffn) |comps| { for (comps) |c| allocator.free(c); allocator.free(comps); };
    try std.testing.expect(ffn != null);
    try std.testing.expectEqualStrings("blk.1.ffn_gate.weight", ffn.?[0]);
    try std.testing.expectEqualStrings("blk.1.ffn_up.weight", ffn.?[1]);
}
