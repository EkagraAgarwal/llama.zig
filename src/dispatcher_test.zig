const std = @import("std");
const dispatcher_mod = @import("dispatcher.zig");
const tensor = @import("tensor.zig");
const compute_graph = @import("compute_graph.zig");

test "Dispatcher.quant_pipeline_name mapping" {
    const Dispatcher = dispatcher_mod.Dispatcher;
    
    // Q8_0
    try std.testing.expectEqualStrings("matvec_q8_0", Dispatcher.quant_pipeline_name(@intFromEnum(tensor.Type.q8_0), true));
    try std.testing.expectEqualStrings("matmul_q8_0", Dispatcher.quant_pipeline_name(@intFromEnum(tensor.Type.q8_0), false));
    
    // Q4_0
    try std.testing.expectEqualStrings("matvec_q4_0", Dispatcher.quant_pipeline_name(@intFromEnum(tensor.Type.q4_0), true));
    try std.testing.expectEqualStrings("matmul_q4_0", Dispatcher.quant_pipeline_name(@intFromEnum(tensor.Type.q4_0), false));
    
    // Fallback
    try std.testing.expectEqualStrings("matmul_f16", Dispatcher.quant_pipeline_name(compute_graph.q4_0_f16_fallback_qtype, true));
}
