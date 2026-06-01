# Architecture-First Refactor Implementation

## Objective
Make the codebase easier to extend with new model architectures by centralizing architecture dispatch, reducing CLI orchestration duplication, and preserving current Llama and Granite behavior.

## Constraints
- Supported architectures for this refactor: `llama`, `granite`.
- Qwen and Gemma are future work and should not remain in the supported architecture enum.
- Preserve current performance-oriented paths: fused QKV, fused gate/up, native quant matmul/matvec, GPU embedding, batched embed+graph submission, GPU top-k.
- Use timeouts for inference smoke tests.

## Implementation Slices
1. Architecture boundary
   - Remove Qwen/Gemma from supported architecture dispatch.
   - Keep unknown architectures as explicit unsupported cases.
   - Update model tests to reflect Llama/Granite-only support.

2. Model graph dispatch
   - Route graph construction through `src/models/interface.zig`.
   - Keep the shared Llama-family builder for Llama and Granite.
   - Stop building transformer layers inline in `main.zig`.

3. Weight and graph optimization extraction
   - Move quantized matmul rewriting into `weights_loader` or a small graph optimizer helper.
   - Use the existing fused component lookup from `weights_loader`.
   - Remove duplicate helper implementations from `main.zig` once no longer used.

4. Runtime modularization
   - Wire `main.zig` toward `cli.zig`, `weights_loader.zig`, and `engine.zig`.
   - Keep behavior unchanged and shrink `main.zig` incrementally.

## Validation Loop
After each slice:

```powershell
zig build test
zig build clean
zig build
powershell -NoProfile -Command "& { $p = Start-Process -FilePath '.\zig-out\bin\llama.zig.exe' -ArgumentList @('--model','models\Llama3.1-8B-Q4_K_M.gguf','--prompt','What is the capital of france ','--max-tokens','8','--temperature','0','--top-k','1','--ctx-size','2048','--report-json') -NoNewWindow -PassThru; if (-not $p.WaitForExit(180000)) { Stop-Process -Id $p.Id -Force; throw 'Inference smoke test timed out' }; exit $p.ExitCode }"
```

## Current Baseline Notes
- `zig build test` passed before implementation.
- The current smoke test uses `models\Llama3.1-8B-Q4_K_M.gguf` with a 180 second timeout.
