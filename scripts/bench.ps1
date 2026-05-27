$ErrorActionPreference = "Continue"

$repo = "D:\llama.zig"
$refBench = Join-Path $repo "Reference\llama.cpp vulkan\llama-bench.exe"
$refCli = Join-Path $repo "Reference\llama.cpp vulkan\llama-cli.exe"
$zigExe = Join-Path $repo "zig-out\bin\llama.zig.exe"
$refSrc = Join-Path $repo "Reference\llama.cpp-src"
$prompt = "The capital of france is"

$models = @(
    @{
        Name = "granite_bf16"
        Path = "D:\llama.zig\models\granite-4.0-350m-BF16.gguf"
    },
    @{
        Name = "llama32_q4km"
        Path = "D:\llama.zig\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    }
)

if (!(Test-Path $zigExe)) { throw "Missing llama.zig binary: $zigExe (run 'zig build' first)" }

$refCommit = "unknown"
if (Test-Path (Join-Path $refSrc ".git")) {
    Push-Location $refSrc
    $refCommit = (git rev-parse HEAD)
    Pop-Location
}

Write-Host "llama.zig perf bench" -ForegroundColor Cyan
Write-Host "Reference llama.cpp-src commit: $refCommit"

foreach ($m in $models) {
    if (!(Test-Path $m.Path)) {
        Write-Host "[skip] $($m.Name): model not found at $($m.Path)" -ForegroundColor Yellow
        continue
    }

    Write-Host "`n=== $($m.Name) ===" -ForegroundColor Cyan

    if (Test-Path $refBench) {
        Write-Host "[reference] llama-bench (pp128 tg64)"
        & $refBench -m $m.Path -p 128 -n 64 2>&1 | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "reference llama-bench failed with exit code $LASTEXITCODE"
        }
    } elseif (Test-Path $refCli) {
        Write-Host "[reference] llama-cli smoke"
        & $refCli -m $m.Path -p $prompt -n 8 --temp 0 -no-cnv 2>&1 | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "reference llama-cli failed with exit code $LASTEXITCODE"
        }
    } else {
        Write-Host "[reference] skipped (no llama-bench or llama-cli)" -ForegroundColor Yellow
    }

    Write-Host "[candidate] llama.zig smoke + throughput"
    & $zigExe --model $m.Path --prompt $prompt --max-tokens 8 --temperature 0 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "llama.zig run failed with exit code $LASTEXITCODE"
    }
}

Write-Host "`nDone." -ForegroundColor Green
