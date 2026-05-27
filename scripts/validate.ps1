$ErrorActionPreference = "Stop"

$repo = "D:\llama.zig"
$refCli = Join-Path $repo "Reference\llama.cpp vulkan\llama-completion.exe"
$zigExe = Join-Path $repo "zig-out\bin\llama.zig.exe"

$models = @(
    @{
        Name = "granite"
        Path = "D:\llama.zig\models\granite-4.0-350m-BF16.gguf"
        Prompt = "The capital of france is"
        MaxTokens = "8"
        Temp = "0"
    },
    @{
        Name = "llama32_q4km"
        Path = "D:\llama.zig\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        Prompt = "The capital of france is"
        MaxTokens = "8"
        Temp = "0"
    }
)

if (!(Test-Path $refCli)) { throw "Missing reference CLI: $refCli" }
if (!(Test-Path $zigExe)) { throw "Missing llama.zig binary: $zigExe (run 'zig build' first)" }

foreach ($m in $models) {
    Write-Host "`n=== $($m.Name) ===" -ForegroundColor Cyan

    Write-Host "[reference] llama.cpp vulkan"
    & $refCli -m $m.Path -p $m.Prompt -n $m.MaxTokens --temp $m.Temp -no-cnv |
        Tee-Object -Variable refOut | Out-Host

    Write-Host "[candidate] llama.zig"
    & $zigExe --model $m.Path --prompt $m.Prompt --max-tokens $m.MaxTokens --temperature $m.Temp |
        Tee-Object -Variable zigOut | Out-Host

    $refText = ($refOut -join "`n")
    $zigText = ($zigOut -join "`n")

    if ($zigText -notmatch "\[ Prompt:\s+[0-9.]+\s+t/s \| Generation:\s+[0-9.]+\s+t/s \]") {
        throw "Missing throughput line in llama.zig output for $($m.Name)"
    }

    Write-Host "[check] throughput line present"
    Write-Host "[check] manual token comparison recommended against reference output"
}
