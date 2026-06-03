param(
    [ValidateSet("4B","9B","Llama-1B","Llama-3B","Granite-350M")]
    [string]$Model = "4B",
    [int]$TimeoutSec = 90,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'

$modelPath = switch ($Model) {
    "4B"          { "D:\llama.zig\models\Qwen3.5-4B-Q4_K_M.gguf" }
    "9B"          { "D:\llama.zig\models\Qwen3.5-9B-Q4_K_M.gguf" }
    "Llama-1B"    { "D:\llama.zig\models\Llama-3.2-1B.Q4_K_M.gguf" }
    "Llama-3B"    { "D:\llama.zig\models\Llama-3.2-3B-Q4_K_M.gguf" }
    "Granite-350M"{ "D:\llama.zig\models\granite-4.0-350m-BF16.gguf" }
}

if (-not (Test-Path $modelPath)) {
    Write-Host "Model file not found: $modelPath"
    exit 2
}

if (-not $SkipBuild) {
    Write-Host "Building llama.zig (ReleaseFast)..."
    & zig build -Doptimize=ReleaseFast
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path "D:\llama.zig\zig-out\bin\llama.zig.exe")) {
    Write-Host "Binary not found after build"
    exit 3
}

$outFile = "D:\llama.zig\smoke_${Model}.out"
$errFile = "D:\llama.zig\smoke_${Model}.err"
if (Test-Path $outFile) { Remove-Item $outFile }
if (Test-Path $errFile) { Remove-Item $errFile }

Write-Host "Running $Model (timeout ${TimeoutSec}s)..."
$proc = Start-Process -FilePath "D:\llama.zig\zig-out\bin\llama.zig.exe" `
    -ArgumentList @("--model", $modelPath, "--prompt", "The capital of France is", `
                    "--max-tokens", "8", "--temperature", "0", "--seed", "1", "--no-chat", "--report-json") `
    -RedirectStandardOutput $outFile `
    -RedirectStandardError  $errFile `
    -PassThru

$exited = $proc.WaitForExit($TimeoutSec * 1000)
$timedOut = -not $exited
if ($timedOut) {
    Write-Host "TIMEOUT: killing PID $($proc.Id) after ${TimeoutSec}s"
    Stop-Process -Id $proc.Id -Force
    $code = 124
} else {
    $code = $proc.ExitCode
}

$out = ""
$err = ""
if (Test-Path $outFile) { $out = Get-Content $outFile -Raw }
if (Test-Path $errFile) { $err = Get-Content $errFile -Raw }

$combined = $out + $err
$hasPanic   = ($combined -match "panic:|error:")
$hasParis   = $out -match "Paris"
$hasAnyText = $out.Length -gt 50

Write-Host "Model=$Model exit=$code timedOut=$timedOut panic=$hasPanic paris=$hasParis anyText=$hasAnyText"
if ($combined.Length -gt 0) {
    Write-Host "--- STDOUT (last 800 chars) ---"
    Write-Host ($out.Substring([Math]::Max(0, $out.Length - 800)))
    Write-Host "--- STDERR (last 800 chars) ---"
    Write-Host ($err.Substring([Math]::Max(0, $err.Length - 800)))
}

if ($hasPanic) { exit 1 }
if (-not $hasAnyText) { exit 2 }
if (($Model -eq "4B" -or $Model -eq "9B") -and -not $hasParis) { exit 3 }
exit $code
