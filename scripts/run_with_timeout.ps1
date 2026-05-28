param(
    [string]$Model,
    [string]$Prompt,
    [int]$MaxTokens = 32,
    [int]$TimeoutSec = 30
)

$args = "--model $Model --prompt `"$Prompt`" --max-tokens $MaxTokens"
Write-Host "Running llama.zig $args with ${TimeoutSec}s timeout..."

$p = Start-Process -FilePath "D:\llama.zig\zig-out\bin\llama.zig.exe" -ArgumentList $args -PassThru -NoNewWindow
if (-not $p.WaitForExit($TimeoutSec * 1000)) {
    Write-Host "Timeout reached! Killing process..."
    $p | Stop-Process -Force
    exit 1
}
exit $p.ExitCode
