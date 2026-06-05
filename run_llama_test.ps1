$ErrorActionPreference = "Stop"
$exe = "D:\llama.zig\zig-out\bin\llama.zig.exe"
$out = "D:\llama.zig\llama_test.txt"
$err = "D:\llama.zig\llama_test_err.txt"
if (Test-Path $out) { Remove-Item $out -Force }
if (Test-Path $err) { Remove-Item $err -Force }
$proc = Start-Process -FilePath $exe -ArgumentList @("--model", "D:\llama.zig\models\Llama-3.2-1B.Q4_K_M.gguf", "--prompt", "The capital of France is", "--max-tokens", "5", "--no-chat") -PassThru -NoNewWindow -RedirectStandardOutput $out -RedirectStandardError $err
$exited = $proc.WaitForExit(60000)
if (-not $exited) {
    $proc.Kill()
    Write-Host "[TIMEOUT]"
} else {
    Write-Host "[EXITED code=$($proc.ExitCode)]"
}
Start-Sleep -Milliseconds 500
Write-Host "--- STDOUT ---"
Get-Content $out -ErrorAction SilentlyContinue
Write-Host "--- STDERR ---"
Get-Content $err -ErrorAction SilentlyContinue | Select-Object -First 15
