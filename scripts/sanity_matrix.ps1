param(
    [string]$MatrixPath = "D:\llama.zig\scripts\model_matrix.json",
    [string]$BinaryPath = "D:\llama.zig\zig-out\bin\llama.zig.exe"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $MatrixPath)) { throw "Missing matrix file: $MatrixPath" }
if (!(Test-Path $BinaryPath)) { throw "Missing llama.zig binary: $BinaryPath (run 'zig build' first)" }

$matrix = Get-Content -Raw $MatrixPath | ConvertFrom-Json
$results = @()

foreach ($m in $matrix.models) {
    if (!(Test-Path $m.path)) {
        Write-Host "[skip] $($m.name): model missing at $($m.path)" -ForegroundColor Yellow
        continue
    }

    Write-Host "`n=== $($m.name) ($($m.family)) ===" -ForegroundColor Cyan
    $enableNative = ($m.path -match "Q8_0|Q4_0|q8_0|q4_0")
    $args = @("--model", $m.path, "--prompt", $m.prompt, "--max-tokens", "$($m.maxTokens)", "--temperature", "$($m.temperature)", "--top-k", "1", "--seed", "1", "--report-json")
    if ($enableNative) { $args += "--enable-q4q8-native" }
    $output = & $BinaryPath @args 2>&1
    $fullText = ($output -join "`n")
    $jsonLine = ($output | Where-Object { $_ -match '^\{"load_s":' } | Select-Object -Last 1)
    if (-not $jsonLine) {
        if ($LASTEXITCODE -ne 0) {
            throw "Run failed for $($m.name) with exit code $LASTEXITCODE"
        }
        throw "Missing JSON metrics line for $($m.name)"
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[warn] non-zero exit ($LASTEXITCODE) after metrics were produced for $($m.name)" -ForegroundColor Yellow
    }
    $metrics = $jsonLine | ConvertFrom-Json

    if ($m.PSObject.Properties.Name -contains "expectedContains" -and $m.expectedContains) {
        if ($fullText -notmatch [regex]::Escape($m.expectedContains)) {
            throw "Output sanity check failed for $($m.name): expected '$($m.expectedContains)'"
        }
    }

    $results += [PSCustomObject]@{
        Name = $m.name
        Family = $m.family
        Prompt = $m.prompt
        LoadS = [double]$metrics.load_s
        PromptTPS = [double]$metrics.prompt_tps
        GenerationTPS = [double]$metrics.generation_tps
        GraphNodes = [int]$metrics.graph_nodes
        OutputSnippet = (($fullText -replace "`r"," ") -replace "`n"," ") -replace "\s+"," "
    }
}

if ($results.Count -eq 0) {
    throw "No matrix entries executed (all models missing)"
}

Write-Host "`n=== Matrix Summary ===" -ForegroundColor Green
$results | Format-Table -AutoSize | Out-Host

$outJson = "D:\llama.zig\scripts\sanity_results.json"
$results | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $outJson
Write-Host "[sanity] wrote $outJson" -ForegroundColor Green
