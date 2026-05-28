$ErrorActionPreference = "Continue"

$repo = "D:\llama.zig"
$zigExe = Join-Path $repo "zig-out\bin\llama.zig.exe"
$matrixPath = Join-Path $repo "scripts\model_matrix.json"
$prompt = "The capital of france is"
$runs = 3
$warmupRuns = 1

if (!(Test-Path $zigExe)) { throw "Missing llama.zig binary: $zigExe (run 'zig build' first)" }
if (!(Test-Path $matrixPath)) { throw "Missing model matrix: $matrixPath" }
$matrix = Get-Content -Raw $matrixPath | ConvertFrom-Json

Write-Host "llama.zig perf bench" -ForegroundColor Cyan
Write-Host "Runs per model: $runs (warmup: $warmupRuns)"
Write-Host "Matrix source: $matrixPath"

function Get-Percentile([double[]]$arr, [double]$p) {
    $s = $arr | Sort-Object
    if ($s.Count -eq 0) { return 0.0 }
    $idx = [Math]::Ceiling($p * $s.Count) - 1
    if ($idx -lt 0) { $idx = 0 }
    if ($idx -ge $s.Count) { $idx = $s.Count - 1 }
    return [double]$s[$idx]
}
function Get-Mean([double[]]$arr) {
    if ($arr.Count -eq 0) { return 0.0 }
    return ($arr | Measure-Object -Average).Average
}
function Get-StdDev([double[]]$arr) {
    if ($arr.Count -le 1) { return 0.0 }
    $mean = Get-Mean $arr
    $sum = 0.0
    foreach ($v in $arr) { $sum += [Math]::Pow(($v - $mean), 2) }
    return [Math]::Sqrt($sum / ($arr.Count - 1))
}

$allStats = @()

foreach ($m in $matrix.models) {
    if (!(Test-Path $m.path)) {
        Write-Host "[skip] $($m.name): model not found at $($m.path)" -ForegroundColor Yellow
        continue
    }

    Write-Host "`n=== $($m.name) ===" -ForegroundColor Cyan

    $samples = @()
    $enableNative = ($m.path -match "Q8_0|Q4_0|q8_0|q4_0")
    for ($i = 0; $i -lt ($warmupRuns + $runs); $i++) {
        if ($i -lt $warmupRuns) {
            $label = "warmup"
        } else {
            $label = "run$($i - $warmupRuns + 1)"
        }
        Write-Host "[candidate] llama.zig $label"
        $args = @("--model", $m.path, "--prompt", $m.prompt, "--max-tokens", "$($m.maxTokens)", "--temperature", "$($m.temperature)", "--top-k", "1", "--seed", "1", "--report-json")
        if ($enableNative) { $args += "--enable-q4q8-native" }
        $out = & $zigExe @args 2>&1
        $jsonLine = ($out | Where-Object { $_ -match '^\{"load_s":' } | Select-Object -Last 1)
        if (-not $jsonLine) {
            if ($LASTEXITCODE -ne 0) {
                throw "llama.zig run failed with exit code $LASTEXITCODE for $($m.name)"
            }
            throw "Missing JSON metrics line for $($m.name)"
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[warn] non-zero exit ($LASTEXITCODE) after metrics for $($m.name)" -ForegroundColor Yellow
        }
        if ($i -ge $warmupRuns) {
            $samples += ($jsonLine | ConvertFrom-Json)
        }
    }

    $promptVals = @($samples | ForEach-Object { [double]$_.prompt_tps })
    $genVals = @($samples | ForEach-Object { [double]$_.generation_tps })
    $loadVals = @($samples | ForEach-Object { [double]$_.load_s })

    $stat = [PSCustomObject]@{
        Name = $m.name
        Family = $m.family
        LoadMedianS = [double](Get-Percentile $loadVals 0.5)
        PromptMedianTPS = [double](Get-Percentile $promptVals 0.5)
        PromptP95TPS = [double](Get-Percentile $promptVals 0.95)
        PromptStdDev = [double](Get-StdDev $promptVals)
        GenMedianTPS = [double](Get-Percentile $genVals 0.5)
        GenP95TPS = [double](Get-Percentile $genVals 0.95)
        GenStdDev = [double](Get-StdDev $genVals)
    }
    $allStats += $stat

    Write-Host ("[stats] load_s median={0:N2}" -f $stat.LoadMedianS)
    Write-Host ("[stats] prompt_tps median={0:N2} p95={1:N2} stddev={2:N2}" -f $stat.PromptMedianTPS, $stat.PromptP95TPS, $stat.PromptStdDev)
    Write-Host ("[stats] generation_tps median={0:N2} p95={1:N2} stddev={2:N2}" -f $stat.GenMedianTPS, $stat.GenP95TPS, $stat.GenStdDev)
}

if ($allStats.Count -gt 0) {
    $outJson = Join-Path $repo "scripts\bench_results.json"
    $allStats | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $outJson
    Write-Host "`n[bench] wrote $outJson" -ForegroundColor Green
    $allStats | Format-Table -AutoSize | Out-Host
}

Write-Host "`nDone." -ForegroundColor Green
