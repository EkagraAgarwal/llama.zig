# run_tests.ps1
# Master test runner orchestrating all test phases

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "STARTING ALL TEST PHASES FOR LLAMA.ZIG (WITH MMAP)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Phase 1: Build the project
Write-Host "`n[PHASE 1] Building project (zig build -Doptimize=ReleaseFast)..." -ForegroundColor Yellow
powershell -File scripts/test_with_timeout.ps1 -TestCommand "zig build -Doptimize=ReleaseFast" -TimeoutSec 120 -FailOnTimeout
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed!"
    exit 1
}
Write-Host "[PHASE 1] Build successful!" -ForegroundColor Green

# Phase 2: CPU & Parity Unit Tests
Write-Host "`n[PHASE 2] Running CPU/Parity Unit Tests (zig build test)..." -ForegroundColor Yellow
powershell -File scripts/test_with_timeout.ps1 -TestCommand "zig build test" -TimeoutSec 60 -FailOnTimeout
if ($LASTEXITCODE -ne 0) {
    Write-Error "CPU Unit tests failed!"
    exit 1
}
Write-Host "[PHASE 2] CPU Unit tests passed!" -ForegroundColor Green

# Phase 3: mmap Unit Tests
Write-Host "`n[PHASE 3] Running mmap Unit Tests (zig build test-mmap)..." -ForegroundColor Yellow
powershell -File scripts/test_with_timeout.ps1 -TestCommand "zig build test-mmap" -TimeoutSec 30 -FailOnTimeout
if ($LASTEXITCODE -ne 0) {
    Write-Error "mmap Unit tests failed!"
    exit 1
}
Write-Host "[PHASE 3] mmap Unit tests passed!" -ForegroundColor Green

# Phase 4: GGUF/mmap Load Integration Tests
Write-Host "`n[PHASE 4] Running GGUF Model Loading Integration Tests (zig build test-integration)..." -ForegroundColor Yellow
powershell -File scripts/test_with_timeout.ps1 -TestCommand "zig build test-integration" -TimeoutSec 60 -FailOnTimeout
if ($LASTEXITCODE -ne 0) {
    Write-Error "Model Loading Integration tests failed!"
    exit 1
}
Write-Host "[PHASE 4] Model Loading Integration tests passed!" -ForegroundColor Green

# Phase 5: End-to-End Inference Integration Tests
Write-Host "`n[PHASE 5] Running End-to-End Inference Integration Tests..." -ForegroundColor Yellow

$modelMatrixPath = "scripts/model_matrix.json"
if (-not (Test-Path $modelMatrixPath)) {
    Write-Error "Model matrix JSON file not found at $modelMatrixPath"
    exit 1
}

$matrix = Get-Content $modelMatrixPath | ConvertFrom-Json
$models = $matrix.models

$failedTests = 0
$totalTests = 0

foreach ($m in $models) {
    $totalTests++
    Write-Host "`n--- Testing Model: $($m.name) ---" -ForegroundColor Cyan
    Write-Host "Path: $($m.path)"
    
    if (-not (Test-Path $m.path)) {
        Write-Warning "Model file not found: $($m.path). Skipping inference test."
        continue
    }

    $prompt = $m.prompt
    $maxTokens = $m.maxTokens
    $temp = $m.temperature
    
    # Build CLI command
    $cmd = "zig-out\bin\llama.zig.exe --model `"$($m.path)`" --prompt `"$prompt`" --max-tokens $maxTokens --temperature $temp"
    
    Write-Host "Executing: $cmd"
    $jsonResultStr = powershell -File scripts/test_with_timeout.ps1 -TestCommand $cmd -TimeoutSec 120
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Execution failed with exit code $LASTEXITCODE" -ForegroundColor Red
        $failedTests++
        continue
    }

    # Parse JSON result
    $result = $jsonResultStr | ConvertFrom-Json
    if ($result.status -eq "timeout") {
        Write-Host "Test timed out!" -ForegroundColor Red
        $failedTests++
        continue
    }

    if ($result.exit_code -ne 0) {
        Write-Host "Executable returned exit code $($result.exit_code)" -ForegroundColor Red
        Write-Host "Stderr: $($result.stderr)"
        $failedTests++
        continue
    }

    $stdout = $result.stdout
    Write-Host "Inference completed in $($result.elapsed_s)s" -ForegroundColor Green
    Write-Host "Generated text:"
    Write-Host $stdout -ForegroundColor Gray

    # Optional output validation
    if ($m.expectedContains) {
        if ($stdout -match $m.expectedContains) {
            Write-Host "Validation Passed: contains expected string '$($m.expectedContains)'" -ForegroundColor Green
        } else {
            Write-Host "Validation Failed: expected string '$($m.expectedContains)' not found in output!" -ForegroundColor Red
            $failedTests++
        }
    }
}

Write-Host "`n==================================================" -ForegroundColor Cyan
if ($failedTests -eq 0) {
    Write-Host "ALL TESTS PASSED SUCCESSFULLY! ($totalTests/$totalTests)" -ForegroundColor Green
    exit 0
} else {
    Write-Host "TEST SUITE FAILED: $failedTests of $totalTests tests failed!" -ForegroundColor Red
    exit 1
}
