$ErrorActionPreference = "Stop"
$repo = "D:\llama.zig"
$sanityScript = Join-Path $repo "scripts\sanity_matrix.ps1"
$matrixFile = Join-Path $repo "scripts\model_matrix.json"

if (!(Test-Path $sanityScript)) { throw "Missing matrix script: $sanityScript" }
if (!(Test-Path $matrixFile)) { throw "Missing matrix file: $matrixFile" }

Write-Host "[matrix] running deterministic family sanity checks" -ForegroundColor Cyan
& $sanityScript -MatrixPath $matrixFile
