param(
    [string]$TestCommand,
    [int]$TimeoutSec = 120,
    [switch]$FailOnTimeout
)

$stdoutFile = [System.IO.Path]::GetTempFileName()
$stderrFile = [System.IO.Path]::GetTempFileName()

try {
    # Start the process with file redirection
    $process = Start-Process powershell.exe -ArgumentList "-NoProfile -NonInteractive -Command `"$TestCommand`"" -NoNewWindow -PassThru -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
    
    # Wait loop
    $completed = $false
    $elapsed = 0
    while ($elapsed -lt $TimeoutSec) {
        if ($process.HasExited) {
            $completed = $true
            $process.WaitForExit() | Out-Null
            break
        }
        Start-Sleep -Seconds 1
        $elapsed++
    }
    
    if (-not $completed) {
        try {
            # Terminate any child processes first (the process tree)
            Get-CimInstance Win32_Process -Filter "ParentProcessId = $($process.Id)" -ErrorAction SilentlyContinue | ForEach-Object {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            }
            $process | Stop-Process -Force -ErrorAction SilentlyContinue
        } catch {}
        
        $stdout = Get-Content $stdoutFile -Raw -ErrorAction SilentlyContinue
        $stderr = Get-Content $stderrFile -Raw -ErrorAction SilentlyContinue
        
        $result = @{
            status = "timeout"
            exit_code = -1
            stdout = if ($stdout) { $stdout } else { "" }
            stderr = if ($stderr) { $stderr } else { "" }
            elapsed_s = $TimeoutSec
        }
        
        $result | ConvertTo-Json -Compress
        
        if ($FailOnTimeout) {
            exit 1
        }
        exit 0
    }
    
    $exitCode = $process.ExitCode
    if ($exitCode -eq $null) {
        $exitCode = 0
    }
    
    $stdout = ""
    $stderr = ""
    for ($i = 0; $i -lt 5; $i++) {
        try {
            if (Test-Path $stdoutFile) {
                $stdout = [System.IO.File]::ReadAllText($stdoutFile)
            }
            if (Test-Path $stderrFile) {
                $stderr = [System.IO.File]::ReadAllText($stderrFile)
            }
            break
        } catch {
            Start-Sleep -Milliseconds 200
        }
    }
    
    $result = @{
        status = "completed"
        exit_code = $exitCode
        stdout = if ($stdout) { $stdout } else { "" }
        stderr = if ($stderr) { $stderr } else { "" }
        elapsed_s = $elapsed
    }
    
    $result | ConvertTo-Json -Compress
    
    if ($exitCode -ne 0) {
        exit $exitCode
    }
    exit 0
    
} finally {
    Remove-Item $stdoutFile -ErrorAction SilentlyContinue
    Remove-Item $stderrFile -ErrorAction SilentlyContinue
}
