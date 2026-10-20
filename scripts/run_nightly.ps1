# 每晚跑一次真实模型回归，把结果写成一份日志。
#
# 由 Windows 计划任务调用；也可以手动跑：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_nightly.ps1
#
# 需要本机 ModelGate 在跑（http://127.0.0.1:8000）。它不在时回归会以凭据/网络错误
# 失败——那本身也是要记录的信息，所以脚本照样写日志并保留退出码。
#
# 为什么用 Start-Process 而不是管道：python 会把模型加载进度写到 stderr，而
# PowerShell 在 `$ErrorActionPreference='Stop'` 下会把原生命令的 stderr 当成错误，
# 直接掐断整条管道（第一版就是这么挂的，日志只剩两行）。

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
# 日志默认落在系统临时目录；需要长期保留时用环境变量指到别处。
$logDir = if ($env:INSIGHTFLOW_NIGHTLY_LOG_DIR) {
    $env:INSIGHTFLOW_NIGHTLY_LOG_DIR
} else {
    Join-Path $env:TEMP 'insightflow-nightly'
}
$stamp = Get-Date -Format 'yyyy-MM-dd_HHmmss'

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "nightly-$stamp.out.log"
$stderr = Join-Path $logDir "nightly-$stamp.err.log"

if (-not (Test-Path $python)) {
    "找不到虚拟环境：$python" | Set-Content -Path $stdout -Encoding UTF8
    exit 2
}

$started = Get-Date
# 输出被重定向到文件时，Python 会用系统默认编码（中文 Windows 上是 GBK）写日志，
# 读回来就是一堆乱码。显式要求 UTF-8，让日志可读。
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$proc = Start-Process -FilePath $python `
    -ArgumentList 'scripts\real_model_regression.py', '--include-retail' `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru -Wait -NoNewWindow

$elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)

# 合并成一份可读日志：先结果，再 stderr 里的告警（模型加载、PDF 浏览器等）
$combined = Join-Path $logDir "nightly-$stamp.log"
"=== $stamp 开始（耗时 $elapsed 秒）===" | Set-Content -Path $combined -Encoding UTF8
if (Test-Path $stdout) { Get-Content $stdout -Encoding UTF8 | Add-Content -Path $combined -Encoding UTF8 }
"=== 退出码 $($proc.ExitCode) ===" | Add-Content -Path $combined -Encoding UTF8
if ((Test-Path $stderr) -and (Get-Item $stderr).Length -gt 0) {
    "--- stderr ---" | Add-Content -Path $combined -Encoding UTF8
    Get-Content $stderr -Encoding UTF8 | Add-Content -Path $combined -Encoding UTF8
}

Remove-Item $stdout, $stderr -ErrorAction SilentlyContinue
Get-Content $combined -Encoding UTF8 | Write-Output
exit $proc.ExitCode
