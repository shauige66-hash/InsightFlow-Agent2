# 一条命令启动部署好的服务（PostgreSQL + Redis 模式）。
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_deploy.ps1
#   ... -File scripts\start_deploy.ps1 -Port 8124
#
# 为什么需要脚本：启动需要五个环境变量，少一个就会退回 SQLite 或连不上库，
# 而症状（页面 "Failed to fetch"、记录查不到）都不指向真正的原因。
#
# 停止：Ctrl+C，或
#   Get-NetTCPConnection -LocalPort 8123 -State Listen |
#     ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

param(
    [int]$Port = 8123,
    [string]$DatabaseUrl = 'postgresql+psycopg://insightflow:insightflow@127.0.0.1:5433/insightflow',
    [string]$RedisUrl = 'redis://127.0.0.1:6380/0',
    [string]$Workdir = ''
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Host "找不到虚拟环境：$python" -ForegroundColor Red
    exit 2
}

# 端口被占用时直接说清楚是谁占的，而不是让 uvicorn 抛 Errno 10048
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    $proc = Get-Process -Id $existing.OwningProcess -ErrorAction SilentlyContinue
    Write-Host "端口 $Port 已被 pid $($existing.OwningProcess)（$($proc.ProcessName)）占用。" -ForegroundColor Yellow
    Write-Host "先停掉它：Stop-Process -Id $($existing.OwningProcess) -Force" -ForegroundColor Yellow
    exit 3
}

# 依赖检查：缺了会以很难定位的方式失败
$checks = @(
    @{ Name = 'PostgreSQL(5433)'; Host = '127.0.0.1'; Port = 5433 },
    @{ Name = 'Redis(6380)';      Host = '127.0.0.1'; Port = 6380 },
    @{ Name = 'ModelGate(8000)';  Host = '127.0.0.1'; Port = 8000 }
)
foreach ($check in $checks) {
    $test = Test-NetConnection -ComputerName $check.Host -Port $check.Port -InformationLevel Quiet -WarningAction SilentlyContinue
    if (-not $test) {
        Write-Host "$($check.Name) 未就绪。" -ForegroundColor Yellow
        Write-Host "  PostgreSQL/Redis：docker compose -f deploy/compose.yaml up -d" -ForegroundColor Yellow
        Write-Host "  ModelGate：cd G:\AI-Projects\modelgate-original; docker compose up -d" -ForegroundColor Yellow
        exit 4
    }
}

if (-not $Workdir) {
    $Workdir = Join-Path $repo 'outputs\deploy'
}

$env:INSIGHTFLOW_STORAGE = 'postgres'
$env:INSIGHTFLOW_DATABASE_URL = $DatabaseUrl
$env:INSIGHTFLOW_REDIS_ENABLED = 'true'
$env:INSIGHTFLOW_REDIS_URL = $RedisUrl
$env:INSIGHTFLOW_WORKDIR = $Workdir
$env:PYTHONIOENCODING = 'utf-8'

Write-Host "启动中：http://127.0.0.1:$Port/ui/  （workdir=$Workdir）" -ForegroundColor Cyan
Write-Host "按 Ctrl+C 停止" -ForegroundColor DarkGray

Push-Location $repo
try {
    & $python scripts\serve_api.py --port $Port
}
finally {
    Pop-Location
}
