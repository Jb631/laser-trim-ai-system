# Laser Trim Analyzer Launcher
Write-Host "Starting Laser Trim Analyzer..." -ForegroundColor Green

# Change to src directory
Set-Location -Path (Join-Path $PSScriptRoot "src")

# Run the application
python -m laser_trim_analyzer

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host "Press any key to continue..." -ForegroundColor Red
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
} 