# Laser Trim AI System - PowerShell Launcher
# Advanced launcher with automatic Python installation option

# Set error action preference
$ErrorActionPreference = "Stop"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Console styling
$Host.UI.RawUI.WindowTitle = "Laser Trim AI System Launcher"
$Host.UI.RawUI.BackgroundColor = "Black"
Clear-Host

function Write-Header {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  Laser Trim AI System - Advanced Launcher  " -ForegroundColor White
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Status {
    param($Message, $Type = "Info")

    $timestamp = Get-Date -Format "HH:mm:ss"

    switch ($Type) {
        "Success" {
            Write-Host "[$timestamp] " -NoNewline -ForegroundColor Gray
            Write-Host "✓ " -NoNewline -ForegroundColor Green
            Write-Host $Message -ForegroundColor Green
        }
        "Error" {
            Write-Host "[$timestamp] " -NoNewline -ForegroundColor Gray
            Write-Host "✗ " -NoNewline -ForegroundColor Red
            Write-Host $Message -ForegroundColor Red
        }
        "Warning" {
            Write-Host "[$timestamp] " -NoNewline -ForegroundColor Gray
            Write-Host "⚠ " -NoNewline -ForegroundColor Yellow
            Write-Host $Message -ForegroundColor Yellow
        }
        default {
            Write-Host "[$timestamp] " -NoNewline -ForegroundColor Gray
            Write-Host "• " -NoNewline -ForegroundColor Cyan
            Write-Host $Message
        }
    }
}

function Test-PythonInstalled {
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]

            if ($major -ge 3 -and $minor -ge 8) {
                Write-Status "Python $($matches[0]) found" "Success"
                return $true
            } else {
                Write-Status "Python $($matches[0]) found but version 3.8+ required" "Warning"
                return $false
            }
        }
    } catch {
        Write-Status "Python not found in PATH" "Warning"
        return $false
    }

    return $false
}

function Install-Python {
    Write-Status "Downloading Python installer..."

    $pythonUrl = "https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe"
    $installerPath = "$env:TEMP\python-installer.exe"

    try {
        # Download Python installer
        Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath -UseBasicParsing
        Write-Status "Download complete" "Success"

        Write-Status "Installing Python (this may take a few minutes)..."

        # Install Python silently
        $installArgs = @(
            "/quiet",
            "InstallAllUsers=0",
            "PrependPath=1",
            "Include_test=0",
            "Include_launcher=1"
        )

        Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait

        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("Path", "User")

        Write-Status "Python installed successfully" "Success"
        return $true

    } catch {
        Write-Status "Failed to install Python: $_" "Error"
        return $false
    } finally {
        # Clean up installer
        if (Test-Path $installerPath) {
            Remove-Item $installerPath -Force
        }
    }
}

function Test-AdminRights {
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Start-LaserTrimSystem {
    Write-Status "Launching Laser Trim AI System..."

    try {
        # Check if launcher exists
        if (-not (Test-Path "launch_app.py")) {
            throw "launch_app.py not found in current directory"
        }

        # Start the Python launcher
        $process = Start-Process python -ArgumentList "launch_app.py" -PassThru -WindowStyle Normal

        if ($process) {
            Write-Status "Application launched successfully!" "Success"
            Write-Host ""
            Write-Host "The launcher window should appear shortly." -ForegroundColor Cyan
            Write-Host "This window will close in 5 seconds..." -ForegroundColor Gray
            Start-Sleep -Seconds 5
            return $true
        }

    } catch {
        Write-Status "Failed to launch: $_" "Error"
        return $false
    }
}

function Show-Menu {
    Write-Host ""
    Write-Host "Select an option:" -ForegroundColor Yellow
    Write-Host "1. Launch Laser Trim AI System" -ForegroundColor White
    Write-Host "2. Install/Update Dependencies Only" -ForegroundColor White
    Write-Host "3. Create Desktop Shortcut" -ForegroundColor White
    Write-Host "4. View Logs" -ForegroundColor White
    Write-Host "5. Exit" -ForegroundColor White
    Write-Host ""

    $choice = Read-Host "Enter your choice (1-5)"
    return $choice
}

function Install-DependenciesOnly {
    Write-Status "Installing dependencies..."

    try {
        if (Test-Path "venv") {
            $venvPython = ".\venv\Scripts\python.exe"
        } else {
            Write-Status "Creating virtual environment..."
            python -m venv venv
            $venvPython = ".\venv\Scripts\python.exe"
        }

        Write-Status "Upgrading pip..."
        & $venvPython -m pip install --upgrade pip

        if (Test-Path "requirements.txt") {
            Write-Status "Installing requirements..."
            & $venvPython -m pip install -r requirements.txt
            Write-Status "Dependencies installed successfully" "Success"
        } else {
            Write-Status "requirements.txt not found" "Warning"
        }

    } catch {
        Write-Status "Failed to install dependencies: $_" "Error"
    }
}

function Create-DesktopShortcut {
    Write-Status "Creating desktop shortcut..."

    try {
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Laser Trim AI System.lnk")
        $Shortcut.TargetPath = "powershell.exe"
        $Shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$ScriptDir\launch_laser_trim.ps1`""
        $Shortcut.WorkingDirectory = $ScriptDir
        $Shortcut.IconLocation = "$ScriptDir\assets\icon.ico"
        $Shortcut.Description = "Launch Laser Trim AI System"
        $Shortcut.Save()

        Write-Status "Desktop shortcut created" "Success"
    } catch {
        Write-Status "Failed to create shortcut: $_" "Error"
    }
}

function View-Logs {
    Write-Status "Opening log file..."

    $logFile = "launcher.log"
    if (Test-Path $logFile) {
        notepad.exe $logFile
    } else {
        Write-Status "No log file found" "Warning"
    }
}

# Main execution
Write-Header

# Check Python installation
Write-Status "Checking Python installation..."
if (-not (Test-PythonInstalled)) {
    Write-Host ""
    Write-Host "Python 3.8+ is required but not found." -ForegroundColor Yellow

    if (Test-AdminRights) {
        $install = Read-Host "Would you like to install Python now? (Y/N)"
        if ($install -eq 'Y' -or $install -eq 'y') {
            if (Install-Python) {
                Write-Status "Please restart this launcher to continue" "Success"
                Read-Host "Press Enter to exit"
                exit
            }
        }
    } else {
        Write-Host "Please install Python 3.8+ from https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    }

    Read-Host "Press Enter to exit"
    exit
}

# Main menu loop
while ($true) {
    $choice = Show-Menu

    switch ($choice) {
        "1" {
            if (Start-LaserTrimSystem) {
                exit
            }
        }
        "2" {
            Install-DependenciesOnly
            Read-Host "Press Enter to continue"
        }
        "3" {
            Create-DesktopShortcut
            Read-Host "Press Enter to continue"
        }
        "4" {
            View-Logs
        }
        "5" {
            Write-Host "Exiting..." -ForegroundColor Gray
            exit
        }
        default {
            Write-Status "Invalid choice. Please try again." "Warning"
        }
    }

    Clear-Host
    Write-Header
}