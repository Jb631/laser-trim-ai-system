# Build the Laser Trim Analyzer as a Windows folder with LaserTrimAnalyzer.exe -- one command.
#
#   From the repo folder, on the work laptop:
#       powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
#       powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1 -Console     (debug build)
#
# What it does, in order:
#   1. refuses anywhere but Windows, and without the app's own .venv (run_v6.bat makes it);
#   2. refuses if dist\LaserTrimAnalyzer already holds a "data" folder (a build replaces that
#      whole folder, and this script never deletes data);
#   3. installs PyInstaller into .venv if it is missing -- THIS DOWNLOADS FROM PyPI -- and stops
#      if the installed PyInstaller does not support .venv's Python;
#   4. builds from packaging\laser_trim_v6.spec into dist\LaserTrimAnalyzer\ (a folder, not a
#      single file). The spec first runs the app's self-check from source: it must pass;
#   5. writes build_info.txt (which commit, when) and "READ ME FIRST.txt" into that folder;
#   6. runs dist\LaserTrimAnalyzer\LaserTrimAnalyzer.exe --check -- no window; every part a
#      packaged build can be missing is tried -- and prints its lines (the .exe also writes them
#      to check_result.txt beside itself, because a windowed .exe has no console);
#   7. prints what to do next.
# It copies NO data. The build's output never contains a "data" folder.
#
# -Console builds a version with a console window, where start-up errors are printed.
#
# WRITTEN ON A MAC AND NEVER RUN (2026-09-30): no PowerShell there. Kept to plain commands for
# Windows PowerShell 5.1, plain ASCII, and everything that could be said in Python is in
# packaging\build_support.py, which IS tested. If a line here is wrong, the message PowerShell
# prints names the line.

param([switch]$Console)

if ($env:OS -ne 'Windows_NT') {
    Write-Host 'This script builds a Windows .exe and only runs on Windows (PyInstaller cannot build for another system).'
    exit 1
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
    Write-Host 'There is no .venv in this folder. Run run_v6.bat once first: it creates .venv with the pinned libraries.'
    exit 1
}

$dist = Join-Path $root 'dist\LaserTrimAnalyzer'
$distData = Join-Path $dist 'data'
if (Test-Path -LiteralPath $distData) {
    Write-Host "STOPPED: $distData exists."
    Write-Host 'A build replaces the whole dist\LaserTrimAnalyzer folder, and this script never deletes data.'
    Write-Host 'Move that "data" folder somewhere else (or delete it yourself), then run this again.'
    exit 1
}

# ---- PyInstaller ----
& $py packaging\build_support.py check-pyinstaller
if ($LASTEXITCODE -eq 3) {
    Write-Host ''
    Write-Host 'Installing PyInstaller into .venv now. THIS DOWNLOADS FROM PyPI (python -m pip install pyinstaller).'
    & $py -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) {
        Write-Host ''
        Write-Host 'STOPPED: pip could not install PyInstaller. Either there is no way out to PyPI from here'
        Write-Host '(proxy?), or no PyInstaller release supports the Python in .venv. The message above says which.'
        exit 1
    }
    & $py packaging\build_support.py check-pyinstaller
}
if ($LASTEXITCODE -ne 0) {
    Write-Host 'STOPPED: see the line above. Nothing was built.'
    exit 1
}

# ---- build ----
$oldConsole = $env:LTA_CONSOLE
if ($Console) { $env:LTA_CONSOLE = '1' } else { $env:LTA_CONSOLE = '0' }
Write-Host ''
if ($Console) { Write-Host 'Building the CONSOLE (debug) version ...' } else { Write-Host 'Building ...  (a few minutes)' }
& $py -m PyInstaller packaging\laser_trim_v6.spec --noconfirm --clean --distpath dist --workpath build
$buildExit = $LASTEXITCODE
$env:LTA_CONSOLE = $oldConsole
$exe = Join-Path $dist 'LaserTrimAnalyzer.exe'
if (($buildExit -ne 0) -or (-not (Test-Path -LiteralPath $exe))) {
    Write-Host ''
    Write-Host 'STOPPED: the build failed. The reason is in the lines above (the last ones first).'
    exit 1
}

# ---- stamp it, and write the READ ME ----
& $py packaging\build_support.py stamp $dist
if ($LASTEXITCODE -ne 0) {
    Write-Host 'STOPPED: build_info.txt and the READ ME could not be written (see above).'
    exit 1
}

# ---- the built app checks itself ----
Write-Host ''
Write-Host 'Running LaserTrimAnalyzer.exe --check (no window) ...'
$result = Join-Path $dist 'check_result.txt'
$proc = Start-Process -FilePath $exe -ArgumentList '--check' -PassThru
$null = $proc.Handle
$finished = $proc.WaitForExit(600000)
if (-not $finished) {
    $proc.Kill()
    Write-Host 'STOPPED: the self-check did not finish within 10 minutes and was ended.'
    exit 1
}
$checkExit = $proc.ExitCode
if (-not (Test-Path -LiteralPath $result)) {
    Write-Host 'STOPPED: the built .exe did not write check_result.txt -- it did not get as far as starting.'
    Write-Host 'Build the console version to see why:  scripts\build_exe.ps1 -Console  and run the .exe from a terminal.'
    exit 1
}
$lines = @(Get-Content -LiteralPath $result)
foreach ($line in $lines) { Write-Host "  $line" }
$lastLine = ''
if ($lines.Count -gt 0) { $lastLine = [string]$lines[$lines.Count - 1] }
if (($checkExit -ne 0) -or ($lastLine -ne 'PACKAGED BUILD OK')) {
    Write-Host ''
    Write-Host 'THE BUILD IS INCOMPLETE -- do not hand it over. The FAIL lines above name what is missing;'
    Write-Host 'send them back (they are also in dist\LaserTrimAnalyzer\check_result.txt).'
    exit 1
}

if (Test-Path -LiteralPath $distData) {
    Write-Host ''
    Write-Host "STOPPED: $distData was created by the build or the check. It must not be in what you hand over."
    exit 1
}

# ---- next ----
Write-Host ''
Write-Host 'BUILT AND CHECKED:  dist\LaserTrimAnalyzer\   (which build: build_info.txt in that folder)'
Write-Host ''
Write-Host 'NEXT'
Write-Host '  1. Zip the folder dist\LaserTrimAnalyzer and hand it over. Zip it BEFORE starting the app from'
Write-Host '     there: a start creates a "data" folder, and the zip must not contain one.'
Write-Host '  2. On her PC: unzip it, then put a "data" folder (analysis.db + config.yaml) beside'
Write-Host '     LaserTrimAnalyzer.exe. Copy it with your own app CLOSED, so the database file is complete.'
Write-Host '     The app only ever opens the "data" folder beside it; READ ME FIRST.txt in the folder says so.'
Write-Host '  3. A later update: build again, zip, and on her PC delete everything in her folder EXCEPT "data",'
Write-Host '     then move the new files in. Her data is never touched; the app upgrades the database itself.'
Write-Host '  To see which database a copy would open, without opening it:  LaserTrimAnalyzer.exe --check'
Write-Host '  then read the "data" line of check_result.txt.'
exit 0
