<#
Pull a few models' files home so feature work can continue off the VPN.

The full rebuild cannot run from home (54 ms per round trip x 284,000 files),
but development does not need every file -- it needs DEEP data on a few
models. This copies every folder whose name starts with a model number, from
all four station folders, keeping the DLTS / LTS / LTS3 / Test Station
structure (the app identifies laser 3 by the LTS3 folder in the path).

READ-ONLY toward the share: robocopy copies FROM it, and none of /MIR /PURGE
/MOV is used, so nothing on the share can be changed or deleted. Restartable:
run it again and it only fetches what is missing.

    .\scripts\pull_model_slice.ps1
    .\scripts\pull_model_slice.ps1 -Models "8232-1","8340-1","6607" -Days 1100

A model can change lasers: 8232-1 ran on laser 1 (DLTS) until 2022-07 and on
laser 2 (LTS) since, so the default 3-year window correctly copies NOTHING
from DLTS\8232-1 ("0 copied, 4178 skipped" is right, not a fault). Laser 1 is
the only one that records cut length and predicted-vs-actual correction at
every position, so that old history is what the cut-length model will want:

    .\scripts\pull_model_slice.ps1 -Models "8232-1" -Stations "DLTS" -Days 5000
#>
param(
    [string[]]$Models = @("8232-1", "8340-1"),
    [int]$Days        = 1100,    # how far back; ~3 years covers the resistance-target history
    [string[]]$Stations = @("DLTS", "LTS", "LTS3", "Test Station"),
    [string]$Source   = "\\192.168.66.9\BTXData\Departments\System Data\TEST_DATA",
    [string]$Dest     = "C:\dev\ltdata"
)

$started  = Get-Date
$failed   = 0
# robocopy reads /MAXAGE values of 1900 or more as a DATE (YYYYMMDD), not as
# days -- so "-Days 5000" was rejected as an invalid date and nothing copied
# (2026-09-20). Always hand it a real date; then any number of days is safe.
$cutoff   = (Get-Date).AddDays(-$Days).ToString("yyyyMMdd")
Write-Host "copying files modified on or after $cutoff"
foreach ($station in $Stations) {
    $root = Join-Path $Source $station
    if (-not (Test-Path $root)) { Write-Host "skip (not found): $root"; continue }
    foreach ($model in $Models) {
        # "8232-1", "8232-1 BLUE", "8232-1 5V R1.00.01" ... but not "8232-10".
        $dirs = Get-ChildItem $root -Directory -Filter "$model*" -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -eq $model -or $_.Name -like "$model *" }
        foreach ($d in $dirs) {
            $target = Join-Path (Join-Path $Dest $station) $d.Name
            Write-Host ("`n== {0}\{1}" -f $station, $d.Name)
            # Capture first, THEN filter: piping robocopy straight into the filter
            # loses its exit code, and the old filter threw the ERROR lines away,
            # so a run that copied nothing still printed a cheerful "Done".
            $out  = robocopy $d.FullName $target *.xls *.xlsx /S /Z /XO /MT:16 /MAXAGE:$cutoff /R:2 /W:5 /NP /NFL /NDL /NJH
            $code = $LASTEXITCODE
            $out | Select-String -Pattern "^\s*(Files|Bytes) :\s+\d|Speed :.*min|ERROR" |
                ForEach-Object { "   " + $_.Line.Trim() }
            if ($code -ge 8) {       # robocopy: 0-7 are flavours of success, 8+ is failure
                $failed++
                Write-Host "   ** ROBOCOPY FAILED (exit code $code) -- this folder was NOT copied" -ForegroundColor Red
                $out | Select-Object -Last 6 | ForEach-Object { "      " + $_ }
            }
        }
    }
}
$size  = (Get-ChildItem $Dest -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum)
$verdict = if ($failed) { "FINISHED WITH $failed FAILED FOLDER(S)" } else { "Done" }
"`n{4} in {0:N0} min. {3} now holds {1:N0} files, {2:N2} GB in total (all runs, not just this one)." -f ((Get-Date) - $started).TotalMinutes, $size.Count, ($size.Sum / 1GB), $Dest, $verdict
if ($failed) { exit 1 }
