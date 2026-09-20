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
#>
param(
    [string[]]$Models = @("8232-1", "8340-1"),
    [int]$Days        = 1100,    # how far back; ~3 years covers the resistance-target history
    [string]$Source   = "\\192.168.66.9\BTXData\Departments\System Data\TEST_DATA",
    [string]$Dest     = "C:\dev\ltdata"
)

$stations = "DLTS", "LTS", "LTS3", "Test Station"
$started  = Get-Date
foreach ($station in $stations) {
    $root = Join-Path $Source $station
    if (-not (Test-Path $root)) { Write-Host "skip (not found): $root"; continue }
    foreach ($model in $Models) {
        # "8232-1", "8232-1 BLUE", "8232-1 5V R1.00.01" ... but not "8232-10".
        $dirs = Get-ChildItem $root -Directory -Filter "$model*" -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -eq $model -or $_.Name -like "$model *" }
        foreach ($d in $dirs) {
            $target = Join-Path (Join-Path $Dest $station) $d.Name
            Write-Host ("`n== {0}\{1}" -f $station, $d.Name)
            robocopy $d.FullName $target *.xls *.xlsx /S /Z /XO /MT:16 /MAXAGE:$Days /R:2 /W:5 /NP /NFL /NDL /NJH |
                Select-String -Pattern "Files :|Bytes :|Speed :.*min" | ForEach-Object { "   " + $_.Line.Trim() }
        }
    }
}
$size  = (Get-ChildItem $Dest -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum)
"`nDone in {0:N0} min: {1:N0} files, {2:N2} GB in {3}" -f ((Get-Date) - $started).TotalMinutes, $size.Count, ($size.Sum / 1GB), $Dest
