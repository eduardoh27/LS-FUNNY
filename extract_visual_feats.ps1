$FE        = "FeatureExtraction.exe"
$InputDir  = "dataset\videos"
$OutputDir = "features\open_face_raw"

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$videos = Get-ChildItem -Path $InputDir -Filter *.mp4 -File
if ($videos.Count -eq 0) {
    Write-Error "⚠️ No *.mp4 files found in $InputDir"
    exit 1
}

$MaxJobs = 8
$Jobs    = @()

foreach ($video in $videos) {
    $fullPath = $video.FullName
    $baseName = $video.BaseName

    $Jobs += Start-Job -ScriptBlock {
        param($exe, $vid, $out)
        Write-Output "[$([DateTime]::Now.ToString('HH:mm:ss'))] Processing $vid…"
        & $exe `
            -f        $vid `
            -out_dir  $out `
            -2Dfp `
            -3Dfp `
            -pdmparams `
            -pose `
            -aus `
            -gaze *>&1
    } -ArgumentList $FE, $fullPath, $OutputDir

    if ($Jobs.Count -ge $MaxJobs) {
        $done = Wait-Job -Job $Jobs -Any
        Receive-Job -Job $done | Write-Host
        $Jobs = $Jobs | Where-Object { $_.State -eq 'Running' }
    }
}

if ($Jobs) {
    Wait-Job -Job $Jobs
    Receive-Job -Job $Jobs | Write-Host
}

Write-Host "`n✅ All videos processed. Results in: $OutputDir"
