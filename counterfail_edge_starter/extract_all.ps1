$dirs = Get-ChildItem -Path "data\raw" -Directory
foreach ($d in $dirs) {
    $tar = Join-Path $d.FullName "records.tar.gz"
    if (Test-Path $tar) {
        $sizeMB = [math]::Round((Get-Item $tar).Length / 1MB, 1)
        Write-Host "Extracting $tar ($sizeMB MB)..."
        tar -xzf $tar -C $d.FullName
        Write-Host "Done: $($d.Name)"
    } else {
        Write-Host "No tar in $($d.Name)"
    }
}
Write-Host "`nAll extractions complete."
