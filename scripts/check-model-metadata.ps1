# モデルのメタデータを確認するスクリプト
# 使用方法: .\scripts\check-model-metadata.ps1 -ModelName "audio-ml-0724da88-2026-01-13T02-28-20-639Z" -UserId "your-user-id"

param(
    [Parameter(Mandatory=$true)]
    [string]$ModelName,
    
    [Parameter(Mandatory=$true)]
    [string]$UserId
)

# Amplify outputsを読み込む
$amplifyOutputsPath = "amplify_outputs.json"
if (-not (Test-Path $amplifyOutputsPath)) {
    Write-Host "Error: amplify_outputs.json not found" -ForegroundColor Red
    exit 1
}

$amplifyOutputs = Get-Content $amplifyOutputsPath | ConvertFrom-Json
$bucketName = $amplifyOutputs.custom.audioMLStorageBucketName

if (-not $bucketName) {
    Write-Host "Error: Bucket name not found in amplify_outputs.json" -ForegroundColor Red
    exit 1
}

Write-Host "Checking model metadata for: $ModelName" -ForegroundColor Cyan
Write-Host "Bucket: $bucketName" -ForegroundColor Cyan
Write-Host "User ID: $UserId" -ForegroundColor Cyan
Write-Host ""

# S3パスを構築
$metadataPath = "models/$UserId/$ModelName/output/model_metadata.json"
Write-Host "Metadata path: $metadataPath" -ForegroundColor Yellow

# AWS CLIでメタデータを取得
try {
    $metadataJson = aws s3 cp "s3://$bucketName/$metadataPath" - 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to download metadata" -ForegroundColor Red
        Write-Host $metadataJson -ForegroundColor Red
        exit 1
    }
    
    $metadata = $metadataJson | ConvertFrom-Json
    
    Write-Host "=== Model Metadata ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Problem Type: $($metadata.problem_type)" -ForegroundColor $(if ($metadata.problem_type -eq 'regression') { 'Green' } else { 'Yellow' })
    Write-Host "Tolerance: $($metadata.tolerance)" -ForegroundColor Cyan
    Write-Host "Target Field: $($metadata.target_field)" -ForegroundColor Cyan
    Write-Host "Classes: $($metadata.classes -join ', ')" -ForegroundColor Cyan
    Write-Host ""
    
    if ($metadata.field_labels) {
        Write-Host "Field Labels:" -ForegroundColor Cyan
        $metadata.field_labels | ForEach-Object {
            Write-Host "  Index $($_.index): $($_.label)" -ForegroundColor White
        }
        Write-Host ""
    }
    
    # 回帰問題として作成されているか確認
    if ($metadata.problem_type -eq 'regression') {
        Write-Host "✓ This model was created as a REGRESSION problem" -ForegroundColor Green
    } else {
        Write-Host "✗ This model was created as a CLASSIFICATION problem" -ForegroundColor Yellow
        Write-Host "  (Not a regression model)" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "Full metadata:" -ForegroundColor Cyan
    $metadata | ConvertTo-Json -Depth 10 | Write-Host
    
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
