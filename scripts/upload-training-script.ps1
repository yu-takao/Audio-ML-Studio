# SageMaker訓練スクリプトをS3にアップロードするスクリプト
# 使用方法: .\scripts\upload-training-script.ps1

$ErrorActionPreference = "Stop"

# バケット名を取得（amplify_outputs.jsonから）
$amplifyOutputs = Get-Content -Path "amplify_outputs.json" | ConvertFrom-Json
$bucketName = $amplifyOutputs.storage.bucket_name

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SageMaker Training Script Uploader" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Target Bucket: $bucketName" -ForegroundColor Green

# 一時ディレクトリを作成
$tempDir = New-Item -ItemType Directory -Force -Path "$env:TEMP\sagemaker-scripts-$(Get-Date -Format 'yyyyMMddHHmmss')"
Write-Host "Temp directory: $tempDir" -ForegroundColor Gray

# スクリプトファイルをコピー
Copy-Item -Path "sagemaker-scripts\train.py" -Destination $tempDir
Copy-Item -Path "sagemaker-scripts\requirements.txt" -Destination $tempDir

# tar.gzを作成
$tarFile = "$tempDir\audio-ml-training.tar.gz"

# Pythonを使ってtar.gzを作成（Windowsでも動作）
$pythonScript = @"
import tarfile
import os
import sys

temp_dir = sys.argv[1]
output_file = sys.argv[2]

with tarfile.open(output_file, 'w:gz') as tar:
    for file in ['train.py', 'requirements.txt']:
        file_path = os.path.join(temp_dir, file)
        if os.path.exists(file_path):
            tar.add(file_path, arcname=file)
            print(f'Added: {file}')

print(f'Created: {output_file}')
"@

$pythonScriptPath = "$tempDir\create_tar.py"
$pythonScript | Out-File -FilePath $pythonScriptPath -Encoding UTF8

Write-Host "Creating tar.gz archive..." -ForegroundColor Yellow
python $pythonScriptPath $tempDir $tarFile

if (!(Test-Path $tarFile)) {
    Write-Host "Error: Failed to create tar.gz file" -ForegroundColor Red
    exit 1
}

Write-Host "Archive created: $tarFile" -ForegroundColor Green

# S3にアップロード（バージョン付きキーでキャッシュ回避）
$s3Key = "public/scripts/audio-ml-training-v2.tar.gz"
Write-Host "Uploading to S3..." -ForegroundColor Yellow
Write-Host "  s3://$bucketName/$s3Key" -ForegroundColor Gray

aws s3 cp $tarFile "s3://$bucketName/$s3Key" --profile trust-tokai-denshi-dev

if ($LASTEXITCODE -eq 0) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Upload completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Script location: s3://$bucketName/$s3Key" -ForegroundColor White
} else {
    Write-Host "Error: Upload failed" -ForegroundColor Red
    exit 1
}

# 一時ファイルを削除
Remove-Item -Path $tempDir -Recurse -Force
Write-Host "Cleaned up temporary files" -ForegroundColor Gray



