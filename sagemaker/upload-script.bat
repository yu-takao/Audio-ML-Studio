@echo off
REM SageMaker トレーニングスクリプトを S3 にアップロード
REM 使用法: upload-script.bat <bucket-name> [region]

setlocal

set BUCKET=%1
set REGION=%2

if "%BUCKET%"=="" (
    echo Usage: upload-script.bat ^<bucket-name^> [region]
    echo Example: upload-script.bat my-training-bucket ap-northeast-1
    exit /b 1
)

if "%REGION%"=="" set REGION=ap-northeast-1

set SCRIPT_DIR=%~dp0
set OUTPUT_FILE=audio-ml-training.tar.gz
set S3_PATH=s3://%BUCKET%/scripts/%OUTPUT_FILE%

echo ========================================
echo Uploading SageMaker Training Script
echo ========================================
echo Bucket: %BUCKET%
echo Region: %REGION%
echo S3 Path: %S3_PATH%
echo ========================================

REM スクリプトをtar.gzに圧縮（tarコマンドが使用可能な場合）
echo Creating archive...
cd /d "%SCRIPT_DIR%"
tar -czvf "%OUTPUT_FILE%" train.py

if %ERRORLEVEL% neq 0 (
    echo Error: tar command failed. Please install tar or use WSL.
    exit /b 1
)

REM S3にアップロード
echo Uploading to S3...
aws s3 cp "%OUTPUT_FILE%" "%S3_PATH%" --region %REGION%

if %ERRORLEVEL% neq 0 (
    echo Error: S3 upload failed.
    exit /b 1
)

REM クリーンアップ
del "%OUTPUT_FILE%"

echo ========================================
echo Done!
echo.
echo Script uploaded to: %S3_PATH%
echo.
echo Next steps:
echo 1. Create a SageMaker IAM role with S3 access
echo 2. Set SAGEMAKER_ROLE_ARN in your Amplify environment
echo ========================================

endlocal



