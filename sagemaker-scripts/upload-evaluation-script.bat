@echo off
REM SageMaker評価スクリプトをS3にアップロード（Windows用）

REM 使用方法:
REM upload-evaluation-script.bat <bucket-name> <region>

set BUCKET_NAME=%1
set REGION=%2

if "%REGION%"=="" set REGION=ap-northeast-1

if "%BUCKET_NAME%"=="" (
  echo 使用方法: upload-evaluation-script.bat ^<bucket-name^> ^<region^>
  exit /b 1
)

echo 📦 評価スクリプトをS3にアップロード中...
echo バケット: %BUCKET_NAME%
echo リージョン: %REGION%

REM 評価スクリプトをアップロード
aws s3 cp evaluate.py s3://%BUCKET_NAME%/public/scripts/evaluation/evaluate.py --region %REGION%

REM requirements.txtも一緒にアップロード
aws s3 cp analyze_requirements.txt s3://%BUCKET_NAME%/public/scripts/evaluation/requirements.txt --region %REGION%

echo ✅ アップロード完了！
echo.
echo 次のステップ：
echo 1. フロントエンドで評価機能を使用
echo 2. S3パス: s3://%BUCKET_NAME%/public/scripts/evaluation/

pause

