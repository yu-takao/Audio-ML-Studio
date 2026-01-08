# S3データ移行スクリプト（PowerShell版）
# 古いバケット（Sandbox）から新しいバケット（本番環境）にデータをコピー

$OLD_BUCKET = "amplify-audioaugmentation-audiomlstoragebucket57fe-m1x276yegzy7"
$NEW_BUCKET = "amplify-d1u3ts7k4s9bu0-ma-audiomlstoragebucket57fe-slhrgtfvefvu"
$OLD_IDENTITY_ID = "f7444a98-d001-7033-2ad2-7c063909f71b"
$NEW_IDENTITY_ID = "0724da88-c0a1-7014-6a64-c03cc9f80c41"
$env:AWS_PROFILE = "trust-tokai-denshi-dev"
$REGION = "ap-northeast-1"

Write-Host "Starting S3 data migration..." -ForegroundColor Cyan
Write-Host "Old bucket: $OLD_BUCKET" -ForegroundColor Gray
Write-Host "New bucket: $NEW_BUCKET" -ForegroundColor Gray
Write-Host ""

# Migrate models (from old identityId to new identityId)
Write-Host "Migrating models..." -ForegroundColor Yellow
aws s3 sync `
  "s3://$OLD_BUCKET/models/$OLD_IDENTITY_ID/" `
  "s3://$NEW_BUCKET/models/$NEW_IDENTITY_ID/" `
  --region $REGION

# Migrate training-data (if exists)
Write-Host "Checking training-data..." -ForegroundColor Yellow
$hasTrainingData = aws s3 ls "s3://$OLD_BUCKET/training-data/$OLD_IDENTITY_ID/" --region $REGION 2>$null
if ($hasTrainingData) {
  Write-Host "  -> Migrating training-data..." -ForegroundColor Yellow
  aws s3 sync `
    "s3://$OLD_BUCKET/training-data/$OLD_IDENTITY_ID/" `
    "s3://$NEW_BUCKET/training-data/$NEW_IDENTITY_ID/" `
    --region $REGION
} else {
  Write-Host "  -> training-data is empty (skipped)" -ForegroundColor Gray
}

# Migrate configs (if exists)
Write-Host "Checking configs..." -ForegroundColor Yellow
$hasConfigs = aws s3 ls "s3://$OLD_BUCKET/configs/$OLD_IDENTITY_ID/" --region $REGION 2>$null
if ($hasConfigs) {
  Write-Host "  -> Migrating configs..." -ForegroundColor Yellow
  aws s3 sync `
    "s3://$OLD_BUCKET/configs/$OLD_IDENTITY_ID/" `
    "s3://$NEW_BUCKET/configs/$NEW_IDENTITY_ID/" `
    --region $REGION
} else {
  Write-Host "  -> configs is empty (skipped)" -ForegroundColor Gray
}

# Migrate public (if exists)
Write-Host "Checking public..." -ForegroundColor Yellow
$hasPublic = aws s3 ls "s3://$OLD_BUCKET/public/" --region $REGION 2>$null
if ($hasPublic) {
  Write-Host "  -> Migrating public..." -ForegroundColor Yellow
  aws s3 sync `
    "s3://$OLD_BUCKET/public/" `
    "s3://$NEW_BUCKET/public/" `
    --region $REGION
} else {
  Write-Host "  -> public is empty (skipped)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Migration completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Migrated data:" -ForegroundColor Cyan
Write-Host "  - models/$OLD_IDENTITY_ID/ -> models/$NEW_IDENTITY_ID/" -ForegroundColor Gray
Write-Host ""
Write-Host "Note: Old bucket was not deleted. Please delete it manually after verification." -ForegroundColor Yellow
