#!/bin/bash

# S3データ移行スクリプト
# 古いidentityIdのデータを新しいidentityIdにコピー

BUCKET_NAME="amplify-audioaugmentation-audiomlstoragebucket57fe-zswzbkwst4vp"
OLD_IDENTITY_ID="f7444a98-d001-7033-2ad2-7c063909f71b"
NEW_IDENTITY_ID="0724da88-c0a1-7014-6a64-c03cc9f80c41"
PROFILE="trust-tokai-denshi-dev"
REGION="ap-northeast-1"

echo "🔄 S3データ移行を開始します..."

# training-data の移行
echo "📁 training-data を移行中..."
aws s3 sync \
  "s3://${BUCKET_NAME}/training-data/${OLD_IDENTITY_ID}/" \
  "s3://${BUCKET_NAME}/training-data/${NEW_IDENTITY_ID}/" \
  --profile ${PROFILE} \
  --region ${REGION}

# configs の移行
echo "📁 configs を移行中..."
aws s3 sync \
  "s3://${BUCKET_NAME}/configs/${OLD_IDENTITY_ID}/" \
  "s3://${BUCKET_NAME}/configs/${NEW_IDENTITY_ID}/" \
  --profile ${PROFILE} \
  --region ${REGION}

# evaluation/temp の移行（あれば）
echo "📁 evaluation/temp を移行中..."
aws s3 sync \
  "s3://${BUCKET_NAME}/evaluation/temp/${OLD_IDENTITY_ID}/" \
  "s3://${BUCKET_NAME}/evaluation/temp/${NEW_IDENTITY_ID}/" \
  --profile ${PROFILE} \
  --region ${REGION} \
  2>/dev/null || true

# evaluation/results の移行（あれば）
echo "📁 evaluation/results を移行中..."
aws s3 sync \
  "s3://${BUCKET_NAME}/evaluation/results/${OLD_IDENTITY_ID}/" \
  "s3://${BUCKET_NAME}/evaluation/results/${NEW_IDENTITY_ID}/" \
  --profile ${PROFILE} \
  --region ${REGION} \
  2>/dev/null || true

echo "✅ データ移行が完了しました！"
