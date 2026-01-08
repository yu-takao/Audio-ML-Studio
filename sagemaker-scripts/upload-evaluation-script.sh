#!/bin/bash
# SageMaker評価スクリプトをS3にアップロード

# 使用方法:
# ./upload-evaluation-script.sh <bucket-name> <region>

BUCKET_NAME=$1
REGION=${2:-ap-northeast-1}

if [ -z "$BUCKET_NAME" ]; then
  echo "使用方法: ./upload-evaluation-script.sh <bucket-name> <region>"
  exit 1
fi

echo "📦 評価スクリプトをS3にアップロード中..."
echo "バケット: $BUCKET_NAME"
echo "リージョン: $REGION"

# 評価スクリプトをアップロード（tarは不要、直接アップロード）
aws s3 cp evaluate.py s3://${BUCKET_NAME}/public/scripts/evaluation/evaluate.py --region ${REGION}

# requirements.txtも一緒にアップロード（SageMakerで依存関係をインストール）
aws s3 cp analyze_requirements.txt s3://${BUCKET_NAME}/public/scripts/evaluation/requirements.txt --region ${REGION}

echo "✅ アップロード完了！"
echo ""
echo "次のステップ："
echo "1. フロントエンドで評価機能を使用"
echo "2. S3パス: s3://${BUCKET_NAME}/public/scripts/evaluation/"

