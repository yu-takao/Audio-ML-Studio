#!/bin/bash

# SageMaker トレーニングスクリプトを S3 にアップロード
# 使用法: ./upload-script.sh <bucket-name> [region]

set -e

BUCKET=${1:-""}
REGION=${2:-"ap-northeast-1"}

if [ -z "$BUCKET" ]; then
    echo "Usage: ./upload-script.sh <bucket-name> [region]"
    echo "Example: ./upload-script.sh my-training-bucket ap-northeast-1"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="audio-ml-training.tar.gz"
S3_PATH="s3://${BUCKET}/scripts/${OUTPUT_FILE}"

echo "========================================"
echo "Uploading SageMaker Training Script"
echo "========================================"
echo "Bucket: ${BUCKET}"
echo "Region: ${REGION}"
echo "S3 Path: ${S3_PATH}"
echo "========================================"

# スクリプトをtar.gzに圧縮
echo "Creating archive..."
cd "$SCRIPT_DIR"
tar -czvf "$OUTPUT_FILE" train.py

# S3にアップロード
echo "Uploading to S3..."
aws s3 cp "$OUTPUT_FILE" "$S3_PATH" --region "$REGION"

# クリーンアップ
rm -f "$OUTPUT_FILE"

echo "========================================"
echo "Done!"
echo ""
echo "Script uploaded to: ${S3_PATH}"
echo ""
echo "Next steps:"
echo "1. Create a SageMaker IAM role with S3 access"
echo "2. Set SAGEMAKER_ROLE_ARN in your Amplify environment"
echo "========================================"



