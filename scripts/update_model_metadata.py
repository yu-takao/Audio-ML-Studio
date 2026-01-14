#!/usr/bin/env python3
"""
既存モデルのメタデータを更新するスクリプト
使用方法: python scripts/update_model_metadata.py --model-name "audio-ml-0724da88-2026-01-13T02-28-20-639Z" --problem-type regression --tolerance 5
"""

import json
import boto3
import sys
import argparse
from io import BytesIO

def update_model_metadata(bucket_name, model_name, problem_type, tolerance):
    """モデルのメタデータを更新"""
    s3 = boto3.client('s3')
    
    # メタデータファイルのパスを構築（ユーザーIDが不明な場合は検索）
    # models/{userId}/{jobName}/output/model_metadata.json の形式
    
    # すべてのmodels/配下のオブジェクトを検索
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix='models/')
    
    metadata_key = None
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            if model_name in key and key.endswith('model_metadata.json'):
                metadata_key = key
                break
        if metadata_key:
            break
    
    if not metadata_key:
        print(f"Error: Model metadata not found for {model_name}", file=sys.stderr)
        return False
    
    print(f"Found metadata at: s3://{bucket_name}/{metadata_key}")
    
    # メタデータをダウンロード
    try:
        response = s3.get_object(Bucket=bucket_name, Key=metadata_key)
        metadata_text = response['Body'].read().decode('utf-8')
        metadata = json.loads(metadata_text)
        
        print("\n=== Current Metadata ===")
        print(f"Problem Type: {metadata.get('problem_type', 'NOT SET')}")
        print(f"Tolerance: {metadata.get('tolerance', 'NOT SET')}")
        
        # メタデータを更新
        metadata['problem_type'] = problem_type
        metadata['tolerance'] = tolerance
        
        print("\n=== Updated Metadata ===")
        print(f"Problem Type: {metadata['problem_type']}")
        print(f"Tolerance: {metadata['tolerance']}")
        
        # メタデータをアップロード
        updated_json = json.dumps(metadata, indent=2, ensure_ascii=False)
        s3.put_object(
            Bucket=bucket_name,
            Key=metadata_key,
            Body=updated_json.encode('utf-8'),
            ContentType='application/json'
        )
        
        print(f"\n[SUCCESS] Updated metadata at: s3://{bucket_name}/{metadata_key}")
        return True
        
    except Exception as e:
        print(f"Error updating metadata: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='Update model metadata in S3')
    parser.add_argument('--model-name', required=True, help='Model name (e.g., audio-ml-0724da88-2026-01-13T02-28-20-639Z)')
    parser.add_argument('--bucket', default='amplify-d1u3ts7k4s9bu0-ma-audiomlstoragebucket57fe-slhrgtfvefvu', help='S3 bucket name')
    parser.add_argument('--problem-type', required=True, choices=['classification', 'regression'], help='Problem type')
    parser.add_argument('--tolerance', type=float, required=True, help='Tolerance value (for regression)')
    
    args = parser.parse_args()
    
    print(f"Updating model: {args.model_name}")
    print(f"Bucket: {args.bucket}")
    print(f"Problem Type: {args.problem_type}")
    print(f"Tolerance: {args.tolerance}\n")
    
    success = update_model_metadata(args.bucket, args.model_name, args.problem_type, args.tolerance)
    
    if success:
        print("\n[SUCCESS] Update completed successfully")
        sys.exit(0)
    else:
        print("\n[FAILED] Update failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
