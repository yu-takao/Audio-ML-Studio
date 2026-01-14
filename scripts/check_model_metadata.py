#!/usr/bin/env python3
"""
モデルのメタデータを確認するスクリプト
使用方法: python scripts/check_model_metadata.py --model-name "audio-ml-0724da88-2026-01-13T02-28-20-639Z"
"""

import json
import boto3
import sys
import argparse
from pathlib import Path

def find_model_metadata(bucket_name, model_name):
    """S3バケット内でモデルのメタデータを検索"""
    s3 = boto3.client('s3')
    
    # すべてのmodels/配下のオブジェクトを検索（public/も含む）
    prefixes = ['models/', 'public/models/']
    found_keys = []
    
    print("Searching for model metadata...")
    
    for prefix in prefixes:
        try:
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    if model_name in key:
                        print(f"Found matching path: {key}")
                        if key.endswith('model_metadata.json'):
                            found_keys.append(key)
        except Exception as e:
            print(f"Warning: Could not search {prefix}: {e}")
    
    if found_keys:
        print(f"\nFound {len(found_keys)} metadata file(s):")
        for key in found_keys:
            print(f"  - {key}")
        return found_keys[0]  # 最初のものを返す
    
    return None

def download_and_check_metadata(bucket_name, metadata_key):
    """メタデータをダウンロードして確認"""
    s3 = boto3.client('s3')
    
    try:
        response = s3.get_object(Bucket=bucket_name, Key=metadata_key)
        metadata_text = response['Body'].read().decode('utf-8')
        metadata = json.loads(metadata_text)
        
        print("\n=== Model Metadata ===")
        print(f"Problem Type: {metadata.get('problem_type', 'classification (default)')}")
        print(f"Tolerance: {metadata.get('tolerance', 0)}")
        print(f"Target Field: {metadata.get('target_field', 'N/A')}")
        
        if 'classes' in metadata:
            print(f"Classes: {', '.join(metadata['classes'])}")
            print(f"Number of classes: {len(metadata['classes'])}")
        
        if 'field_labels' in metadata:
            print("\nField Labels:")
            for label in metadata['field_labels']:
                print(f"  Index {label['index']}: {label['label']}")
        
        # 回帰問題として作成されているか確認
        print("\n=== Analysis ===")
        if metadata.get('problem_type') == 'regression':
            print("✓ This model was created as a REGRESSION problem")
        else:
            print("✗ This model was created as a CLASSIFICATION problem")
            print("  (problem_type field is missing or set to 'classification')")
        
        print("\nFull metadata:")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        
        return metadata
        
    except Exception as e:
        print(f"Error downloading metadata: {e}", file=sys.stderr)
        return None

def list_all_models(bucket_name):
    """すべてのモデルをリストアップ"""
    s3 = boto3.client('s3')
    
    models = []
    prefixes = ['models/', 'public/models/']  # 両方のパスを検索
    
    for prefix in prefixes:
        try:
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('model_metadata.json'):
                        # models/{userId}/{jobName}/output/model_metadata.json の形式
                        parts = key.split('/')
                        if len(parts) >= 4:
                            job_name = parts[-2] if parts[-2] == 'output' else parts[-3] if len(parts) >= 3 else 'unknown'
                            # jobNameを取得（outputの前の部分）
                            for i, part in enumerate(parts):
                                if part == 'output' and i > 0:
                                    job_name = parts[i-1]
                                    break
                            models.append((job_name, key))
        except Exception as e:
            print(f"Warning: Could not list {prefix}: {e}")
    
    return models

def main():
    parser = argparse.ArgumentParser(description='Check model metadata from S3')
    parser.add_argument('--model-name', help='Model name (e.g., audio-ml-0724da88-2026-01-13T02-28-20-639Z)')
    parser.add_argument('--bucket', default='amplify-audioaugmentation-audiomlstoragebucket57fe-auidriqi877t', help='S3 bucket name')
    parser.add_argument('--list-all', action='store_true', help='List all available models')
    
    args = parser.parse_args()
    
    if args.list_all:
        print("Listing all available models...\n")
        models = list_all_models(args.bucket)
        if models:
            print(f"Found {len(models)} model(s):\n")
            for job_name, key in sorted(models):
                print(f"  {job_name}")
                print(f"    Path: {key}\n")
        else:
            print("No models found.")
        return
    
    if not args.model_name:
        parser.error("--model-name is required unless --list-all is specified")
    
    print(f"Searching for model: {args.model_name}")
    print(f"Bucket: {args.bucket}\n")
    
    # メタデータファイルを検索
    metadata_key = find_model_metadata(args.bucket, args.model_name)
    
    if not metadata_key:
        print(f"\nError: Model metadata not found for {args.model_name}", file=sys.stderr)
        print("\nAvailable models:")
        models = list_all_models(args.bucket)
        matching = [m for m in models if args.model_name.lower() in m[0].lower()]
        if matching:
            print(f"\nFound {len(matching)} similar model(s):")
            for job_name, key in matching:
                print(f"  {job_name}")
        else:
            print("  (No similar models found)")
            if models:
                print(f"\nAll {len(models)} available models:")
                for job_name, key in models[:10]:  # 最初の10個だけ表示
                    print(f"  {job_name}")
                if len(models) > 10:
                    print(f"  ... and {len(models) - 10} more")
        sys.exit(1)
    
    # メタデータをダウンロードして確認
    download_and_check_metadata(args.bucket, metadata_key)

if __name__ == '__main__':
    main()
