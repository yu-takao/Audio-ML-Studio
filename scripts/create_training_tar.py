#!/usr/bin/env python3
"""訓練スクリプトのtar.gzを作成"""
import tarfile
import os
import sys

def create_tar():
    # ファイルパス
    train_py = 'sagemaker/train.py'
    requirements_txt = 'sagemaker/requirements.txt'
    output_tar = 'audio-ml-training-v2.tar.gz'
    
    # ファイルの存在確認
    if not os.path.exists(train_py):
        print(f"Error: {train_py} not found", file=sys.stderr)
        return False
    
    if not os.path.exists(requirements_txt):
        print(f"Warning: {requirements_txt} not found, creating empty file", file=sys.stderr)
        # 空のrequirements.txtを作成
        with open(requirements_txt, 'w') as f:
            f.write('')
    
    # tar.gzを作成
    try:
        with tarfile.open(output_tar, 'w:gz') as tar:
            tar.add(train_py, arcname='train.py')
            tar.add(requirements_txt, arcname='requirements.txt')
        
        file_size = os.path.getsize(output_tar)
        print(f"Created {output_tar} ({file_size} bytes)")
        return True
    except Exception as e:
        print(f"Error creating tar.gz: {e}", file=sys.stderr)
        return False

if __name__ == '__main__':
    success = create_tar()
    sys.exit(0 if success else 1)
