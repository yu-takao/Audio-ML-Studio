# クラウド訓練機能のセットアップガイド

このガイドでは、AWS SageMaker スクリプトモードを使用したクラウド訓練機能のセットアップ手順を説明します。

## 特徴

✅ **Dockerビルド不要** - AWSが提供するTensorFlowイメージを使用
✅ **シンプルなセットアップ** - Pythonスクリプトを S3 にアップロードするだけ
✅ **オンデマンド課金** - 訓練時のみ GPU インスタンスが起動

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│  React App (Amplify Hosting)                                    │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │ 訓練モード選択   │    │ データアップロード│                    │
│  │ ○ ブラウザ       │    │ → S3           │                    │
│  │ ● クラウド       │    │                │                    │
│  └─────────────────┘    └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  AWS                                                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ SageMaker Training Job                                    │  │
│  │                                                           │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐ │  │
│  │  │ AWS提供の    │ +  │ train.py    │ =  │ GPU訓練      │ │  │
│  │  │ TensorFlow   │    │ (S3から)    │    │ (ml.g4dn)    │ │  │
│  │  │ イメージ     │    │             │    │              │ │  │
│  │  └─────────────┘    └─────────────┘    └──────────────┘ │  │
│  │                                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ↓ モデル保存                          │
│                    ┌──────────────┐                            │
│                    │ S3 Bucket    │                            │
│                    │ (tfjs_model) │                            │
│                    └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

## 前提条件

- AWS アカウント
- AWS CLI がインストール済み（`aws configure` 設定済み）
- Node.js 18以上

**Docker は不要です！**

## セットアップ手順

### 1. SageMaker IAM ロールの作成

AWS コンソールで IAM ロールを作成します：

1. **IAM > ロール > ロールを作成**
2. **信頼されたエンティティ**: `sagemaker.amazonaws.com`
3. **以下のポリシーをアタッチ**:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`

4. ロールARNをメモ（例: `arn:aws:iam::123456789012:role/SageMakerExecutionRole`）

### 2. トレーニングスクリプトをS3にアップロード

#### Windows の場合:

```powershell
cd audio-augmentation-app\sagemaker
.\upload-script.bat <your-bucket-name> ap-northeast-1
```

#### Mac/Linux の場合:

```bash
cd audio-augmentation-app/sagemaker
chmod +x upload-script.sh
./upload-script.sh <your-bucket-name> ap-northeast-1
```

#### 手動の場合:

```bash
# スクリプトを圧縮
tar -czvf audio-ml-training.tar.gz train.py

# S3にアップロード
aws s3 cp audio-ml-training.tar.gz s3://<your-bucket>/scripts/
```

### 3. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成：

```env
SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerExecutionRole
VITE_API_ENDPOINT=https://your-api-id.execute-api.ap-northeast-1.amazonaws.com
```

### 4. Amplify バックエンドのデプロイ

```bash
cd audio-augmentation-app

# 依存関係をインストール
npm install

# サンドボックス環境を起動（開発用）
npx ampx sandbox

# 本番環境にデプロイ
npx ampx pipeline-deploy --branch main
```

### 5. フロントエンドの起動

```bash
npm run dev
```

## 使い方

1. **データフォルダを選択**
2. **メタデータを設定**（ターゲット、補助パラメータ）
3. **訓練モードで「クラウドで訓練」を選択**
4. **「アップロード」でデータをS3に転送**
5. **「訓練開始」でSageMaker Training Jobを起動**
6. **完了後、「モデルを読み込む」でダウンロード**

## コスト目安

| リソース | 課金 | 概算コスト |
|---------|------|-----------|
| S3 | 保存量 | ~$0.01/GB/月 |
| Lambda | 呼び出し | ほぼ無料 |
| **SageMaker Training** | **訓練時間のみ** | **$0.526/時間** |

### 例: 1000ファイルの訓練（10-30分）

**約 $0.09 ～ $0.26 / 回**

## トラブルシューティング

### Q: 訓練ジョブが起動しない

1. SageMaker IAM ロールに必要な権限があるか確認
2. S3 バケットへのアクセス権限を確認
3. `train.py` が正しく S3 にアップロードされているか確認

### Q: 訓練中にエラーが発生

CloudWatch Logs でエラーを確認：
```bash
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/TrainingJobs
```

### Q: モデルがダウンロードできない

1. S3 バケットの CORS 設定を確認
2. Amplify Storage のアクセス権限を確認

## ファイル構成

```
audio-augmentation-app/
├── amplify/
│   ├── backend.ts              # バックエンド定義
│   ├── auth/resource.ts        # 認証設定
│   ├── storage/resource.ts     # S3設定
│   └── functions/
│       ├── start-training/     # 訓練開始Lambda
│       └── get-training-status/ # ステータス確認Lambda
├── sagemaker/
│   ├── train.py                # トレーニングスクリプト
│   ├── upload-script.sh        # S3アップロード（Mac/Linux）
│   └── upload-script.bat       # S3アップロード（Windows）
└── src/components/
    └── CloudTraining.tsx       # クラウド訓練UI
```

## スクリプトモード vs Docker

| 項目 | スクリプトモード（現在） | Docker (BYOC) |
|------|------------------------|---------------|
| **Dockerビルド** | 不要 ✅ | 必要 |
| **ECR** | 不要 ✅ | 必要 |
| **セットアップ** | 簡単 ✅ | 複雑 |
| **カスタマイズ** | 制限あり | 自由 |
| **推奨ケース** | 標準的なML | 特殊な環境が必要 |
