# モデル評価機能 実装サマリー

## ✅ 実装完了項目

### 1. バックエンド (AWS)

#### S3ストレージ設定
- **ファイル**: `amplify/storage/resource.ts`
- **変更内容**:
  - `evaluation/temp/` パス追加（一時データ用）
  - `evaluation/results/` パス追加（結果保存用）

#### S3ライフサイクルポリシー
- **ファイル**: `amplify/backend.ts`
- **変更内容**:
  - `evaluation/temp/` → 24時間で自動削除
  - `evaluation/results/` → 7日で自動削除
  - AWS CDKで実装

#### Lambda関数: start-evaluation
- **ファイル**:
  - `amplify/functions/start-evaluation/resource.ts`
  - `amplify/functions/start-evaluation/handler.ts`
- **機能**:
  - SageMaker Processing Jobを起動
  - S3からデータとモデルを読み込み
  - 評価スクリプトを実行

#### Lambda関数: get-evaluation-status
- **ファイル**:
  - `amplify/functions/get-evaluation-status/resource.ts`
  - `amplify/functions/get-evaluation-status/handler.ts`
- **機能**:
  - Processing Jobのステータスを取得
  - 完了時に結果パスを返す

#### バックエンド統合
- **ファイル**: `amplify/backend.ts`
- **変更内容**:
  - 新しいLambda関数を登録
  - IAMポリシーを追加
  - Function URLを作成
  - amplify_outputs.jsonに出力

### 2. SageMaker評価スクリプト

#### 評価スクリプト
- **ファイル**: `sagemaker-scripts/evaluate.py`
- **機能**:
  - TFJSモデルを読み込み
  - 音声ファイルを処理（スペクトログラム変換）
  - 全ファイルを推論
  - 精度メトリクスを計算
  - 混同行列を生成
  - 結果をJSON/CSVで保存

#### アップロードスクリプト
- **ファイル**:
  - `sagemaker-scripts/upload-evaluation-script.sh` (Mac/Linux)
  - `sagemaker-scripts/upload-evaluation-script.bat` (Windows)
- **機能**:
  - `evaluate.py` をS3にアップロード
  - `requirements.txt` をS3にアップロード

### 3. フロントエンド (React)

#### 混同行列コンポーネント
- **ファイル**: `src/components/ConfusionMatrix.tsx`
- **機能**:
  - 混同行列を視覚化
  - ヒートマップ表示
  - 正解/誤りの色分け
  - パーセンテージ表示

#### 結果表示コンポーネント
- **ファイル**: `src/components/InferenceResults.tsx`
- **機能**:
  - 全体メトリクスを表示（Accuracy, F1, Precision, Recall）
  - 混同行列を表示
  - クラス別メトリクステーブル
  - ファイル別予測結果テーブル（最大100件）
  - サマリー統計（総サンプル数、正解数、誤り数）

#### メイン評価コンポーネント
- **ファイル**: `src/components/ModelEvaluation.tsx`
- **機能**:
  - 5ステップのウィザード形式UI
    1. モデル選択（S3から）
    2. データ選択（ローカルフォルダ）
    3. メタデータ設定
    4. 評価実行中（進捗表示）
    5. 結果表示
  - S3へのアップロード（進捗バー付き）
  - SageMaker Processing Job起動
  - ステータスポーリング（10秒間隔）
  - 結果の自動読み込み
  - エラーハンドリング

#### App統合
- **ファイル**: `src/App.tsx`
- **変更内容**:
  - 新しいタブ「モデル評価」を追加
  - アイコン: `Target`
  - タブID: `evaluation`
  - `ModelEvaluation`コンポーネントをマウント

### 4. ドキュメント

#### セットアップガイド
- **ファイル**: `MODEL_EVALUATION_SETUP.md`
- **内容**:
  - アーキテクチャ図
  - セットアップ手順
  - 使い方
  - トラブルシューティング
  - コスト目安
  - ベストプラクティス

## 📊 ファイル構成

```
audio-augmentation-app/
├── amplify/
│   ├── backend.ts                              (変更) Lambda統合、ライフサイクル追加
│   ├── storage/
│   │   └── resource.ts                         (変更) evaluationパス追加
│   └── functions/
│       ├── start-evaluation/                   (新規) 評価ジョブ起動
│       │   ├── resource.ts
│       │   └── handler.ts
│       └── get-evaluation-status/              (新規) ステータス取得
│           ├── resource.ts
│           └── handler.ts
├── sagemaker-scripts/
│   ├── evaluate.py                             (新規) 評価スクリプト
│   ├── upload-evaluation-script.sh             (新規) アップロード用
│   └── upload-evaluation-script.bat            (新規) アップロード用(Win)
├── src/
│   ├── App.tsx                                 (変更) 評価タブ追加
│   └── components/
│       ├── ModelEvaluation.tsx                 (新規) メイン評価画面
│       ├── InferenceResults.tsx                (新規) 結果表示
│       └── ConfusionMatrix.tsx                 (新規) 混同行列
├── MODEL_EVALUATION_SETUP.md                   (新規) セットアップガイド
└── IMPLEMENTATION_SUMMARY.md                   (新規) このファイル
```

## 🎯 主要な技術的判断

### 1. すべてS3経由に統一

**理由**:
- コードがシンプル（条件分岐なし）
- 大量データにも対応
- 一貫性のあるアーキテクチャ
- SageMakerとの親和性が高い

### 2. S3ライフサイクルポリシーで自動削除

**理由**:
- コード不要（AWS側で自動処理）
- 確実に削除される（Lambdaの失敗に依存しない）
- ストレージコストを抑える
- AWSのベストプラクティス

**設定**:
- `evaluation/temp/`: 24時間で削除
- `evaluation/results/`: 7日で削除

### 3. SageMaker Processing（訓練ではなく）

**理由**:
- 推論専用（軽量）
- CPUインスタンスで十分（コスト削減）
- 訓練ジョブより柔軟
- 短時間で完了

**インスタンスタイプ**: `ml.m5.xlarge` (CPU)

### 4. メタデータ設定の再利用

**理由**:
- 既存の`MetadataConfig`コンポーネントを再利用
- 一貫性のあるUI/UX
- 訓練時と同じロジック
- 開発工数の削減

### 5. ポーリング方式のステータス確認

**理由**:
- シンプルな実装
- WebSocketやEventBridgeより軽量
- 10秒間隔で十分（処理時間5-15分）
- フロントエンドで完結

## 🔄 データフロー

```
1. ユーザーがモデルを選択
   ↓
2. ユーザーがローカルフォルダからデータを選択
   ↓
3. ファイル名からメタデータを自動抽出
   ↓
4. ユーザーがメタデータ設定（訓練時と同じ）
   ↓
5. フロント: データをS3にアップロード
   (evaluation/temp/<userId>/eval-<timestamp>/)
   ↓
6. フロント: Lambda (start-evaluation) を呼び出し
   ↓
7. Lambda: SageMaker Processing Job を起動
   - 入力: S3のデータ + モデル + スクリプト
   - 出力: S3の結果パス
   ↓
8. SageMaker: evaluate.py を実行
   - モデル読み込み
   - 全ファイルを推論
   - メトリクス計算
   - 結果をS3に保存
   ↓
9. フロント: 10秒ごとにステータスをポーリング
   (Lambda: get-evaluation-status)
   ↓
10. 完了時: フロントが結果をS3から読み込み
    (metrics.json + predictions.csv)
   ↓
11. フロント: 結果を表示
    - 混同行列
    - 精度メトリクス
    - クラス別性能
    - ファイル別予測
   ↓
12. AWS: 24時間後に一時データを自動削除
    7日後に結果を自動削除
```

## 📈 パフォーマンス

| 項目 | 性能 |
|-----|------|
| アップロード速度 | ネットワーク依存 |
| 処理時間 | 5-15分（データ量による） |
| ポーリング間隔 | 10秒 |
| 結果表示 | 即座 |
| 最大ファイル数 | 数千ファイル（SageMakerの制限内） |

## 💰 コスト

| 項目 | コスト |
|-----|--------|
| Lambda実行 | ほぼ無料（短時間） |
| S3ストレージ | ~$0.01/GB/月 |
| S3リクエスト | 微小 |
| SageMaker Processing | $0.269/時間 (ml.m5.xlarge) |
| **合計（500ファイル）** | **約$0.02-0.04/回** |

## 🔒 セキュリティ

### IAMポリシー
- Lambda → SageMaker: 必要最小限の権限
- Lambda → S3: 読み書き権限のみ
- SageMaker → S3: IAM PassRole経由

### データ保護
- 一時データは自動削除
- Cognito認証が必須
- S3アクセスは認証ユーザーのみ

### CORS設定
- Lambda Function URLでCORS設定
- オリジン制限（必要に応じて調整可能）

## 🎓 今後の改善案

### 短期的
1. バッチサイズの自動調整
2. より詳細なエラーメッセージ
3. 評価履歴の保存
4. 結果のエクスポート機能（PDF/Excel）

### 中期的
1. A/Bテスト機能（複数モデル比較）
2. リアルタイム推論（WebSocket）
3. データドリフト検出
4. 異常検知

### 長期的
1. AutoML機能
2. モデルバージョン管理
3. CI/CD統合
4. 本番モニタリング

## 📝 メンテナンス

### 定期的に確認すべき項目
- [ ] S3ライフサイクルポリシーが動作しているか
- [ ] Lambda関数のタイムアウト設定
- [ ] SageMakerインスタンスの可用性
- [ ] CloudWatch Logsのサイズ

### アップデート時の注意
- モデルフォーマットの互換性
- メタデータスキーマの変更
- TensorFlow/TFJSバージョン
- Python依存関係

## 🎉 完了！

これで、訓練したモデルを別データで評価できるようになりました！

**次のステップ:**
1. 評価スクリプトをS3にアップロード
2. バックエンドをデプロイ
3. フロントエンドを起動
4. 実際のデータで試す

詳細は `MODEL_EVALUATION_SETUP.md` を参照してください。

