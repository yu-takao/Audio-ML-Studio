import { defineStorage } from '@aws-amplify/backend';

/**
 * ストレージ設定（S3）
 * - training-data/: 訓練データ（音声ファイル）
 * - models/: 訓練済みモデル
 * - configs/: 訓練設定
 */
export const storage = defineStorage({
  name: 'audioMLStorage',
  access: (allow) => ({
    // 訓練データ用パス（認証済みユーザーがアップロード可能）
    'training-data/{entity_id}/*': [
      // 訓練データの一覧取得・読み書き・削除を許可
      allow.entity('identity').to(['read', 'write', 'delete', 'list']),
    ],
    // モデル保存用パス（認証ユーザー全員に一覧・読書き・削除を許可）
    // ※ {entity_id} とのプレフィックス衝突を避けるため、単一パスで管理
    'models/*': [
      allow.authenticated.to(['read', 'write', 'delete', 'list']),
    ],
    // 訓練設定用パス
    'configs/{entity_id}/*': [
      allow.entity('identity').to(['read', 'write', 'delete', 'list']),
    ],
    // ゲストアクセス用（デモ用）
    'public/*': [
      // 認証ユーザーは一覧取得もできるようにする
      allow.guest.to(['read', 'write']),
      allow.authenticated.to(['read', 'write', 'delete', 'list']),
    ],
  }),
});



