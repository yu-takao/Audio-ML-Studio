import { defineAuth } from '@aws-amplify/backend';

/**
 * 認証設定
 * ゲストアクセスを許可（ログインなしでも使用可能）
 */
export const auth = defineAuth({
  loginWith: {
    email: true,
  },
  // ゲストアクセスを許可
  // groups: ['admin', 'user'],
});



