import { defineFunction } from '@aws-amplify/backend';

/**
 * SageMaker Training Jobを起動するLambda関数
 * スクリプトモードを使用（Dockerビルド不要）
 */
export const startTrainingFunction = defineFunction({
  name: 'start-training',
  entry: './handler.ts',
  timeoutSeconds: 30,
  memoryMB: 256,
  runtime: 20,
});
