import { defineFunction } from '@aws-amplify/backend';

/**
 * モデル解析（Grad-CAM等）を開始するLambda関数
 * SageMaker Processing Jobを起動する
 */
export const startAnalysisFunction = defineFunction({
  name: 'start-analysis',
  entry: './handler.ts',
  timeoutSeconds: 90, // 増やして再ビルドを誘発
  memoryMB: 512,      // 増やして再ビルドを誘発
  runtime: 20,
});



