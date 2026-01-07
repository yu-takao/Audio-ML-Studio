import { defineFunction } from '@aws-amplify/backend';

/**
 * モデル解析（Grad-CAM等）を開始するLambda関数
 * SageMaker Processing Jobを起動する
 */
export const startAnalysisFunction = defineFunction({
  name: 'start-analysis',
  entry: './handler.ts',
  timeoutSeconds: 60,
  memoryMB: 256,
  runtime: 20,
});

