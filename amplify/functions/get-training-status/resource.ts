import { defineFunction } from '@aws-amplify/backend';

/**
 * SageMaker Training Jobのステータスを取得するLambda関数
 */
export const getTrainingStatusFunction = defineFunction({
  name: 'get-training-status',
  entry: './handler.ts',
  timeoutSeconds: 10,
  memoryMB: 128,
  runtime: 20,
});



