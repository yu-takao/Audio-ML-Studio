import { defineFunction } from '@aws-amplify/backend';

export const getEvaluationStatusFunction = defineFunction({
  name: 'get-evaluation-status',
  timeoutSeconds: 30,
  memoryMB: 256,
});

