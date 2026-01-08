import { defineFunction } from '@aws-amplify/backend';

export const startEvaluationFunction = defineFunction({
  name: 'start-evaluation',
  timeoutSeconds: 60,
  memoryMB: 512,
});

