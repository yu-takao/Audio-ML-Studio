import { defineFunction } from '@aws-amplify/backend';

export const startEvaluationFunction = defineFunction({
  name: 'start-evaluation',
  timeoutSeconds: 60,
  memoryMB: 512,
  // Force redeploy: 2026-01-09T13:28:00 - Fix NUMBER_VALUE conversion error
});

