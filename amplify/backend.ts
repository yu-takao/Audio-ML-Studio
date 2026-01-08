import { defineBackend } from '@aws-amplify/backend';
import { auth } from './auth/resource';
import { storage } from './storage/resource';
import { startTrainingFunction } from './functions/start-training/resource';
import { getTrainingStatusFunction } from './functions/get-training-status/resource';
import { startAnalysisFunction } from './functions/start-analysis/resource';
import { startEvaluationFunction } from './functions/start-evaluation/resource';
import { getEvaluationStatusFunction } from './functions/get-evaluation-status/resource';
import { PolicyStatement, Effect } from 'aws-cdk-lib/aws-iam';
import { FunctionUrlAuthType, HttpMethod } from 'aws-cdk-lib/aws-lambda';
import { Duration } from 'aws-cdk-lib';

/**
 * バックエンド定義
 * - 認証（Cognito）
 * - ストレージ（S3）
 * - Lambda関数（SageMaker操作用）
 */
const backend = defineBackend({
  auth,
  storage,
  startTrainingFunction,
  getTrainingStatusFunction,
  startAnalysisFunction,
  startEvaluationFunction,
  getEvaluationStatusFunction,
});

// SageMaker操作用のIAMポリシーを追加
const sagemakerPolicy = new PolicyStatement({
  effect: Effect.ALLOW,
  actions: [
    'sagemaker:CreateTrainingJob',
    'sagemaker:DescribeTrainingJob',
    'sagemaker:StopTrainingJob',
    'sagemaker:ListTrainingJobs',
    'sagemaker:CreateProcessingJob',
    'sagemaker:DescribeProcessingJob',
    'sagemaker:StopProcessingJob',
    'sagemaker:ListProcessingJobs',
    'sagemaker:AddTags',
    'sagemaker:DeleteTags',
    'sagemaker:ListTags',
  ],
  resources: ['*'],
});

// S3アクセス用のIAMポリシー
const s3Policy = new PolicyStatement({
  effect: Effect.ALLOW,
  actions: [
    's3:GetObject',
    's3:PutObject',
    's3:DeleteObject',
    's3:ListBucket',
  ],
  resources: [
    backend.storage.resources.bucket.bucketArn,
    `${backend.storage.resources.bucket.bucketArn}/*`,
  ],
});

// IAM PassRole（SageMakerがS3にアクセスするため）
const passRolePolicy = new PolicyStatement({
  effect: Effect.ALLOW,
  actions: ['iam:PassRole'],
  resources: ['*'],
  conditions: {
    StringEquals: {
      'iam:PassedToService': 'sagemaker.amazonaws.com',
    },
  },
});

// Lambda関数にポリシーを追加
backend.startTrainingFunction.resources.lambda.addToRolePolicy(sagemakerPolicy);
backend.startTrainingFunction.resources.lambda.addToRolePolicy(s3Policy);
backend.startTrainingFunction.resources.lambda.addToRolePolicy(passRolePolicy);

backend.getTrainingStatusFunction.resources.lambda.addToRolePolicy(sagemakerPolicy);

// SageMaker IAMロールのARNを設定（作成したロールのARNに置き換えてください）
const sagemakerRoleArn = process.env.SAGEMAKER_ROLE_ARN || 'arn:aws:iam::910478837984:role/SageMakerExecutionRole';

// 環境変数を設定
backend.startTrainingFunction.resources.lambda.addEnvironment(
  'TRAINING_BUCKET',
  backend.storage.resources.bucket.bucketName
);
backend.startTrainingFunction.resources.lambda.addEnvironment(
  'SAGEMAKER_ROLE_ARN',
  sagemakerRoleArn
);
backend.getTrainingStatusFunction.resources.lambda.addEnvironment(
  'TRAINING_BUCKET',
  backend.storage.resources.bucket.bucketName
);

// startAnalysis Lambda設定
backend.startAnalysisFunction.resources.lambda.addToRolePolicy(sagemakerPolicy);
backend.startAnalysisFunction.resources.lambda.addToRolePolicy(s3Policy);
backend.startAnalysisFunction.resources.lambda.addToRolePolicy(passRolePolicy);
backend.startAnalysisFunction.resources.lambda.addEnvironment(
  'TRAINING_BUCKET',
  backend.storage.resources.bucket.bucketName
);
backend.startAnalysisFunction.resources.lambda.addEnvironment(
  'SAGEMAKER_ROLE_ARN',
  sagemakerRoleArn
);

// S3ライフサイクルポリシーを設定（一時データの自動削除）
backend.storage.resources.bucket.addLifecycleRule({
  id: 'DeleteEvaluationTempData',
  prefix: 'evaluation/temp/',
  enabled: true,
  expiration: Duration.days(1), // 24時間後に削除
});

backend.storage.resources.bucket.addLifecycleRule({
  id: 'DeleteEvaluationResults',
  prefix: 'evaluation/results/',
  enabled: true,
  expiration: Duration.days(7), // 7日後に削除
});

// Evaluation Lambda設定
backend.startEvaluationFunction.resources.lambda.addToRolePolicy(sagemakerPolicy);
backend.startEvaluationFunction.resources.lambda.addToRolePolicy(s3Policy);
backend.startEvaluationFunction.resources.lambda.addToRolePolicy(passRolePolicy);
backend.startEvaluationFunction.resources.lambda.addEnvironment(
  'TRAINING_BUCKET',
  backend.storage.resources.bucket.bucketName
);
backend.startEvaluationFunction.resources.lambda.addEnvironment(
  'SAGEMAKER_ROLE_ARN',
  sagemakerRoleArn
);

backend.getEvaluationStatusFunction.resources.lambda.addToRolePolicy(sagemakerPolicy);

// Lambda Function URLを追加（CORSはFunction URL側で処理）
const startTrainingUrl = backend.startTrainingFunction.resources.lambda.addFunctionUrl({
  authType: FunctionUrlAuthType.NONE,
  cors: {
    allowedOrigins: ['*'],
    allowedMethods: [HttpMethod.ALL],
    allowedHeaders: ['*'],
  },
});

const getStatusUrl = backend.getTrainingStatusFunction.resources.lambda.addFunctionUrl({
  authType: FunctionUrlAuthType.NONE,
  cors: {
    allowedOrigins: ['*'],
    allowedMethods: [HttpMethod.ALL],
    allowedHeaders: ['*'],
  },
});

const startAnalysisUrl = backend.startAnalysisFunction.resources.lambda.addFunctionUrl({
  authType: FunctionUrlAuthType.NONE,
  cors: {
    allowedOrigins: ['*'],
    allowedMethods: [HttpMethod.ALL],
    allowedHeaders: ['*'],
  },
});

const startEvaluationUrl = backend.startEvaluationFunction.resources.lambda.addFunctionUrl({
  authType: FunctionUrlAuthType.NONE,
  cors: {
    allowedOrigins: ['*'],
    allowedMethods: [HttpMethod.ALL],
    allowedHeaders: ['*'],
  },
});

const getEvaluationStatusUrl = backend.getEvaluationStatusFunction.resources.lambda.addFunctionUrl({
  authType: FunctionUrlAuthType.NONE,
  cors: {
    allowedOrigins: ['*'],
    allowedMethods: [HttpMethod.ALL],
    allowedHeaders: ['*'],
  },
});

// カスタム出力を追加（amplify_outputs.jsonに含める）
backend.addOutput({
  custom: {
    startTrainingUrl: startTrainingUrl.url,
    getTrainingStatusUrl: getStatusUrl.url,
    startAnalysisUrl: startAnalysisUrl.url,
    startEvaluationUrl: startEvaluationUrl.url,
    getEvaluationStatusUrl: getEvaluationStatusUrl.url,
  },
});

export default backend;



