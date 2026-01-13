import type { Handler } from 'aws-lambda';
import { SageMakerClient, CreateProcessingJobCommand } from '@aws-sdk/client-sagemaker';

const sagemakerClient = new SageMakerClient({});

// AWS提供のTensorFlow Trainingイメージ（リージョン別）
// TensorFlow, numpy, scipy, scikit-learn がすべて互換性検証済みでプリインストール
const PROCESSING_IMAGES: Record<string, string> = {
  'ap-northeast-1': '763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/tensorflow-training:2.13.0-cpu-py310-ubuntu20.04-sagemaker',
  'us-east-1': '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.13.0-cpu-py310-ubuntu20.04-sagemaker',
  'us-west-2': '763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.13.0-cpu-py310-ubuntu20.04-sagemaker',
  'eu-west-1': '763104351884.dkr.ecr.eu-west-1.amazonaws.com/tensorflow-training:2.13.0-cpu-py310-ubuntu20.04-sagemaker',
};

interface EvaluationConfig {
  dataPath: string; // S3パス（例: evaluation/temp/user123/dataset-20260108/）
  modelPath: string; // S3パス（例: models/user123/model.json）
  targetField: number | string;
  auxiliaryFields: (number | string)[];
  classNames: string[];
  inputHeight: number;
  inputWidth: number;
}

interface EvaluationRequest {
  config: EvaluationConfig;
  userId: string;
  jobName?: string;
}

const responseHeaders = {
  'Content-Type': 'application/json',
};

export const handler: Handler = async (event) => {
  console.log('=== START EVALUATION HANDLER v2 ===');
  console.log('Event:', JSON.stringify(event, null, 2));
  console.log('Start evaluation job request');

  try {
    const body: EvaluationRequest = JSON.parse(event.body || '{}');
    const { config, userId, jobName } = body;

    console.log('Request body:', {
      userId,
      targetField: config?.targetField,
      targetFieldType: typeof config?.targetField,
      auxiliaryFields: config?.auxiliaryFields,
      auxiliaryFieldsTypes: config?.auxiliaryFields?.map(f => typeof f),
      inputHeight: config?.inputHeight,
      inputWidth: config?.inputWidth,
    });

    if (!config || !userId) {
      return {
        statusCode: 400,
        headers: responseHeaders,
        body: JSON.stringify({ error: 'Missing config or userId' }),
      };
    }

    // targetFieldの検証とデフォルト値
    if (config.targetField === undefined || config.targetField === null) {
      return {
        statusCode: 400,
        headers: responseHeaders,
        body: JSON.stringify({ error: 'targetField is required' }),
      };
    }

    const bucket = process.env.TRAINING_BUCKET;
    const sagemakerRoleArn = process.env.SAGEMAKER_ROLE_ARN;
    const region = process.env.AWS_REGION || 'ap-northeast-1';

    if (!bucket || !sagemakerRoleArn) {
      return {
        statusCode: 500,
        headers: responseHeaders,
        body: JSON.stringify({
          error: 'Server configuration missing',
          details: { bucket: !!bucket, sagemakerRoleArn: !!sagemakerRoleArn },
        }),
      };
    }

    const processingImage = PROCESSING_IMAGES[region] || PROCESSING_IMAGES['ap-northeast-1'];
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const processingJobName = jobName || `audio-eval-${userId.slice(0, 8)}-${timestamp}`;

    // 環境変数を文字列として準備（すべての値を明示的に文字列に変換）
    // SageMakerは環境変数を文字列としてのみ受け付けるため、数値も文字列に変換する必要がある
    const targetFieldValue = config.targetField !== undefined && config.targetField !== null
      ? String(config.targetField)
      : '0';

    const auxiliaryFieldsValue = Array.isArray(config.auxiliaryFields)
      ? JSON.stringify(config.auxiliaryFields.map(f => String(f !== undefined && f !== null ? f : '')))
      : JSON.stringify([]);

    const classNamesValue = Array.isArray(config.classNames)
      ? JSON.stringify(config.classNames.map(c => String(c || '')))
      : JSON.stringify([]);

    const inputHeightValue = config.inputHeight !== undefined && config.inputHeight !== null
      ? String(config.inputHeight)
      : '128';

    const inputWidthValue = config.inputWidth !== undefined && config.inputWidth !== null
      ? String(config.inputWidth)
      : '128';

    // すべての環境変数を確実に文字列に変換（型安全性のため）
    const environmentVars: Record<string, string> = {
      BUCKET_NAME: String(bucket || ''),
      USER_ID: String(userId || ''),
      JOB_NAME: String(processingJobName || ''),
      TARGET_FIELD: String(targetFieldValue),
      AUXILIARY_FIELDS: String(auxiliaryFieldsValue),
      CLASS_NAMES: String(classNamesValue),
      INPUT_HEIGHT: String(inputHeightValue),
      INPUT_WIDTH: String(inputWidthValue),
    };

    // デバッグ: すべての値が文字列であることを確認
    console.log('Environment variables (before SageMaker):', JSON.stringify(environmentVars, null, 2));
    console.log('Environment variables types:', Object.entries(environmentVars).map(([k, v]) => ({
      key: k,
      value: v,
      type: typeof v,
      isString: typeof v === 'string',
    })));

    // SageMaker Processing Jobの設定
    const command = new CreateProcessingJobCommand({
      ProcessingJobName: processingJobName,
      RoleArn: sagemakerRoleArn,
      AppSpecification: {
        ImageUri: processingImage,
        ContainerEntrypoint: ['python3', '/opt/ml/processing/input/code/evaluate.py'],
      },
      ProcessingInputs: [
        // 評価スクリプト（個別ファイルとしてフォルダから）
        {
          InputName: 'code',
          S3Input: {
            S3Uri: `s3://${bucket}/public/scripts/evaluation-v2/`,
            LocalPath: '/opt/ml/processing/input/code',
            S3DataType: 'S3Prefix',
            S3InputMode: 'File',
            S3DataDistributionType: 'FullyReplicated',
          },
        },
        // 評価データ
        {
          InputName: 'data',
          S3Input: {
            S3Uri: `s3://${bucket}/${config.dataPath}`,
            LocalPath: '/opt/ml/processing/input/data',
            S3DataType: 'S3Prefix',
            S3InputMode: 'File',
          },
        },
        // モデルファイル
        {
          InputName: 'model',
          S3Input: {
            S3Uri: `s3://${bucket}/${config.modelPath}`,
            LocalPath: '/opt/ml/processing/input/model',
            S3DataType: 'S3Prefix',
            S3InputMode: 'File',
          },
        },
      ],
      ProcessingOutputConfig: {
        Outputs: [
          {
            OutputName: 'results',
            S3Output: {
              S3Uri: `s3://${bucket}/evaluation/results/${userId}/${processingJobName}/`,
              LocalPath: '/opt/ml/processing/output',
              S3UploadMode: 'EndOfJob',
            },
          },
        ],
      },
      ProcessingResources: {
        ClusterConfig: {
          InstanceCount: 1,
          InstanceType: 'ml.m5.xlarge', // CPUインスタンス（コスト効率良い）
          VolumeSizeInGB: 30,
        },
      },
      StoppingCondition: {
        MaxRuntimeInSeconds: 3600, // 1時間
      },
      // 環境変数を確実に文字列型として渡す（SageMakerの要件）
      Environment: environmentVars as Record<string, string>,
      Tags: [
        { Key: 'Application', Value: 'AudioMLStudio' },
        { Key: 'JobType', Value: 'Evaluation' },
        { Key: 'UserId', Value: userId },
      ],
    });

    await sagemakerClient.send(command);

    return {
      statusCode: 200,
      headers: responseHeaders,
      body: JSON.stringify({
        success: true,
        processingJobName,
        message: 'Evaluation job started successfully',
        estimatedTime: '5-15 minutes',
        resultPath: `evaluation/results/${userId}/${processingJobName}/`,
      }),
    };
  } catch (error) {
    console.error('Error starting evaluation job:', error);

    return {
      statusCode: 500,
      headers: responseHeaders,
      body: JSON.stringify({
        error: 'Failed to start evaluation job',
        details: (error as Error).message,
      }),
    };
  }
};

