import type { Handler } from 'aws-lambda';
import { SageMakerClient, CreateProcessingJobCommand } from '@aws-sdk/client-sagemaker';

const sagemakerClient = new SageMakerClient({});

// AWS提供のScikit-learn/TensorFlowイメージ（リージョン別）
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
  console.log('Event:', JSON.stringify(event, null, 2));

  try {
    const body: EvaluationRequest = JSON.parse(event.body || '{}');
    const { config, userId, jobName } = body;

    if (!config || !userId) {
      return {
        statusCode: 400,
        headers: responseHeaders,
        body: JSON.stringify({ error: 'Missing config or userId' }),
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

    // SageMaker Processing Jobの設定
    const command = new CreateProcessingJobCommand({
      ProcessingJobName: processingJobName,
      RoleArn: sagemakerRoleArn,
      AppSpecification: {
        ImageUri: processingImage,
        ContainerEntrypoint: ['python3', '/opt/ml/processing/input/code/evaluate.py'],
      },
      ProcessingInputs: [
        // 評価スクリプト
        {
          InputName: 'code',
          S3Input: {
            S3Uri: `s3://${bucket}/public/scripts/evaluation/`,
            LocalPath: '/opt/ml/processing/input/code',
            S3DataType: 'S3Prefix',
            S3InputMode: 'File',
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
      Environment: {
        BUCKET_NAME: bucket,
        USER_ID: userId,
        JOB_NAME: processingJobName,
        TARGET_FIELD: String(config.targetField),
        AUXILIARY_FIELDS: JSON.stringify(config.auxiliaryFields || []),
        CLASS_NAMES: JSON.stringify(config.classNames),
        INPUT_HEIGHT: String(config.inputHeight || 128),
        INPUT_WIDTH: String(config.inputWidth || 128),
      },
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

