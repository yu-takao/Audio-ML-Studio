import type { Handler } from 'aws-lambda';
import { SageMakerClient, CreateProcessingJobCommand } from '@aws-sdk/client-sagemaker';

const sagemakerClient = new SageMakerClient({});

// AWS提供のScikit-learn Processingイメージ（リージョン別）
// numpy, scipy, scikit-learnがプリインストール済み
const PROCESSING_IMAGES: Record<string, string> = {
  'ap-northeast-1': '354813040037.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
  'us-east-1': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
  'us-west-2': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
  'eu-west-1': '141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
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
    });

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

    // 環境変数を文字列として準備
    const environmentVars: Record<string, string> = {
      BUCKET_NAME: String(bucket),
      USER_ID: String(userId),
      JOB_NAME: String(processingJobName),
      TARGET_FIELD: String(config.targetField),
      AUXILIARY_FIELDS: JSON.stringify((config.auxiliaryFields || []).map(String)),
      CLASS_NAMES: JSON.stringify(config.classNames.map(String)),
      INPUT_HEIGHT: String(config.inputHeight || 128),
      INPUT_WIDTH: String(config.inputWidth || 128),
    };

    console.log('Environment variables:', environmentVars);

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
            S3Uri: `s3://${bucket}/public/scripts/evaluation/`,
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
      Environment: environmentVars,
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

