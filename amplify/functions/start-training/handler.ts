import type { Handler } from 'aws-lambda';
import { SageMakerClient, CreateTrainingJobCommand } from '@aws-sdk/client-sagemaker';

// Last deployed: 2026-01-06T16:17:00Z - Force redeploy with correct v2 script path and metadata params
const sagemakerClient = new SageMakerClient({});

// AWS提供のTensorFlowトレーニングイメージ（リージョン別）
const TENSORFLOW_IMAGES: Record<string, string> = {
  'ap-northeast-1': '763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/tensorflow-training:2.13.0-gpu-py310-cu118-ubuntu20.04-sagemaker',
  'us-east-1': '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.13.0-gpu-py310-cu118-ubuntu20.04-sagemaker',
  'us-west-2': '763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.13.0-gpu-py310-cu118-ubuntu20.04-sagemaker',
  'eu-west-1': '763104351884.dkr.ecr.eu-west-1.amazonaws.com/tensorflow-training:2.13.0-gpu-py310-cu118-ubuntu20.04-sagemaker',
};

interface FieldLabel {
  index: number;
  label: string;
}

interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
  testSplit: number;
  dataPath: string;
  targetField: string;
  auxiliaryFields: string[];
  fieldLabels?: FieldLabel[]; // フィールドラベル情報
  problemType?: 'classification' | 'regression'; // 問題タイプ
  tolerance?: number; // 許容範囲
  classNames: string[];
  inputHeight: number;
  inputWidth: number;
}

interface TrainingRequest {
  config: TrainingConfig;
  userId: string;
  jobName?: string;
}

// レスポンスヘッダー（CORSはFunction URLで処理）
const responseHeaders = {
  'Content-Type': 'application/json',
};

export const handler: Handler = async (event) => {
  console.log('Event:', JSON.stringify(event, null, 2));

  try {
    const body: TrainingRequest = JSON.parse(event.body || '{}');
    const { config, userId, jobName } = body;

    if (!config || !userId) {
      return {
        statusCode: 400,
        headers: responseHeaders,
        body: JSON.stringify({ error: 'Missing config or userId' }),
      };
    }

    // デバッグ: 受信したconfigをログ出力
    console.log('Received config:', JSON.stringify(config, null, 2));

    const bucket = process.env.TRAINING_BUCKET;
    const sagemakerRoleArn = process.env.SAGEMAKER_ROLE_ARN;
    const region = process.env.AWS_REGION || 'ap-northeast-1';

    if (!bucket || !sagemakerRoleArn) {
      return {
        statusCode: 500,
        headers: responseHeaders,
        body: JSON.stringify({
          error: 'Server configuration missing',
          details: { bucket: !!bucket, sagemakerRoleArn: !!sagemakerRoleArn }
        }),
      };
    }

    // AWS提供のTensorFlowイメージを使用（Dockerビルド不要）
    const trainingImage = TENSORFLOW_IMAGES[region] || TENSORFLOW_IMAGES['ap-northeast-1'];

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const trainingJobName = jobName || `audio-ml-${userId.slice(0, 8)}-${timestamp}`;

    // ハイパーパラメータ（SageMakerスクリプトモード用）
    const hyperParameters = {
      // SageMaker スクリプトモード設定
      'sagemaker_program': 'train.py',
      // バージョン付きキーで常に最新スクリプトを取得
      'sagemaker_submit_directory': `s3://${bucket}/public/scripts/audio-ml-training-v2.tar.gz`,
      'sagemaker_region': region,
      // S3アップロード用パラメータ（スクリプト側で環境変数と併用）
      'bucket_name': bucket,
      'user_id': userId,
      'job_name': trainingJobName,
      // モデルパラメータ
      'epochs': String(config.epochs),
      'batch_size': String(config.batchSize),
      'learning_rate': String(config.learningRate),
      'validation_split': String(config.validationSplit),
      'test_split': String(config.testSplit),
      'input_height': String(config.inputHeight || 128),
      'input_width': String(config.inputWidth || 128),
      'target_field': config.targetField,
      'auxiliary_fields': JSON.stringify(config.auxiliaryFields || []),
      'field_labels': JSON.stringify(config.fieldLabels || []), // フィールドラベル情報を追加
      'problem_type': config.problemType || 'classification', // 問題タイプを追加
      'tolerance': String(config.tolerance || 0), // 許容範囲を追加
      'class_names': JSON.stringify(config.classNames),
    };

    const command = new CreateTrainingJobCommand({
      TrainingJobName: trainingJobName,
      AlgorithmSpecification: {
        TrainingImage: trainingImage,
        TrainingInputMode: 'File',
      },
      RoleArn: sagemakerRoleArn,
      InputDataConfig: [
        {
          ChannelName: 'training',
          DataSource: {
            S3DataSource: {
              S3DataType: 'S3Prefix',
              S3Uri: `s3://${bucket}/${config.dataPath}`,
              S3DataDistributionType: 'FullyReplicated',
            },
          },
          ContentType: 'application/x-audio',
          CompressionType: 'None',
        },
      ],
      OutputDataConfig: {
        S3OutputPath: `s3://${bucket}/models/${userId}/`,
      },
      ResourceConfig: {
        InstanceType: 'ml.g4dn.xlarge',
        InstanceCount: 1,
        VolumeSizeInGB: 50,
      },
      StoppingCondition: {
        MaxRuntimeInSeconds: 7200,
      },
      HyperParameters: hyperParameters,
      // トレーニングスクリプトにS3アップロード先情報を渡す
      Environment: {
        // 既存の環境変数に影響を与えないよう必要最低限を渡す
        BUCKET_NAME: bucket,
        USER_ID: userId,
        JOB_NAME: trainingJobName,
      },
      Tags: [
        { Key: 'Application', Value: 'AudioMLStudio' },
        { Key: 'UserId', Value: userId },
      ],
    });

    await sagemakerClient.send(command);

    return {
      statusCode: 200,
      headers: responseHeaders,
      body: JSON.stringify({
        success: true,
        trainingJobName,
        message: 'Training job started successfully',
        estimatedTime: '10-30 minutes',
      }),
    };
  } catch (error) {
    console.error('Error starting training job:', error);

    return {
      statusCode: 500,
      headers: responseHeaders,
      body: JSON.stringify({
        error: 'Failed to start training job',
        details: (error as Error).message,
      }),
    };
  }
};
