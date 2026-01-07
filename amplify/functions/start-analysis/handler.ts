import type { Handler } from 'aws-lambda';
import { SageMakerClient, CreateProcessingJobCommand } from '@aws-sdk/client-sagemaker';

const sagemakerClient = new SageMakerClient({});

interface AnalysisRequest {
  userId: string;
  modelJobName: string; // 解析対象モデルのジョブ名
  dataPath?: string; // 解析対象データのS3パス（省略時はモデルの訓練データから推定）
}

// レスポンスヘッダー
const responseHeaders = {
  'Content-Type': 'application/json',
};

export const handler: Handler = async (event) => {
  console.log('Event:', JSON.stringify(event, null, 2));

  try {
    // リクエストボディを解析
    let request: AnalysisRequest;
    if (event.body) {
      request = JSON.parse(event.body);
    } else {
      request = event;
    }

    const { userId, modelJobName, dataPath } = request;

    if (!userId || !modelJobName) {
      return {
        statusCode: 400,
        headers: responseHeaders,
        body: JSON.stringify({ error: 'Missing userId or modelJobName' }),
      };
    }

    // 環境変数から取得
    const bucket = process.env.TRAINING_BUCKET;
    const sagemakerRoleArn = process.env.SAGEMAKER_ROLE_ARN;
    const region = process.env.AWS_REGION || 'ap-northeast-1';

    if (!bucket || !sagemakerRoleArn) {
      return {
        statusCode: 500,
        headers: responseHeaders,
        body: JSON.stringify({
          error: 'Server configuration error',
          details: { bucket: !!bucket, sagemakerRoleArn: !!sagemakerRoleArn },
        }),
      };
    }

    // Processing Job名を生成
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const processingJobName = `analysis-${userId.substring(0, 8)}-${timestamp}`.substring(0, 63);

    // モデルパス（model.tar.gzのあるディレクトリ）
    const modelS3Uri = `s3://${bucket}/models/${userId}/${modelJobName}/output`;

    // データパス（指定がなければ訓練データを探す）
    // 通常はpublic/training-data/{userId}/{timestamp}の構造
    // ここでは簡易的にモデルメタデータからは取らず、dataPathを必須とする
    const analysisDataPath = dataPath || `public/training-data/${userId}`;
    const dataS3Uri = `s3://${bucket}/${analysisDataPath}`;

    // 出力パス
    const outputS3Uri = `s3://${bucket}/models/${userId}/${modelJobName}/analysis/${timestamp}`;

    // Processing用コンテナ（TensorFlow）
    const processingImage = `763104351884.dkr.ecr.${region}.amazonaws.com/tensorflow-training:2.13.0-cpu-py310-ubuntu20.04-sagemaker`;

    // スクリプトパス
    const scriptS3Uri = `s3://${bucket}/public/scripts/analyze.py`;

    console.log('Starting Processing Job:', {
      processingJobName,
      modelS3Uri,
      dataS3Uri,
      outputS3Uri,
    });

    const command = new CreateProcessingJobCommand({
      ProcessingJobName: processingJobName,
      ProcessingResources: {
        ClusterConfig: {
          InstanceCount: 1,
          InstanceType: 'ml.m5.large',
          VolumeSizeInGB: 30,
        },
      },
      AppSpecification: {
        ImageUri: processingImage,
        ContainerEntrypoint: ['python3', '/opt/ml/processing/input/code/analyze.py'],
        ContainerArguments: [
          '--model-path', modelS3Uri,
          '--data-path', dataS3Uri,
          '--output-path', outputS3Uri,
          '--target-field', '1',
          '--max-samples-per-class', '10',
        ],
      },
      ProcessingInputs: [
        {
          InputName: 'model',
          S3Input: {
            S3Uri: modelS3Uri,
            LocalPath: '/opt/ml/processing/model',
            S3DataType: 'S3Prefix',
            S3InputMode: 'File',
            S3DataDistributionType: 'FullyReplicated',
          },
        },
        {
          InputName: 'data',
          S3Input: {
            S3Uri: dataS3Uri,
            LocalPath: '/opt/ml/processing/data',
            S3DataType: 'S3Prefix',
            S3InputMode: 'File',
            S3DataDistributionType: 'FullyReplicated',
          },
        },
        {
          InputName: 'code',
          S3Input: {
            S3Uri: scriptS3Uri,
            LocalPath: '/opt/ml/processing/input/code',
            S3DataType: 'S3Prefix',
            S3InputMode: 'File',
            S3DataDistributionType: 'FullyReplicated',
          },
        },
      ],
      ProcessingOutputConfig: {
        Outputs: [
          {
            OutputName: 'analysis',
            S3Output: {
              S3Uri: outputS3Uri,
              LocalPath: '/opt/ml/processing/output',
              S3UploadMode: 'EndOfJob',
            },
          },
        ],
      },
      RoleArn: sagemakerRoleArn,
      StoppingCondition: {
        MaxRuntimeInSeconds: 3600, // 1時間
      },
    });

    await sagemakerClient.send(command);

    return {
      statusCode: 200,
      headers: responseHeaders,
      body: JSON.stringify({
        processingJobName,
        status: 'InProgress',
        modelJobName,
        outputPath: outputS3Uri,
      }),
    };
  } catch (error) {
    console.error('Error starting analysis:', error);
    return {
      statusCode: 500,
      headers: responseHeaders,
      body: JSON.stringify({
        error: 'Failed to start analysis',
        details: (error as Error).message,
      }),
    };
  }
};

