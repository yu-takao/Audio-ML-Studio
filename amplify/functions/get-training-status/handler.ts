import type { Handler } from 'aws-lambda';
import { SageMakerClient, DescribeTrainingJobCommand } from '@aws-sdk/client-sagemaker';

const sagemakerClient = new SageMakerClient({});

interface StatusRequest {
  trainingJobName: string;
}

// レスポンスヘッダー（CORSはFunction URLで処理）
const responseHeaders = {
  'Content-Type': 'application/json',
  };

export const handler: Handler = async (event) => {
  console.log('Event:', JSON.stringify(event, null, 2));

  try {
    // クエリパラメータまたはボディから取得
    let trainingJobName: string | undefined;
    
    if (event.queryStringParameters?.trainingJobName) {
      trainingJobName = event.queryStringParameters.trainingJobName;
    } else if (event.body) {
      const body: StatusRequest = JSON.parse(event.body);
      trainingJobName = body.trainingJobName;
    }

    if (!trainingJobName) {
      return {
        statusCode: 400,
        headers: responseHeaders,
        body: JSON.stringify({ error: 'Missing trainingJobName' }),
      };
    }

    // SageMaker Training Jobのステータスを取得
    const command = new DescribeTrainingJobCommand({
      TrainingJobName: trainingJobName,
    });

    const response = await sagemakerClient.send(command);

    // 訓練メトリクスを抽出
    const metrics = response.FinalMetricDataList?.map((metric) => ({
      name: metric.MetricName,
      value: metric.Value,
      timestamp: metric.Timestamp?.toISOString(),
    })) || [];

    // 進捗を計算
    const startTime = response.TrainingStartTime?.getTime() || Date.now();
    const currentTime = Date.now();
    const maxRuntime = (response.StoppingCondition?.MaxRuntimeInSeconds || 7200) * 1000;
    const elapsedTime = currentTime - startTime;
    const progress = Math.min(100, (elapsedTime / maxRuntime) * 100);

    // モデル出力パスを取得
    const modelPath = response.ModelArtifacts?.S3ModelArtifacts;

    return {
      statusCode: 200,
      headers: responseHeaders,
      body: JSON.stringify({
        trainingJobName: response.TrainingJobName,
        status: response.TrainingJobStatus,
        secondaryStatus: response.SecondaryStatus,
        failureReason: response.FailureReason,
        creationTime: response.CreationTime?.toISOString(),
        trainingStartTime: response.TrainingStartTime?.toISOString(),
        trainingEndTime: response.TrainingEndTime?.toISOString(),
        lastModifiedTime: response.LastModifiedTime?.toISOString(),
        progress,
        elapsedSeconds: Math.floor(elapsedTime / 1000),
        metrics,
        modelPath,
        instanceType: response.ResourceConfig?.InstanceType,
      }),
    };
  } catch (error) {
    console.error('Error getting training status:', error);

    // ジョブが見つからない場合
    if ((error as Error).name === 'ResourceNotFoundException') {
      return {
        statusCode: 404,
        headers: responseHeaders,
        body: JSON.stringify({
          error: 'Training job not found',
          details: (error as Error).message,
        }),
      };
    }

    return {
      statusCode: 500,
      headers: responseHeaders,
      body: JSON.stringify({
        error: 'Failed to get training status',
        details: (error as Error).message,
      }),
    };
  }
};


