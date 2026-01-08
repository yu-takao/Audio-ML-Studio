import type { Handler } from 'aws-lambda';
import { SageMakerClient, DescribeProcessingJobCommand } from '@aws-sdk/client-sagemaker';

const sagemakerClient = new SageMakerClient({});

const responseHeaders = {
  'Content-Type': 'application/json',
};

export const handler: Handler = async (event) => {
  console.log('Event:', JSON.stringify(event, null, 2));

  try {
    const body = JSON.parse(event.body || '{}');
    const { processingJobName } = body;

    if (!processingJobName) {
      return {
        statusCode: 400,
        headers: responseHeaders,
        body: JSON.stringify({ error: 'Missing processingJobName' }),
      };
    }

    const command = new DescribeProcessingJobCommand({
      ProcessingJobName: processingJobName,
    });

    const response = await sagemakerClient.send(command);

    // ステータス情報を整形
    const status = {
      jobName: response.ProcessingJobName,
      status: response.ProcessingJobStatus,
      createdAt: response.CreationTime,
      startedAt: response.ProcessingStartTime,
      endedAt: response.ProcessingEndTime,
      failureReason: response.FailureReason,
      // 出力パスを返す
      outputPath: response.ProcessingOutputConfig?.Outputs?.[0]?.S3Output?.S3Uri,
    };

    return {
      statusCode: 200,
      headers: responseHeaders,
      body: JSON.stringify(status),
    };
  } catch (error) {
    console.error('Error getting evaluation status:', error);

    // ジョブが見つからない場合
    if ((error as any).name === 'ValidationException') {
      return {
        statusCode: 404,
        headers: responseHeaders,
        body: JSON.stringify({
          error: 'Processing job not found',
          details: (error as Error).message,
        }),
      };
    }

    return {
      statusCode: 500,
      headers: responseHeaders,
      body: JSON.stringify({
        error: 'Failed to get evaluation status',
        details: (error as Error).message,
      }),
    };
  }
};

