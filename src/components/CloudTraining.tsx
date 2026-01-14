import { useState, useCallback, useEffect } from 'react';
import { uploadData, getUrl, downloadData } from 'aws-amplify/storage';
import outputs from '../../amplify_outputs.json';
import {
  Cloud,
  CloudUpload,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Clock,
  Cpu,
  Download,
  Play,
  Server,
  Zap,
  FolderUp,
  FileAudio,
  Database,
  BarChart3,
  Target,
  TrendingUp,
  Award,
} from 'lucide-react';

interface FieldLabel {
  index: number;
  label: string;
}

interface CloudTrainingProps {
  userId: string; // Cognito認証ユーザーID
  config: {
    epochs: number;
    batchSize: number;
    learningRate: number;
    validationSplit: number;
    testSplit: number;
  };
  datasetInfo: {
    totalFiles: number;
    classes: string[];
  } | null;
  targetField: string;
  auxiliaryFields: string[];
  fieldLabels: FieldLabel[]; // フィールドラベル情報
  problemType: 'classification' | 'regression'; // 問題タイプ
  tolerance: number; // 許容範囲
  fileInfoList: { file: File; path: string; folderName: string }[];
  s3DatasetPath?: string; // S3から選択した場合のパス
  onModelReady: (modelPath: string) => void;
  onTrainingStart?: () => void;
  onUploadComplete?: (dataPath: string) => void; // アップロード完了時にメタデータを保存
}

interface TrainingJob {
  trainingJobName: string;
  status: 'InProgress' | 'Completed' | 'Failed' | 'Stopped' | 'Stopping';
  secondaryStatus?: string;
  progress: number;
  elapsedSeconds: number;
  modelPath?: string;
  metrics?: { name: string; value: number }[];
  error?: string;
}

interface UploadProgress {
  current: number;
  total: number;
  fileName: string;
  bytesUploaded: number;
  bytesTotal: number;
  status: 'uploading' | 'completed' | 'error';
}

// モデルメタデータ（訓練結果）
interface ModelMetadata {
  classes: string[];
  input_shape: number[];
  target_field: string;
  // 後方互換：新しい学習スクリプトのみ付与
  dataset?: {
    split_mode?: 'presplit' | 'random' | string;
    counts?: {
      train?: number;
      validation?: number;
      test?: number;
      total?: number;
    };
    class_distribution?: {
      train?: Record<string, number>;
      validation?: Record<string, number>;
      test?: Record<string, number>;
    };
  };
  training_params: {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    validation_split: number;
    test_split: number;
  };
  metrics: {
    test_loss: number;
    test_accuracy: number;
    final_train_loss: number;
    final_train_accuracy: number;
    final_val_loss: number;
    final_val_accuracy: number;
  };
  history: {
    loss: number[];
    accuracy: number[];
    val_loss: number[];
    val_accuracy: number[];
  };
}

export function CloudTraining({
  userId,
  config,
  datasetInfo,
  targetField,
  auxiliaryFields,
  fieldLabels,
  problemType,
  tolerance,
  fileInfoList,
  s3DatasetPath,
  onModelReady,
  onTrainingStart,
  onUploadComplete,
}: CloudTrainingProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [uploadComplete, setUploadComplete] = useState(!!s3DatasetPath); // S3パスがあれば既にアップロード済み
  const [uploadedPath, setUploadedPath] = useState<string>(s3DatasetPath || '');
  const [uploadedFileCount, setUploadedFileCount] = useState(0);
  
  const [isStartingTraining, setIsStartingTraining] = useState(false);
  const [currentJob, setCurrentJob] = useState<TrainingJob | null>(null);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);
  
  const [error, setError] = useState<string | null>(null);
  const [isAmplifyConfigured, setIsAmplifyConfigured] = useState(false);
  
  // モデルメタデータ（精度情報）
  const [modelMetadata, setModelMetadata] = useState<ModelMetadata | null>(null);
  const [isLoadingMetadata, setIsLoadingMetadata] = useState(false);
  const [showTrainingHistory, setShowTrainingHistory] = useState(false);
  
  // 過去の訓練ジョブ
  const [savedJobs, setSavedJobs] = useState<TrainingJob[]>([]);
  const [showPastJobs, setShowPastJobs] = useState(false);
  const [isRestoringJob, setIsRestoringJob] = useState(false);
  // @ts-expect-error - Used in loadTrainingHistoryFromS3
  const [isLoadingJobs, setIsLoadingJobs] = useState(true);

  // S3での訓練履歴パス
  const TRAINING_HISTORY_PATH = `public/user-data/${userId}/training-history.json`;

  /**
   * S3から訓練履歴を読み込む
   */
  const loadTrainingHistoryFromS3 = useCallback(async () => {
    if (!userId) return;
    
    setIsLoadingJobs(true);
    try {
      const result = await downloadData({ path: TRAINING_HISTORY_PATH }).result;
      const text = await result.body.text();
      const jobs: TrainingJob[] = JSON.parse(text);
      setSavedJobs(jobs);
      
      // 進行中のジョブがあれば復元
      const inProgressJob = jobs.find(j => j.status === 'InProgress');
      if (inProgressJob && !currentJob) {
        setIsRestoringJob(true);
        setCurrentJob(inProgressJob);
        setTimeout(() => setIsRestoringJob(false), 1000);
      }
    } catch (err: unknown) {
      // ファイルが存在しない場合は空配列（NotFound/NoSuchKeyエラーは正常）
      const errorName = (err as { name?: string })?.name || '';
      if (errorName === 'NotFound' || errorName === 'NoSuchKey') {
        console.log('No training history found, starting fresh');
      } else {
        console.warn('Error loading training history:', err);
      }
      setSavedJobs([]);
    } finally {
      setIsLoadingJobs(false);
    }
  }, [userId, TRAINING_HISTORY_PATH, currentJob]);

  /**
   * S3に訓練履歴を保存する
   */
  const saveTrainingHistoryToS3 = useCallback(
    async (jobs: TrainingJob[]) => {
      if (!userId) return;

      const jsonData = JSON.stringify(jobs, null, 2);

      // S3の503 Slow Down対策でリトライを追加
      const maxAttempts = 3;
      let attempt = 0;
      while (attempt < maxAttempts) {
        try {
          await uploadData({
            path: TRAINING_HISTORY_PATH,
            data: jsonData,
            options: {
              contentType: 'application/json',
            },
          }).result;
          console.log('Training history saved to S3');
          return;
        } catch (err) {
          attempt += 1;
          console.error(`Failed to save training history (attempt ${attempt}):`, err);
          if (attempt >= maxAttempts) break;
          // エクスポネンシャルバックオフ（500ms, 1s, 2s）
          const delay = 500 * Math.pow(2, attempt - 1);
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    },
    [userId, TRAINING_HISTORY_PATH]
  );

  // 保存されたジョブを読み込む（初回）
  useEffect(() => {
    if (userId) {
      loadTrainingHistoryFromS3();
    }
  }, [userId]);

  // ジョブ状態が変わったらS3に保存
  useEffect(() => {
    if (currentJob && userId) {
      setSavedJobs(prev => {
        const existing = prev.findIndex(j => j.trainingJobName === currentJob.trainingJobName);
        let updated: TrainingJob[];
        if (existing >= 0) {
          updated = [...prev];
          updated[existing] = currentJob;
        } else {
          updated = [currentJob, ...prev];
        }
        // 最新10件まで保持
        updated = updated.slice(0, 10);
        
        // S3に保存（非同期）
        saveTrainingHistoryToS3(updated);
        
        return updated;
      });
    }
  }, [currentJob, userId, saveTrainingHistoryToS3]);

  // S3パスが変更されたら状態を更新
  useEffect(() => {
    if (s3DatasetPath) {
      setUploadComplete(true);
      setUploadedPath(s3DatasetPath);
    }
  }, [s3DatasetPath]);

  // Amplify設定チェック
  useEffect(() => {
    const checkAmplifyConfig = async () => {
      try {
        await getUrl({ path: 'test-config-check' }).catch(() => {});
        setIsAmplifyConfigured(true);
      } catch {
        setIsAmplifyConfigured(false);
      }
    };
    checkAmplifyConfig();
  }, []);

  // APIエンドポイント（amplify_outputs.jsonから取得）
  const customConfig = (outputs as { custom?: { startTrainingUrl?: string; getTrainingStatusUrl?: string } }).custom;
  const START_TRAINING_URL = customConfig?.startTrainingUrl || '';
  const GET_STATUS_URL = customConfig?.getTrainingStatusUrl || '';
  
  // URLの検証（デバッグ用）
  useEffect(() => {
    if (!GET_STATUS_URL) {
      console.error('[CloudTraining] GET_STATUS_URL is not set. Check amplify_outputs.json');
      setError('APIエンドポイントが設定されていません。ページをリロードしてください。');
    } else {
      console.log('[CloudTraining] GET_STATUS_URL:', GET_STATUS_URL);
    }
  }, [GET_STATUS_URL]);

  // S3データセットを使用するかどうか
  const isUsingS3Dataset = !!s3DatasetPath;

  // userIdはpropsから取得するため、getUserIdは不要

  /**
   * データをS3にアップロード
   */
  const uploadDataToS3 = useCallback(async () => {
    if (fileInfoList.length === 0) {
      setError('アップロードするファイルがありません');
      return;
    }

    setIsUploading(true);
    setError(null);
    setUploadComplete(false);
    setUploadedFileCount(0);

    try {
      // userIdはpropsから取得
      const timestamp = Date.now();
      const basePath = `public/training-data/${userId}/${timestamp}`;

      let successCount = 0;
      let totalBytes = 0;
      let uploadedBytes = 0;

      for (const fileInfo of fileInfoList) {
        totalBytes += fileInfo.file.size;
      }

      for (let i = 0; i < fileInfoList.length; i++) {
        const fileInfo = fileInfoList[i];
        const filePath = `${basePath}/${fileInfo.path}`;
        
        setUploadProgress({
          current: i + 1,
          total: fileInfoList.length,
          fileName: fileInfo.file.name,
          bytesUploaded: uploadedBytes,
          bytesTotal: totalBytes,
          status: 'uploading',
        });

        try {
          await uploadData({
            path: filePath,
            data: fileInfo.file,
            options: {
              contentType: 'audio/wav',
              onProgress: ({ transferredBytes }) => {
                setUploadProgress(prev => prev ? {
                  ...prev,
                  bytesUploaded: uploadedBytes + transferredBytes,
                } : null);
              },
            },
          }).result;

          uploadedBytes += fileInfo.file.size;
          successCount++;
          setUploadedFileCount(successCount);
        } catch (uploadErr) {
          console.error(`Failed to upload ${fileInfo.file.name}:`, uploadErr);
        }
      }

      if (successCount === 0) {
        throw new Error('すべてのファイルのアップロードに失敗しました');
      }

      setUploadedPath(basePath);
      setUploadComplete(true);
      setUploadProgress({
        current: fileInfoList.length,
        total: fileInfoList.length,
        fileName: '',
        bytesUploaded: totalBytes,
        bytesTotal: totalBytes,
        status: 'completed',
      });

      console.log(`Uploaded ${successCount}/${fileInfoList.length} files to ${basePath}`);
      
      // メタデータを保存
      if (onUploadComplete) {
        onUploadComplete(basePath);
      }
    } catch (err) {
      const errorMessage = (err as Error).message;
      
      if (errorMessage.includes('No credentials') || errorMessage.includes('not configured')) {
        setError('Amplifyが設定されていません。');
      } else {
        setError('アップロードに失敗しました: ' + errorMessage);
      }
      
      setUploadProgress(prev => prev ? { ...prev, status: 'error' } : null);
    } finally {
      setIsUploading(false);
    }
  }, [fileInfoList, userId, onUploadComplete]);

  /**
   * モデルメタデータをS3から取得
   */
  const loadModelMetadata = useCallback(async (modelPath: string) => {
    setIsLoadingMetadata(true);
    try {
      // modelPathは s3://bucket/models/userId/jobName/output/model.tar.gz の形式
      // メタデータは同じディレクトリに保存されている
      // SageMakerの出力は /output/ 以下に展開される
      
      // バケット名とパスを抽出
      const pathMatch = modelPath.match(/s3:\/\/([^\/]+)\/(.+)/);
      if (!pathMatch) {
        console.error('Invalid model path:', modelPath);
        return;
      }
      
      const [, , fullPath] = pathMatch;
      // model.tar.gz を model_metadata.json に置き換え
      const metadataPath = fullPath.replace('model.tar.gz', 'model_metadata.json');
      
      console.log('Loading metadata from:', metadataPath);
      
      const downloadResult = await downloadData({
        path: metadataPath,
      }).result;
      
      const text = await downloadResult.body.text();
      const metadata: ModelMetadata = JSON.parse(text);
      
      setModelMetadata(metadata);
      console.log('Model metadata loaded:', metadata);
    } catch (err) {
      console.error('Failed to load model metadata:', err);
      // メタデータの取得に失敗してもエラーは表示しない（オプション機能のため）
    } finally {
      setIsLoadingMetadata(false);
    }
  }, []);

  /**
   * 過去のジョブを選択して表示
   */
  const selectPastJob = useCallback(async (job: TrainingJob) => {
    setCurrentJob(job);
    setModelMetadata(null);
    setShowPastJobs(false);
    
    // 完了済みの場合はメタデータを読み込む
    if (job.status === 'Completed' && job.modelPath) {
      await loadModelMetadata(job.modelPath);
    }
    // 進行中のジョブの場合はuseEffectでポーリングを開始
  }, [loadModelMetadata]);

  /**
   * 保存されたジョブをクリア
   */
  const clearSavedJobs = useCallback(async () => {
    // S3に空の履歴を保存
    await saveTrainingHistoryToS3([]);
    setSavedJobs([]);
    if (currentJob?.status !== 'InProgress') {
      setCurrentJob(null);
      setModelMetadata(null);
    }
  }, [currentJob, saveTrainingHistoryToS3]);

  /**
   * 新しい訓練を開始（現在のジョブをリセット）
   */
  const startNewTraining = useCallback(() => {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
    setCurrentJob(null);
    setModelMetadata(null);
    setShowTrainingHistory(false);
  }, [pollingInterval]);

  /**
   * クラウド訓練を開始
   */
  const startCloudTraining = useCallback(async () => {
    if (!uploadedPath) return;

    if (!START_TRAINING_URL) {
      setError('APIエンドポイントが設定されていません。Amplify Sandboxをデプロイしてください。');
      return;
    }

    setIsStartingTraining(true);
    setError(null);
    onTrainingStart?.();

    try {
      // userIdはpropsから取得
      const response = await fetch(START_TRAINING_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId,
          config: {
            ...config,
            dataPath: uploadedPath,
            targetField,
            auxiliaryFields,
            fieldLabels, // フィールドラベル情報を追加
            problemType, // 問題タイプを追加
            tolerance, // 許容範囲を追加
            classNames: datasetInfo?.classes || [],
            inputHeight: 128,
            inputWidth: 128,
          },
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `API error: ${response.status}`);
      }

      const result = await response.json();
      
      setCurrentJob({
        trainingJobName: result.trainingJobName,
        status: 'InProgress',
        progress: 0,
        elapsedSeconds: 0,
      });

      startPolling(result.trainingJobName);
    } catch (err) {
      setError('訓練の開始に失敗しました: ' + (err as Error).message);
    } finally {
      setIsStartingTraining(false);
    }
  }, [uploadedPath, datasetInfo, config, targetField, auxiliaryFields, fieldLabels, problemType, tolerance, START_TRAINING_URL, userId, onTrainingStart]);

  /**
   * 訓練ステータスをポーリング
   */
  const startPolling = useCallback((trainingJobName: string) => {
    if (pollingInterval) {
      clearInterval(pollingInterval);
    }

    const interval = setInterval(async () => {
      try {
        if (!GET_STATUS_URL) {
          console.error('[CloudTraining] GET_STATUS_URL is not set, cannot poll status');
          clearInterval(interval);
          setPollingInterval(null);
          setError('APIエンドポイントが設定されていません。ページをリロードしてください。');
          return;
        }
        
        const url = `${GET_STATUS_URL}?trainingJobName=${encodeURIComponent(trainingJobName)}`;
        console.log('[CloudTraining] Polling status from:', url);
        
        const response = await fetch(url);

        if (!response.ok) {
          throw new Error(`API error: ${response.status} ${response.statusText}`);
        }

        const status = await response.json();

        setCurrentJob({
          trainingJobName: status.trainingJobName,
          status: status.status,
          secondaryStatus: status.secondaryStatus,
          progress: status.progress,
          elapsedSeconds: status.elapsedSeconds,
          modelPath: status.modelPath,
          metrics: status.metrics,
        });

        if (status.status === 'Completed' || status.status === 'Failed' || status.status === 'Stopped') {
          clearInterval(interval);
          setPollingInterval(null);

          if (status.status === 'Completed' && status.modelPath) {
            // モデルメタデータを取得
            loadModelMetadata(status.modelPath);
            onModelReady(status.modelPath);
          }
        }
      } catch (err) {
        console.error('Failed to poll status:', err);
      }
    }, 5000);

    setPollingInterval(interval);
  }, [pollingInterval, GET_STATUS_URL, onModelReady]);

  // 進行中のジョブが選択されたらポーリングを開始
  useEffect(() => {
    if (currentJob?.status === 'InProgress' && !pollingInterval) {
      startPolling(currentJob.trainingJobName);
    }
  }, [currentJob?.trainingJobName, currentJob?.status, pollingInterval, startPolling]);

  // クリーンアップ
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  const StatusBadge = ({ status }: { status: string }) => {
    const colors: Record<string, string> = {
      InProgress: 'bg-blue-500/20 text-blue-400 border-blue-500/50',
      Completed: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50',
      Failed: 'bg-red-500/20 text-red-400 border-red-500/50',
      Stopped: 'bg-zinc-500/20 text-zinc-400 border-zinc-500/50',
      Stopping: 'bg-amber-500/20 text-amber-400 border-amber-500/50',
    };

    return (
      <span className={`px-2 py-1 text-xs rounded-full border ${colors[status] || colors.InProgress}`}>
        {status}
      </span>
    );
  };

  return (
    <div className="bg-gradient-to-r from-sky-500/10 to-cyan-500/10 rounded-xl border border-sky-500/30 p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-sky-500/20">
          <Cloud className="w-5 h-5 text-sky-400" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">クラウド訓練 (AWS SageMaker)</h2>
          <p className="text-sm text-zinc-400">
            GPUインスタンスで高速に訓練を実行します
          </p>
        </div>
      </div>

      {/* Amplify未設定の警告 */}
      {!isAmplifyConfigured && (
        <div className="mb-4 p-3 bg-amber-500/10 border border-amber-500/50 rounded-lg flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <div className="text-amber-300 font-medium">ローカル開発モード</div>
            <div className="text-sm text-amber-400/80">
              Amplifyが設定されていません。クラウド訓練を使用するには、
              <code className="bg-zinc-800 px-1 rounded">npx ampx sandbox</code> を実行してください。
            </div>
          </div>
        </div>
      )}

      {/* メリット説明 */}
      <div className="mb-4 grid grid-cols-3 gap-3">
        <div className="bg-zinc-900/50 rounded-lg p-3 flex items-center gap-2">
          <Zap className="w-4 h-4 text-amber-400" />
          <span className="text-sm text-zinc-300">高速GPU処理</span>
        </div>
        <div className="bg-zinc-900/50 rounded-lg p-3 flex items-center gap-2">
          <Server className="w-4 h-4 text-emerald-400" />
          <span className="text-sm text-zinc-300">オンデマンド課金</span>
        </div>
        <div className="bg-zinc-900/50 rounded-lg p-3 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-violet-400" />
          <span className="text-sm text-zinc-300">PC負荷なし</span>
        </div>
      </div>

      {/* 過去の訓練ジョブ */}
      {savedJobs.length > 0 && (
        <div className="mb-4">
          <button
            onClick={() => setShowPastJobs(!showPastJobs)}
            className="flex items-center gap-2 text-sm text-sky-400 hover:text-sky-300 transition-colors"
          >
            <Database className="w-4 h-4" />
            {showPastJobs ? '過去の訓練を閉じる' : `過去の訓練を表示 (${savedJobs.length}件)`}
          </button>
          
          {showPastJobs && (
            <div className="mt-2 space-y-2">
              {savedJobs.map((job) => (
                <div
                  key={job.trainingJobName}
                  className={`p-3 rounded-lg border transition-all cursor-pointer ${
                    currentJob?.trainingJobName === job.trainingJobName
                      ? 'bg-sky-500/20 border-sky-500/50'
                      : 'bg-zinc-800/50 border-zinc-700 hover:border-zinc-600'
                  }`}
                  onClick={() => selectPastJob(job)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {job.status === 'Completed' ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      ) : job.status === 'Failed' ? (
                        <AlertCircle className="w-4 h-4 text-red-400" />
                      ) : job.status === 'InProgress' ? (
                        <Loader2 className="w-4 h-4 text-sky-400 animate-spin" />
                      ) : (
                        <Clock className="w-4 h-4 text-zinc-400" />
                      )}
                      <span className="text-sm text-white font-mono truncate max-w-[200px]">
                        {job.trainingJobName}
                      </span>
                    </div>
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      job.status === 'Completed' ? 'bg-emerald-500/20 text-emerald-400' :
                      job.status === 'Failed' ? 'bg-red-500/20 text-red-400' :
                      job.status === 'InProgress' ? 'bg-sky-500/20 text-sky-400' :
                      'bg-zinc-600/20 text-zinc-400'
                    }`}>
                      {job.status}
                    </span>
                  </div>
                </div>
              ))}
              
              <div className="flex gap-2 mt-2">
                <button
                  onClick={clearSavedJobs}
                  className="text-xs text-zinc-500 hover:text-zinc-400 transition-colors"
                >
                  履歴をクリア
                </button>
                {currentJob && (
                  <button
                    onClick={startNewTraining}
                    className="text-xs text-sky-400 hover:text-sky-300 transition-colors"
                  >
                    新しい訓練を開始
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* 復元中の表示 */}
      {isRestoringJob && (
        <div className="mb-4 p-3 bg-sky-500/10 border border-sky-500/50 rounded-lg flex items-center gap-2">
          <Loader2 className="w-4 h-4 text-sky-400 animate-spin" />
          <span className="text-sky-300 text-sm">前回の訓練を復元中...</span>
        </div>
      )}

      {/* エラー表示 */}
      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/50 rounded-lg flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <span className="text-red-300 text-sm">{error}</span>
        </div>
      )}

      {/* ステップ1: データ準備 */}
      <div className="mb-4 p-4 bg-zinc-900/50 rounded-lg">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={`w-6 h-6 rounded-full flex items-center justify-center ${
              uploadComplete ? 'bg-emerald-500' : 'bg-zinc-700'
            }`}>
              {uploadComplete ? (
                <CheckCircle2 className="w-4 h-4 text-white" />
              ) : (
                <span className="text-white text-sm">1</span>
              )}
            </div>
            <span className="text-white font-medium">
              {isUsingS3Dataset ? 'S3データセット' : 'データをS3にアップロード'}
            </span>
          </div>
          
          {/* S3データセット使用時 */}
          {isUsingS3Dataset && (
            <div className="flex items-center gap-2 text-sm text-emerald-400">
              <Database className="w-4 h-4" />
              <span>選択済み</span>
          </div>
          )}
          
          {/* ローカルアップロード時 */}
          {!isUsingS3Dataset && !uploadComplete && (
            <button
              onClick={uploadDataToS3}
              disabled={isUploading || fileInfoList.length === 0}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                isUploading || fileInfoList.length === 0
                  ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                  : 'bg-sky-500 text-white hover:bg-sky-600'
              }`}
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  アップロード中...
                </>
              ) : (
                <>
                  <CloudUpload className="w-4 h-4" />
                  アップロード開始
                </>
              )}
            </button>
          )}
        </div>

        {/* S3パス表示 */}
        {isUsingS3Dataset && (
          <div className="mt-2 text-xs text-zinc-500 font-mono bg-zinc-800 rounded px-2 py-1">
            {s3DatasetPath}
          </div>
        )}

        {/* アップロード対象の情報（ローカルの場合） */}
        {!isUsingS3Dataset && !uploadComplete && !isUploading && fileInfoList.length > 0 && (
          <div className="flex items-center gap-4 text-sm text-zinc-400 mt-2">
            <div className="flex items-center gap-1">
              <FileAudio className="w-4 h-4" />
              <span>{fileInfoList.length} ファイル</span>
            </div>
            <div className="flex items-center gap-1">
              <FolderUp className="w-4 h-4" />
              <span>{formatBytes(fileInfoList.reduce((sum, f) => sum + f.file.size, 0))}</span>
            </div>
          </div>
        )}

        {/* アップロード進捗 */}
        {uploadProgress && uploadProgress.status === 'uploading' && (
          <div className="mt-3 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-zinc-400 truncate max-w-[200px]">{uploadProgress.fileName}</span>
              <span className="text-zinc-400">
                {uploadProgress.current} / {uploadProgress.total}
              </span>
            </div>
            <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-sky-500 transition-all"
                style={{ width: `${(uploadProgress.current / uploadProgress.total) * 100}%` }}
              />
            </div>
            <div className="text-xs text-zinc-500">
              {formatBytes(uploadProgress.bytesUploaded)} / {formatBytes(uploadProgress.bytesTotal)}
            </div>
          </div>
        )}

        {!isUsingS3Dataset && uploadComplete && (
          <div className="mt-2 text-sm text-emerald-400 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" />
            {uploadedFileCount} ファイルをアップロード完了
          </div>
        )}
      </div>

      {/* ステップ2: 訓練開始 */}
      <div className="mb-4 p-4 bg-zinc-900/50 rounded-lg">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={`w-6 h-6 rounded-full flex items-center justify-center ${
              currentJob?.status === 'Completed' ? 'bg-emerald-500' : 
              currentJob ? 'bg-sky-500' : 'bg-zinc-700'
            }`}>
              {currentJob?.status === 'Completed' ? (
                <CheckCircle2 className="w-4 h-4 text-white" />
              ) : (
                <span className="text-white text-sm">2</span>
              )}
            </div>
            <span className="text-white font-medium">クラウドで訓練</span>
          </div>
          {!currentJob && (
            <button
              onClick={startCloudTraining}
              disabled={!uploadComplete || isStartingTraining || !START_TRAINING_URL}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                !uploadComplete || isStartingTraining || !START_TRAINING_URL
                  ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                  : 'bg-emerald-500 text-white hover:bg-emerald-600'
              }`}
            >
              {isStartingTraining ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  開始中...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  訓練開始
                </>
              )}
            </button>
          )}
        </div>

        {/* 訓練ジョブ情報 */}
        {currentJob && (
          <div className="mt-3 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <StatusBadge status={currentJob.status} />
                <span className="text-sm text-zinc-400">{currentJob.secondaryStatus}</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-zinc-400">
                <Clock className="w-4 h-4" />
                {formatTime(currentJob.elapsedSeconds)}
              </div>
            </div>

            {currentJob.status === 'InProgress' && (
              <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-sky-500 to-cyan-500 transition-all"
                  style={{ width: `${currentJob.progress}%` }}
                />
              </div>
            )}

            {currentJob.status === 'Completed' && currentJob.metrics && (
              <div className="grid grid-cols-2 gap-2">
                {currentJob.metrics.map((metric) => (
                  <div key={metric.name} className="bg-zinc-800 rounded-lg p-2">
                    <div className="text-xs text-zinc-500">{metric.name}</div>
                    <div className="text-white font-medium">{metric.value.toFixed(4)}</div>
                  </div>
                ))}
              </div>
            )}

            {currentJob.status === 'Failed' && (
              <div className="text-sm text-red-400">
                訓練に失敗しました。CloudWatch Logsでエラーを確認してください。
              </div>
            )}
          </div>
        )}
      </div>

      {/* ステップ3: モデル完成・精度情報 */}
      {currentJob?.status === 'Completed' && currentJob.modelPath && (
        <div className="p-4 bg-zinc-900/50 rounded-lg space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full flex items-center justify-center bg-emerald-500">
                <CheckCircle2 className="w-4 h-4 text-white" />
              </div>
              <span className="text-white font-medium">モデル完成</span>
            </div>
            <button
              onClick={() => onModelReady(currentJob.modelPath!)}
              className="flex items-center gap-2 px-4 py-2 bg-violet-500 text-white rounded-lg hover:bg-violet-600 transition-all"
            >
              <Download className="w-4 h-4" />
              モデルを読み込む
            </button>
          </div>

          {/* 精度情報 */}
          {isLoadingMetadata ? (
            <div className="flex items-center gap-2 text-zinc-400 text-sm">
              <Loader2 className="w-4 h-4 animate-spin" />
              精度情報を読み込み中...
            </div>
          ) : modelMetadata ? (
            <div className="space-y-4">
              {/* メイン精度指標 */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-gradient-to-br from-emerald-500/20 to-emerald-600/10 border border-emerald-500/30 rounded-xl p-3">
                  <div className="flex items-center gap-2 text-emerald-400 text-xs mb-1">
                    <Target className="w-3 h-3" />
                    テスト精度
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {(modelMetadata.metrics.test_accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-gradient-to-br from-sky-500/20 to-sky-600/10 border border-sky-500/30 rounded-xl p-3">
                  <div className="flex items-center gap-2 text-sky-400 text-xs mb-1">
                    <TrendingUp className="w-3 h-3" />
                    検証精度
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {(modelMetadata.metrics.final_val_accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-gradient-to-br from-amber-500/20 to-amber-600/10 border border-amber-500/30 rounded-xl p-3">
                  <div className="flex items-center gap-2 text-amber-400 text-xs mb-1">
                    <Award className="w-3 h-3" />
                    訓練精度
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {(modelMetadata.metrics.final_train_accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-gradient-to-br from-rose-500/20 to-rose-600/10 border border-rose-500/30 rounded-xl p-3">
                  <div className="flex items-center gap-2 text-rose-400 text-xs mb-1">
                    <BarChart3 className="w-3 h-3" />
                    テスト損失
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {modelMetadata.metrics.test_loss.toFixed(4)}
                  </div>
                </div>
              </div>

              {/* クラス数 */}
              <div className="flex items-center gap-4 text-sm text-zinc-400">
                <span>クラス数: <span className="text-white font-medium">{modelMetadata.classes.length}</span></span>
                <span>エポック数: <span className="text-white font-medium">{modelMetadata.history.accuracy.length}</span></span>
              </div>

              {/* 訓練履歴グラフ */}
              <div>
                <button
                  onClick={() => setShowTrainingHistory(!showTrainingHistory)}
                  className="flex items-center gap-2 text-sm text-sky-400 hover:text-sky-300 transition-colors"
                >
                  <BarChart3 className="w-4 h-4" />
                  {showTrainingHistory ? '訓練履歴を閉じる' : '訓練履歴を表示'}
                </button>
                
                {showTrainingHistory && (
                  <div className="mt-3 space-y-4">
                    {/* 精度の推移 */}
                    <div className="bg-zinc-800/50 rounded-lg p-4">
                      <h4 className="text-sm font-medium text-white mb-3">精度の推移</h4>
                      <div className="h-32 flex items-end gap-px">
                        {modelMetadata.history.accuracy.map((acc, i) => {
                          const valAcc = modelMetadata.history.val_accuracy[i];
                          return (
                            <div key={i} className="flex-1 flex flex-col gap-px items-center">
                              <div
                                className="w-full bg-sky-500/60 rounded-t"
                                style={{ height: `${valAcc * 100}%` }}
                                title={`検証: ${(valAcc * 100).toFixed(1)}%`}
                              />
                              <div
                                className="w-full bg-emerald-500/60 rounded-t"
                                style={{ height: `${acc * 100}%` }}
                                title={`訓練: ${(acc * 100).toFixed(1)}%`}
                              />
                            </div>
                          );
                        })}
                      </div>
                      <div className="flex justify-between text-xs text-zinc-500 mt-2">
                        <span>エポック 1</span>
                        <div className="flex items-center gap-4">
                          <span className="flex items-center gap-1">
                            <div className="w-2 h-2 bg-emerald-500 rounded" />
                            訓練
                          </span>
                          <span className="flex items-center gap-1">
                            <div className="w-2 h-2 bg-sky-500 rounded" />
                            検証
                          </span>
                        </div>
                        <span>エポック {modelMetadata.history.accuracy.length}</span>
                      </div>
                    </div>

                    {/* 損失の推移 */}
                    <div className="bg-zinc-800/50 rounded-lg p-4">
                      <h4 className="text-sm font-medium text-white mb-3">損失の推移</h4>
                      <div className="h-32 flex items-end gap-px">
                        {modelMetadata.history.loss.map((loss, i) => {
                          const valLoss = modelMetadata.history.val_loss[i];
                          const maxLoss = Math.max(...modelMetadata.history.loss, ...modelMetadata.history.val_loss);
                          return (
                            <div key={i} className="flex-1 flex flex-col gap-px items-center">
                              <div
                                className="w-full bg-rose-500/60 rounded-t"
                                style={{ height: `${(valLoss / maxLoss) * 100}%` }}
                                title={`検証: ${valLoss.toFixed(4)}`}
                              />
                              <div
                                className="w-full bg-amber-500/60 rounded-t"
                                style={{ height: `${(loss / maxLoss) * 100}%` }}
                                title={`訓練: ${loss.toFixed(4)}`}
                              />
                            </div>
                          );
                        })}
                      </div>
                      <div className="flex justify-between text-xs text-zinc-500 mt-2">
                        <span>エポック 1</span>
                        <div className="flex items-center gap-4">
                          <span className="flex items-center gap-1">
                            <div className="w-2 h-2 bg-amber-500 rounded" />
                            訓練
                          </span>
                          <span className="flex items-center gap-1">
                            <div className="w-2 h-2 bg-rose-500 rounded" />
                            検証
                          </span>
                        </div>
                        <span>エポック {modelMetadata.history.loss.length}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </div>
      )}

      {/* コスト目安 */}
      <div className="mt-4 text-xs text-zinc-500 flex items-center gap-1">
        <AlertCircle className="w-3 h-3" />
        <span>
          推定コスト: 約 $0.10-0.30 / 訓練（ml.g4dn.xlarge, 10-30分）
        </span>
      </div>
    </div>
  );
}
