import { useState, useCallback, useEffect } from 'react';
import { list, uploadData, downloadData } from 'aws-amplify/storage';
import { fetchAuthSession } from 'aws-amplify/auth';
import { useFileSystem } from '../hooks/useFileSystem';
import { MetadataConfig } from './MetadataConfig';
import { InferenceResults } from './InferenceResults';
import {
  analyzeFilenames,
  generateClassLabel,
  type ParsedMetadata,
  type TargetFieldConfig,
  type AuxiliaryFieldConfig,
} from '../utils/metadataParser';
import {
  FolderOpen,
  Brain,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Upload,
  Play,
  RefreshCw,
  Calendar,
  Target,
  ChevronRight,
} from 'lucide-react';
import outputs from '../../amplify_outputs.json';

interface ModelEvaluationProps {
  userId: string;
}

// フィールドラベル情報
interface FieldLabel {
  index: number;
  label: string;
}

// 保存されているモデル情報
interface SavedModel {
  path: string;
  name: string;
  createdAt: Date;
  classes?: string[];
  targetField?: string;
  auxiliaryFields?: AuxiliaryFieldConfig[];
  fieldLabels?: FieldLabel[]; // フィールドラベル情報
  problemType?: 'classification' | 'regression';
  tolerance?: number;
}

// 評価ジョブのステータス
type JobStatus = 'InProgress' | 'Completed' | 'Failed' | 'Stopping' | 'Stopped';

// ステップ定義
type Step = 'select-model' | 'select-data' | 'configure-metadata' | 'running' | 'results';

interface EvaluationMetrics {
  accuracy: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  confusion_matrix?: number[][];
  class_metrics?: Array<{
    class_name: string;
    precision: number;
    recall: number;
    f1_score: number;
    support: number;
  }>;
  // 回帰用の指標
  mae?: number;
  mse?: number;
  rmse?: number;
  r2_score?: number;
  tolerance?: number;
  accuracy_with_tolerance?: number;
  problem_type?: 'classification' | 'regression';
}

interface FilePrediction {
  filename: string;
  true_label: string;
  predicted_label: string;
  confidence: string;
  correct: boolean;
}

export function ModelEvaluation({ userId }: ModelEvaluationProps) {
  const [currentStep, setCurrentStep] = useState<Step>('select-model');

  // モデル選択
  const [savedModels, setSavedModels] = useState<SavedModel[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [selectedModel, setSelectedModel] = useState<SavedModel | null>(null);

  // データ選択（ファイルシステム）
  const {
    inputFolder,
    wavFiles,
    isLoading: isLoadingFiles,
    error: _fileError,
    selectInputFolder,
  } = useFileSystem();

  // メタデータ設定
  const [metadata, setMetadata] = useState<ParsedMetadata | null>(null);
  const [targetConfig, setTargetConfig] = useState<TargetFieldConfig | null>(null);
  const [auxiliaryFields, setAuxiliaryFields] = useState<AuxiliaryFieldConfig[]>([]);
  const [problemType, setProblemType] = useState<'classification' | 'regression'>('classification');

  // 評価ジョブ
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [jobName, setJobName] = useState('');
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 評価結果
  const [evaluationMetrics, setEvaluationMetrics] = useState<EvaluationMetrics | null>(null);
  const [filePredictions, setFilePredictions] = useState<FilePrediction[]>([]);

  // Lambda関数のURL
  const startEvaluationUrl = outputs.custom?.startEvaluationUrl;
  const getEvaluationStatusUrl = outputs.custom?.getEvaluationStatusUrl;

  // 保存済みモデルのリストを取得
  const loadSavedModels = useCallback(async () => {
    setIsLoadingModels(true);
    try {
      // ユーザー専用のモデルパスを取得
      const result = await list({
        path: `models/${userId}/`,
        options: {
          listAll: true,
        },
      });

      // model.tar.gz をモデルの存在判定に利用
      const models: SavedModel[] = [];

      for (const item of result.items) {
        if (item.path.endsWith('model.tar.gz')) {
          // models/{userId}/{jobName}/output/model.tar.gz のパターン
          const parts = item.path.split('/');
          const jobName = parts.length >= 4 ? parts[2] : 'unknown';
          const modelPath = parts.slice(0, parts.length - 1).join('/'); // models/{userId}/{jobName}/output

          // メタデータを読み込む
          try {
            const metadataPath = `${modelPath}/model_metadata.json`;
            const metadataResult = await downloadData({ path: metadataPath }).result;
            const metadataText = await metadataResult.body.text();
            const metadata = JSON.parse(metadataText);

            models.push({
              path: modelPath,
              name: jobName,
              createdAt: item.lastModified || new Date(),
              classes: metadata.classes || [],
              targetField: metadata.target_field,
              auxiliaryFields: metadata.auxiliary_fields || [],
              fieldLabels: metadata.field_labels || [], // フィールドラベル情報
              problemType: metadata.problem_type || 'classification',
              tolerance: metadata.tolerance || 0,
            });
          } catch (err) {
            // メタデータがない場合もモデルは表示
            console.warn(`Metadata not found for ${jobName}:`, err);
            models.push({
              path: modelPath,
              name: jobName,
              createdAt: item.lastModified || new Date(),
            });
          }
        }
      }

      models.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
      setSavedModels(models);
    } catch (err) {
      console.error('Error loading models:', err);
      setError('モデルの読み込みに失敗しました');
    } finally {
      setIsLoadingModels(false);
    }
  }, [userId]);

  useEffect(() => {
    loadSavedModels();
  }, [loadSavedModels]);

  // データ選択時にメタデータを解析し、選択したモデルのフィールドラベルを適用
  useEffect(() => {
    if (wavFiles.length > 0) {
      const filenames = wavFiles.map(f => f.name);
      const analyzed = analyzeFilenames(filenames, undefined, '_');

      // 選択したモデルのフィールドラベルを適用
      if (selectedModel?.fieldLabels && selectedModel.fieldLabels.length > 0) {
        const updatedFields = analyzed.fields.map(field => {
          const savedLabel = selectedModel.fieldLabels?.find(fl => fl.index === field.index);
          return savedLabel ? { ...field, label: savedLabel.label } : field;
        });
        analyzed.fields = updatedFields;
      }

      setMetadata(analyzed);

      // 選択したモデルのターゲットフィールドを自動選択
      if (selectedModel?.targetField !== undefined) {
        const targetIndex = parseInt(selectedModel.targetField, 10);
        const targetField = analyzed.fields.find(f => f.index === targetIndex);
        if (targetField) {
          setTargetConfig({
            fieldIndex: targetIndex,
            fieldName: targetField.label,
            useAsTarget: true,
            groupingMode: 'individual',
            problemType: selectedModel.problemType || 'classification',
            tolerance: selectedModel.tolerance || 0,
          });
          // problemTypeのstateも更新
          if (selectedModel.problemType) {
            setProblemType(selectedModel.problemType);
          }
        }
      }
    }
  }, [wavFiles, selectedModel]);

  // モデルを選択
  const handleSelectModel = (model: SavedModel) => {
    setSelectedModel(model);
    setCurrentStep('select-data');
  };

  // データフォルダを選択
  const handleSelectData = async () => {
    await selectInputFolder();
    if (wavFiles.length > 0) {
      setCurrentStep('configure-metadata');
    }
  };

  // 評価を開始
  const handleStartEvaluation = async () => {
    if (!selectedModel || !targetConfig || wavFiles.length === 0) {
      setError('モデル、データ、メタデータ設定が必要です');
      return;
    }

    if (targetConfig.fieldIndex === undefined || targetConfig.fieldIndex === null) {
      setError('ターゲットフィールドが設定されていません');
      return;
    }

    if (!startEvaluationUrl) {
      setError('評価機能のエンドポイントが設定されていません');
      return;
    }

    setCurrentStep('running');
    setError(null);
    setIsUploading(true);

    try {
      // 認証セッションからIdentity IDを取得
      const session = await fetchAuthSession();
      const identityId = session.identityId;
      if (!identityId) {
        throw new Error('Identity IDを取得できませんでした');
      }

      // 1. データをS3にアップロード
      const timestamp = Date.now();
      const dataPath = `evaluation/temp/${identityId}/eval-${timestamp}`;

      let uploadedCount = 0;
      for (const file of wavFiles) {
        const blob = new Blob([await file.file.arrayBuffer()], { type: 'audio/wav' });
        await uploadData({
          path: ({ identityId }) => `evaluation/temp/${identityId}/eval-${timestamp}/${file.name}`,
          data: blob,
        }).result;

        uploadedCount++;
        setUploadProgress(Math.floor((uploadedCount / wavFiles.length) * 100));
      }

      setIsUploading(false);

      // 2. クラス名を生成
      const classNames = new Set<string>();
      for (const file of wavFiles) {
        if (!targetConfig) continue;
        const label = generateClassLabel(file.name, undefined, targetConfig, '_');
        classNames.add(label);
      }

      // 3. SageMaker評価ジョブを開始（Identity IDを使用）
      const response = await fetch(startEvaluationUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: identityId, // Identity IDを使用（S3アクセス制御と一致）
          config: {
            dataPath,
            modelPath: selectedModel.path,
            targetField: targetConfig.fieldIndex,
            auxiliaryFields: auxiliaryFields.map(f => f.fieldIndex),
            classNames: Array.from(classNames).sort(),
            inputHeight: 128,
            inputWidth: 128,
            problemType: targetConfig.problemType || problemType || 'classification',
            tolerance: targetConfig.tolerance || 0,
          },
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Lambda error response:', errorText);
        throw new Error(`評価ジョブの開始に失敗しました: ${response.status} ${errorText}`);
      }

      const result = await response.json();
      setJobName(result.processingJobName);
      setJobStatus('InProgress');

      // ポーリング開始
      pollJobStatus(result.processingJobName);
    } catch (err) {
      console.error('Error starting evaluation:', err);
      setError((err as Error).message || '評価の開始に失敗しました');
      setCurrentStep('configure-metadata');
      setIsUploading(false);
    }
  };

  // ジョブステータスをポーリング
  const pollJobStatus = useCallback(
    async (processingJobName: string) => {
      const poll = async () => {
        try {
          const response = await fetch(getEvaluationStatusUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ processingJobName }),
          });

          if (!response.ok) {
            throw new Error('ステータス取得に失敗しました');
          }

          const status = await response.json();
          setJobStatus(status.status);

          if (status.status === 'Completed') {
            // 結果を読み込む
            await loadEvaluationResults(status.outputPath);
            setCurrentStep('results');
            return;
          } else if (status.status === 'Failed') {
            setError(`評価ジョブが失敗しました: ${status.failureReason || '不明なエラー'}`);
            setCurrentStep('configure-metadata');
            return;
          }

          // 継続してポーリング
          setTimeout(poll, 10000); // 10秒ごと
        } catch (err) {
          console.error('Error polling status:', err);
          setError('ステータスの取得に失敗しました');
        }
      };

      poll();
    },
    [getEvaluationStatusUrl]
  );

  // 評価結果を読み込む
  const loadEvaluationResults = async (outputPath: string) => {
    try {
      console.log('Loading results from outputPath:', outputPath);

      // S3パスからバケット名を除去: s3://bucket/evaluation/results/identity-id/job-name/
      // → evaluation/results/identity-id/job-name/
      let metricsPath = outputPath.replace('s3://', '').replace(/^[^/]+\//, '');
      console.log('After bucket removal:', metricsPath);

      // Identity ID部分を除去してジョブ名だけを残す
      // evaluation/results/identity-id/job-name/ → job-name/
      const match = metricsPath.match(/^evaluation\/results\/[^/]+\/(.+)$/);
      if (!match) {
        throw new Error(`Invalid output path format: ${outputPath}`);
      }
      const jobPath = match[1]; // audio-eval-ap-north-2026-01-09T10-13-10-279Z/
      console.log('Job path:', jobPath);
      console.log('Full metrics file path:', `evaluation/results/{identityId}/${jobPath}metrics.json`);

      // Amplify Storageのpath関数を使用してIdentity IDを明示的に解決
      const metricsFile = await downloadData({
        path: ({ identityId }) => `evaluation/results/${identityId}/${jobPath}metrics.json`
      }).result;
      const metricsText = await metricsFile.body.text();
      const metrics = JSON.parse(metricsText);
      setEvaluationMetrics(metrics);

      // predictions.csvを読み込む
      try {
        const predictionsFile = await downloadData({
          path: ({ identityId }) => `evaluation/results/${identityId}/${jobPath}predictions.csv`
        }).result;
        const predictionsText = await predictionsFile.body.text();
        const lines = predictionsText.split('\n').filter(l => l.trim());

        // CSVパース（簡易版）
        const predictions: FilePrediction[] = [];
        for (let i = 1; i < lines.length; i++) {
          const parts = lines[i].split(',');
          if (parts.length >= 5) {
            predictions.push({
              filename: parts[0],
              true_label: parts[1],
              predicted_label: parts[2],
              confidence: parts[3],
              correct: parts[4] === 'True' || parts[4] === 'true',
            });
          }
        }
        setFilePredictions(predictions);
      } catch (err) {
        console.warn('predictions.csv not found or error:', err);
      }
    } catch (err) {
      console.error('Error loading results:', err);
      setError('評価結果の読み込みに失敗しました');
    }
  };

  // ステップインジケーター
  const steps = [
    { id: 'select-model', label: 'モデル選択', icon: <Brain className="w-4 h-4" /> },
    { id: 'select-data', label: 'データ選択', icon: <FolderOpen className="w-4 h-4" /> },
    { id: 'configure-metadata', label: 'メタデータ設定', icon: <Target className="w-4 h-4" /> },
    { id: 'running', label: '評価実行中', icon: <Loader2 className="w-4 h-4 animate-spin" /> },
    { id: 'results', label: '結果表示', icon: <CheckCircle2 className="w-4 h-4" /> },
  ];

  const currentStepIndex = steps.findIndex(s => s.id === currentStep);

  return (
    <div className="space-y-6">
      {/* ステップインジケーター */}
      <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
        <div className="flex items-center justify-between">
          {steps.map((step, idx) => (
            <div key={step.id} className="flex items-center">
              <div
                className={`flex items-center gap-2 ${idx <= currentStepIndex ? 'text-violet-400' : 'text-zinc-600'
                  }`}
              >
                {step.icon}
                <span className="text-sm font-medium hidden sm:inline">{step.label}</span>
              </div>
              {idx < steps.length - 1 && (
                <ChevronRight
                  className={`w-5 h-5 mx-2 ${idx < currentStepIndex ? 'text-violet-400' : 'text-zinc-600'
                    }`}
                />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* エラー表示 */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/50 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-400">エラー</p>
            <p className="text-sm text-red-300 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* ステップ1: モデル選択 */}
      {currentStep === 'select-model' && (
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">評価するモデルを選択</h3>
            <button
              onClick={loadSavedModels}
              disabled={isLoadingModels}
              className="p-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <RefreshCw className={`w-4 h-4 ${isLoadingModels ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {isLoadingModels ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-violet-400" />
            </div>
          ) : savedModels.length === 0 ? (
            <div className="text-center py-12 text-zinc-400">
              <Brain className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>保存されているモデルがありません</p>
              <p className="text-sm mt-1">まず「モデル構築」タブでモデルを訓練してください</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {savedModels.map((model) => (
                <button
                  key={model.path}
                  onClick={() => handleSelectModel(model)}
                  className="text-left p-4 rounded-lg border border-zinc-800 hover:border-violet-500/50 bg-zinc-800/30 hover:bg-zinc-800/50 transition-colors"
                >
                  <div className="flex items-start gap-3">
                    <Brain className="w-5 h-5 text-violet-400 flex-shrink-0 mt-1" />
                    <div className="flex-1 min-w-0">
                      <h4 className="font-medium text-white truncate">{model.name}</h4>
                      <div className="flex items-center gap-2 mt-2 text-xs text-zinc-400">
                        <Calendar className="w-3 h-3" />
                        <span>{model.createdAt.toLocaleString('ja-JP')}</span>
                      </div>
                      {model.problemType && (
                        <div className="mt-2">
                          <span className={`text-xs px-2 py-1 rounded ${
                            model.problemType === 'regression' 
                              ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' 
                              : 'bg-violet-500/20 text-violet-400 border border-violet-500/30'
                          }`}>
                            {model.problemType === 'regression' ? '回帰問題' : '分類問題'}
                          </span>
                          {model.problemType === 'regression' && model.tolerance !== undefined && (
                            <span className="ml-2 text-xs text-zinc-400">
                              許容範囲: {model.tolerance}
                            </span>
                          )}
                        </div>
                      )}
                      {model.classes && model.classes.length > 0 && (
                        <div className="mt-2 text-xs text-zinc-400">
                          <span>クラス数: {model.classes.length}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ステップ2: データ選択 */}
      {currentStep === 'select-data' && (
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">評価データを選択</h3>

          {selectedModel && (
            <div className="mb-6 p-4 bg-zinc-800/30 rounded-lg border border-zinc-700">
              <p className="text-sm text-zinc-400">選択中のモデル:</p>
              <p className="text-white font-medium mt-1">{selectedModel.name}</p>
            </div>
          )}

          <button
            onClick={handleSelectData}
            disabled={isLoadingFiles}
            className="w-full py-4 px-6 rounded-lg bg-violet-600 hover:bg-violet-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 font-medium"
          >
            {isLoadingFiles ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>読み込み中...</span>
              </>
            ) : (
              <>
                <FolderOpen className="w-5 h-5" />
                <span>フォルダを選択</span>
              </>
            )}
          </button>

          {inputFolder && (
            <div className="mt-6 p-4 bg-zinc-800/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-4 h-4 text-green-400" />
                <span className="text-sm font-medium text-white">選択済み</span>
              </div>
              <p className="text-sm text-zinc-400">{inputFolder.name}</p>
              <p className="text-xs text-zinc-500 mt-1">{wavFiles.length} ファイル</p>

              <button
                onClick={() => setCurrentStep('configure-metadata')}
                className="mt-4 w-full py-2 px-4 rounded-lg bg-violet-600 hover:bg-violet-700 font-medium flex items-center justify-center gap-2"
              >
                <span>次へ</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      )}

      {/* ステップ3: メタデータ設定 */}
      {currentStep === 'configure-metadata' && metadata && (
        <div className="space-y-6">
          <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">メタデータ設定</h3>
            <p className="text-sm text-zinc-400 mb-4">
              訓練時と同じ設定を使用してください
            </p>

            <MetadataConfig
              metadata={metadata}
              targetConfig={targetConfig}
              auxiliaryFields={auxiliaryFields}
              onTargetConfigChange={setTargetConfig}
              onAuxiliaryFieldsChange={setAuxiliaryFields}
              onFieldLabelChange={() => { }} // 評価時はラベル変更不要
              problemType={problemType}
              onProblemTypeChange={(type) => {
                setProblemType(type);
                if (targetConfig) {
                  setTargetConfig({ ...targetConfig, problemType: type });
                }
              }}
              tolerance={targetConfig?.tolerance || 0}
              onToleranceChange={(tolerance) => {
                if (targetConfig) {
                  setTargetConfig({ ...targetConfig, tolerance });
                }
              }}
            />
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => setCurrentStep('select-data')}
              className="flex-1 py-3 px-6 rounded-lg bg-zinc-800 hover:bg-zinc-700 font-medium"
            >
              戻る
            </button>
            <button
              onClick={handleStartEvaluation}
              disabled={!targetConfig}
              className="flex-1 py-3 px-6 rounded-lg bg-violet-600 hover:bg-violet-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2"
            >
              <Play className="w-5 h-5" />
              <span>評価開始</span>
            </button>
          </div>
        </div>
      )}

      {/* ステップ4: 評価実行中 */}
      {currentStep === 'running' && (
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-8">
          <div className="text-center">
            {isUploading ? (
              <>
                <Upload className="w-12 h-12 text-violet-400 mx-auto mb-4 animate-pulse" />
                <h3 className="text-lg font-semibold text-white mb-2">データをアップロード中...</h3>
                <div className="w-full max-w-md mx-auto mt-4">
                  <div className="bg-zinc-800 rounded-full h-2">
                    <div
                      className="bg-violet-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  <p className="text-sm text-zinc-400 mt-2">{uploadProgress}%</p>
                </div>
              </>
            ) : (
              <>
                <Loader2 className="w-12 h-12 text-violet-400 mx-auto mb-4 animate-spin" />
                <h3 className="text-lg font-semibold text-white mb-2">評価を実行中...</h3>
                <p className="text-sm text-zinc-400 mb-4">
                  ステータス: {jobStatus || '開始中'}
                </p>
                <p className="text-xs text-zinc-500">
                  予想時間: 5-15分（データ量による）
                </p>
                {jobName && (
                  <p className="text-xs text-zinc-600 mt-2 font-mono">{jobName}</p>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* ステップ5: 結果表示 */}
      {currentStep === 'results' && evaluationMetrics && selectedModel && (
        <div className="space-y-6">
          <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-lg font-semibold text-white">評価結果</h3>
                <p className="text-sm text-zinc-400 mt-1">
                  モデル: {selectedModel.name}
                </p>
              </div>
              <button
                onClick={() => {
                  setCurrentStep('select-model');
                  setSelectedModel(null);
                  setEvaluationMetrics(null);
                  setFilePredictions([]);
                  setError(null);
                }}
                className="py-2 px-4 rounded-lg bg-zinc-800 hover:bg-zinc-700 font-medium text-sm"
              >
                新しい評価
              </button>
            </div>

            <InferenceResults
              metrics={evaluationMetrics}
              predictions={filePredictions.length > 0 ? filePredictions : undefined}
              classNames={selectedModel.classes || []}
            />
          </div>
        </div>
      )}
    </div>
  );
}
