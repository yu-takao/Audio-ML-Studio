import { useState, useCallback, useEffect } from 'react';
import { list, getUrl } from 'aws-amplify/storage';
import {
  Brain,
  Loader2,
  RefreshCw,
  Calendar,
  Target,
  Award,
  TrendingUp,
  ChevronDown,
  ChevronUp,
  BarChart3,
  Layers,
  Clock,
  AlertCircle,
  FolderOpen,
  Database,
  Activity,
  ImageIcon,
  Play,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import outputs from '../../amplify_outputs.json';

interface ModelBrowserProps {
  userId: string;
}

// モデルメタデータ（訓練結果）
interface ModelMetadata {
  classes: string[];
  input_shape: number[];
  target_field: string;
  // 後方互換のため optional（新しい学習スクリプトのみ付与）
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

// 解析結果サマリー
interface AnalysisSummary {
  classes: string[];
  samples_per_class: Record<string, number>;
  analyzed_per_class: Record<string, number>;
  frequency_importance: Record<string, number[]>;
  sample_results: Array<{
    filename: string;
    true_class: string;
    pred_class: string;
    confidence: number;
    image: string;
  }>;
  output_files: {
    frequency_importance: string;
    class_avg_spectrograms: string[];
    class_avg_gradcams: string[];
    sample_gradcams: string[];
  };
}

// S3に保存されているモデル情報
interface SavedModel {
  path: string;
  name: string;
  createdAt: Date;
  metadata?: ModelMetadata;
  isLoadingMetadata?: boolean;
  analysisStatus?: 'none' | 'loading' | 'available' | 'not_found';
  analysisSummary?: AnalysisSummary;
  analysisPath?: string; // analysis_summary.jsonが見つかったディレクトリ
}

export function ModelBrowser({ userId }: ModelBrowserProps) {
  const [models, setModels] = useState<SavedModel[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<SavedModel | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [isStartingAnalysis, setIsStartingAnalysis] = useState(false);
  const [analysisImages, setAnalysisImages] = useState<Record<string, string>>({});
  const [loadingImages, setLoadingImages] = useState<Set<string>>(new Set());
  const [selectedAnalysisImage, setSelectedAnalysisImage] = useState<string | null>(null);
  
  // Lambda URL for analysis
  const startAnalysisUrl = (outputs as { custom?: { startAnalysisUrl?: string } }).custom?.startAnalysisUrl;

  /**
   * S3からモデル一覧を取得
   */
  const loadModels = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // ユーザー専用のモデルパスを取得
      const result = await list({
        path: `models/${userId}/`,
        options: {
          listAll: true,
        },
      });

      // model.tar.gz をモデルの存在判定に利用
      const modelList: SavedModel[] = result.items
        .filter(item => item.path.endsWith('model.tar.gz'))
        .map(item => {
          const parts = item.path.split('/'); // models/{userId}/{jobName}/output/model.tar.gz
          const jobName = parts.length >= 4 ? parts[2] : 'unknown';
          // モデルルートパス（output直前まで）
          const modelPath = parts.slice(0, parts.length - 1).join('/'); // models/userId/job/output
          return {
            path: modelPath,
            name: jobName,
            createdAt: item.lastModified || new Date(),
          };
        });

      // 日付順でソート（新しいものが上）
      modelList.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
      
      setModels(modelList);
    } catch (err) {
      console.error('Failed to load models:', err);
      setError('モデル一覧の読み込みに失敗しました');
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * モデルのメタデータを読み込む
   * - 署名URL + no-store で取得し、常にS3の最新を反映（キャッシュ対策）
   */
  const loadModelMetadata = useCallback(async (model: SavedModel) => {

    // ローディング状態を設定
    setModels(prev => prev.map(m => 
      m.path === model.path ? { ...m, isLoadingMetadata: true } : m
    ));

    try {
      // model.path は models/{userId}/{jobName}/output
      // 古い環境や手動差し替えでパスがズレる場合があるため候補を順に試す
      const candidates = [
        `${model.path}/model_metadata.json`,
        // 念のため: output直下ではなく job 直下に置かれているケース
        model.path.endsWith('/output') ? `${model.path.replace(/\/output$/, '')}/model_metadata.json` : '',
        // 念のため: public/ 配下に置かれている旧形式
        `public/${model.path}/model_metadata.json`,
      ].filter(Boolean);

      let lastErr: unknown = null;
      let text: string | null = null;
      let usedPath: string | null = null;

      for (const metadataPath of candidates) {
        try {
          // ブラウザキャッシュを確実に回避するため、署名付きURLを都度発行して no-store で取得
          const signed = await getUrl({ path: metadataPath, options: { expiresIn: 60 } });
          const res = await fetch(signed.url.toString(), { cache: 'no-store' });
          if (!res.ok) throw new Error(`Failed to fetch metadata (${res.status})`);
          text = await res.text();
          usedPath = metadataPath;
          break;
        } catch (e) {
          lastErr = e;
        }
      }

      if (!text) throw lastErr || new Error('Failed to load model metadata');

      const metadata: ModelMetadata = JSON.parse(text);

      const updatedModel = { ...model, metadata, isLoadingMetadata: false };
      
      setModels(prev => prev.map(m => 
        m.path === model.path ? updatedModel : m
      ));
      setSelectedModel(updatedModel);
    } catch (err) {
      console.error('Failed to load model metadata:', err);
      setModels(prev => prev.map(m => 
        m.path === model.path ? { ...m, isLoadingMetadata: false } : m
      ));
    }
  }, []);

  // 初回読み込み
  useEffect(() => {
    loadModels();
  }, [loadModels]);

  // 日付フォーマット
  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('ja-JP', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  // 精度をパーセントで表示
  const formatAccuracy = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatSplitMode = (mode?: string) => {
    if (!mode) return '未記録';
    if (mode === 'presplit') return '事前分割（train/validation/test）';
    if (mode === 'random') return 'ランダム分割（legacy）';
    return mode;
  };

  /**
   * 解析結果の読み込み
   */
  const loadAnalysisResults = useCallback(async (model: SavedModel) => {
    // モデルパスからanalysisディレクトリを検索
    // models/{userId}/{jobName}/analysis/{timestamp}/analysis_summary.json
    const baseAnalysisPath = model.path.replace(/\/output$/, '') + '/analysis';
    
    setModels(prev => prev.map(m => 
      m.path === model.path ? { ...m, analysisStatus: 'loading' } : m
    ));
    
    try {
      // analysis ディレクトリ配下のファイルを一覧
      const result = await list({
        path: baseAnalysisPath + '/',
        options: { listAll: true },
      });

      // analysis_summary.json を探す
      const summaryFiles = result.items.filter(item => 
        item.path.endsWith('analysis_summary.json')
      );

      if (summaryFiles.length === 0) {
        setModels(prev => prev.map(m => 
          m.path === model.path ? { ...m, analysisStatus: 'not_found' } : m
        ));
        return;
      }

      // 最新の解析結果を取得（最も新しい lastModified）
      const latestSummary = summaryFiles.sort((a, b) => 
        (b.lastModified?.getTime() || 0) - (a.lastModified?.getTime() || 0)
      )[0];

      // analysis_summary.json のディレクトリパス
      const analysisDir = latestSummary.path.replace('/analysis_summary.json', '');

      // JSONを取得
      const signed = await getUrl({ path: latestSummary.path, options: { expiresIn: 300 } });
      const res = await fetch(signed.url.toString(), { cache: 'no-store' });
      if (!res.ok) throw new Error('Failed to fetch analysis summary');
      const summary: AnalysisSummary = await res.json();

      const updatedModel = {
        ...model,
        analysisStatus: 'available' as const,
        analysisSummary: summary,
        analysisPath: analysisDir,
      };

      setModels(prev => prev.map(m => 
        m.path === model.path ? updatedModel : m
      ));
      if (selectedModel?.path === model.path) {
        setSelectedModel(updatedModel);
      }
    } catch (err) {
      console.error('Failed to load analysis results:', err);
      setModels(prev => prev.map(m => 
        m.path === model.path ? { ...m, analysisStatus: 'not_found' } : m
      ));
    }
  }, [selectedModel]);

  /**
   * 解析を開始
   */
  const startAnalysis = useCallback(async (model: SavedModel) => {
    if (!startAnalysisUrl) {
      setError('解析機能が設定されていません（startAnalysisUrl が見つかりません）');
      return;
    }

    setIsStartingAnalysis(true);
    try {
      const response = await fetch(startAnalysisUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId,
          modelJobName: model.name,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Failed to start analysis');
      }

      const result = await response.json();
      console.log('Analysis started:', result);
      alert(`解析を開始しました。\nジョブ名: ${result.processingJobName}\n\n処理完了まで数分〜数十分かかります。完了後、このページを更新してください。`);
    } catch (err) {
      console.error('Failed to start analysis:', err);
      setError(`解析の開始に失敗しました: ${(err as Error).message}`);
    } finally {
      setIsStartingAnalysis(false);
    }
  }, [startAnalysisUrl, userId]);

  /**
   * 解析画像を読み込み
   */
  const loadAnalysisImage = useCallback(async (imagePath: string) => {
    if (analysisImages[imagePath] || loadingImages.has(imagePath)) return;

    setLoadingImages(prev => new Set(prev).add(imagePath));
    try {
      const signed = await getUrl({ path: imagePath, options: { expiresIn: 300 } });
      setAnalysisImages(prev => ({ ...prev, [imagePath]: signed.url.toString() }));
    } catch (err) {
      console.error('Failed to load analysis image:', err);
    } finally {
      setLoadingImages(prev => {
        const next = new Set(prev);
        next.delete(imagePath);
        return next;
      });
    }
  }, [analysisImages, loadingImages]);

  return (
    <div className="space-y-6">
      {/* ヘッダー */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-violet-500/20">
            <Brain className="w-5 h-5 text-violet-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">保存済みモデル</h3>
            <p className="text-sm text-zinc-400">過去に訓練したモデルを確認・評価</p>
          </div>
        </div>
        <button
          onClick={loadModels}
          disabled={isLoading}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 
                   text-zinc-300 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          更新
        </button>
      </div>

      {/* エラー表示 */}
      {error && (
        <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/30">
          <div className="flex items-center gap-2 text-red-400">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* ローディング */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-violet-400 animate-spin" />
        </div>
      )}

      {/* モデルがない場合 */}
      {!isLoading && models.length === 0 && (
        <div className="text-center py-12 bg-zinc-800/30 rounded-xl border border-zinc-700/50">
          <FolderOpen className="w-12 h-12 text-zinc-600 mx-auto mb-4" />
          <p className="text-zinc-400">保存済みのモデルがありません</p>
          <p className="text-sm text-zinc-500 mt-2">訓練を完了するとモデルが保存されます</p>
        </div>
      )}

      {/* モデル一覧とプレビュー */}
      {!isLoading && models.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* モデル一覧 */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-zinc-400 mb-3">モデル一覧 ({models.length}件)</h4>
            <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
              {models.map((model) => (
                <button
                  key={model.path}
                  onClick={() => loadModelMetadata(model)}
                  className={`w-full p-4 rounded-lg border transition-all text-left
                    ${selectedModel?.path === model.path
                      ? 'bg-violet-500/20 border-violet-500/50'
                      : 'bg-zinc-800/50 border-zinc-700/50 hover:bg-zinc-800'
                    }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <Brain className={`w-4 h-4 ${selectedModel?.path === model.path ? 'text-violet-400' : 'text-zinc-500'}`} />
                        <span className="font-medium text-white truncate">{model.name}</span>
                      </div>
                      <div className="flex items-center gap-2 mt-2 text-xs text-zinc-500">
                        <Calendar className="w-3 h-3" />
                        <span>{formatDate(model.createdAt)}</span>
                      </div>
                    </div>
                    {model.isLoadingMetadata && (
                      <Loader2 className="w-4 h-4 text-violet-400 animate-spin" />
                    )}
                    {model.metadata && (
                      <div className="text-right">
                        <span className="text-lg font-bold text-emerald-400">
                          {formatAccuracy(model.metadata.metrics.test_accuracy)}
                        </span>
                        <p className="text-xs text-zinc-500">テスト精度</p>
                      </div>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* モデル詳細 */}
          <div className="bg-zinc-800/30 rounded-xl border border-zinc-700/50 p-6">
            {!selectedModel ? (
              <div className="text-center py-12">
                <Brain className="w-12 h-12 text-zinc-600 mx-auto mb-4" />
                <p className="text-zinc-400">モデルを選択してください</p>
              </div>
            ) : selectedModel.isLoadingMetadata ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 text-violet-400 animate-spin" />
              </div>
            ) : selectedModel.metadata ? (
              <div className="space-y-6">
                {/* モデル名 */}
                <div>
                  <h4 className="text-lg font-semibold text-white">{selectedModel.name}</h4>
                  <p className="text-sm text-zinc-500 mt-1">{formatDate(selectedModel.createdAt)}</p>
                </div>

                {/* 評価メトリクス */}
                <div>
                  <h5 className="text-sm font-medium text-zinc-400 mb-3 flex items-center gap-2">
                    <Award className="w-4 h-4" />
                    評価結果
                  </h5>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
                      <div className="flex items-center gap-2 text-emerald-400 mb-1">
                        <Target className="w-4 h-4" />
                        <span className="text-xs font-medium">テスト精度</span>
                      </div>
                      <span className="text-2xl font-bold text-white">
                        {formatAccuracy(selectedModel.metadata.metrics.test_accuracy)}
                      </span>
                    </div>
                    <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/30">
                      <div className="flex items-center gap-2 text-blue-400 mb-1">
                        <TrendingUp className="w-4 h-4" />
                        <span className="text-xs font-medium">検証精度</span>
                      </div>
                      <span className="text-2xl font-bold text-white">
                        {formatAccuracy(selectedModel.metadata.metrics.final_val_accuracy)}
                      </span>
                    </div>
                    <div className="p-3 rounded-lg bg-violet-500/10 border border-violet-500/30">
                      <div className="flex items-center gap-2 text-violet-400 mb-1">
                        <BarChart3 className="w-4 h-4" />
                        <span className="text-xs font-medium">訓練精度</span>
                      </div>
                      <span className="text-2xl font-bold text-white">
                        {formatAccuracy(selectedModel.metadata.metrics.final_train_accuracy)}
                      </span>
                    </div>
                    <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30">
                      <div className="flex items-center gap-2 text-amber-400 mb-1">
                        <Layers className="w-4 h-4" />
                        <span className="text-xs font-medium">テスト損失</span>
                      </div>
                      <span className="text-2xl font-bold text-white">
                        {selectedModel.metadata.metrics.test_loss.toFixed(4)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* クラス情報 */}
                <div>
                  <h5 className="text-sm font-medium text-zinc-400 mb-3">
                    クラス ({selectedModel.metadata.classes.length}種類)
                  </h5>
                  <div className="flex flex-wrap gap-2">
                    {selectedModel.metadata.classes.map((cls, idx) => (
                      <span
                        key={idx}
                        className="px-3 py-1 rounded-full bg-zinc-700/50 text-sm text-zinc-300"
                      >
                        {cls}
                      </span>
                    ))}
                  </div>
                </div>

                {/* 訓練パラメータ */}
                <div>
                  <h5 className="text-sm font-medium text-zinc-400 mb-3 flex items-center gap-2">
                    <Clock className="w-4 h-4" />
                    訓練パラメータ
                  </h5>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="flex justify-between p-2 rounded bg-zinc-800/50">
                      <span className="text-zinc-500">エポック数</span>
                      <span className="text-white">{selectedModel.metadata.training_params.epochs}</span>
                    </div>
                    <div className="flex justify-between p-2 rounded bg-zinc-800/50">
                      <span className="text-zinc-500">バッチサイズ</span>
                      <span className="text-white">{selectedModel.metadata.training_params.batch_size}</span>
                    </div>
                    <div className="flex justify-between p-2 rounded bg-zinc-800/50">
                      <span className="text-zinc-500">学習率</span>
                      <span className="text-white">{selectedModel.metadata.training_params.learning_rate}</span>
                    </div>
                    <div className="flex justify-between p-2 rounded bg-zinc-800/50">
                      <span className="text-zinc-500">検証割合</span>
                      <span className="text-white">{(selectedModel.metadata.training_params.validation_split * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>

                {/* データ内訳（後方互換：無い場合は未記録） */}
                <div>
                  <h5 className="text-sm font-medium text-zinc-400 mb-3 flex items-center gap-2">
                    <Database className="w-4 h-4" />
                    データ数の内訳
                  </h5>

                  <div className="space-y-2">
                    {(() => {
                      const ds = selectedModel.metadata?.dataset;
                      const counts = ds?.counts;
                      const hasCounts = !!counts && Object.keys(counts).length > 0;
                      return (
                        <>
                    <div className="flex justify-between p-2 rounded bg-zinc-800/50 text-sm">
                      <span className="text-zinc-500">分割方式</span>
                      <span className="text-white">{formatSplitMode(selectedModel.metadata.dataset?.split_mode)}</span>
                    </div>

                    {hasCounts ? (
                      <div className="grid grid-cols-4 gap-2 text-sm">
                        <div className="p-3 rounded-lg bg-zinc-800/50 border border-zinc-700/50">
                          <div className="text-xs text-zinc-500">train</div>
                          <div className="text-lg font-bold text-white">
                            {selectedModel.metadata.dataset.counts.train ?? '—'}
                          </div>
                        </div>
                        <div className="p-3 rounded-lg bg-zinc-800/50 border border-zinc-700/50">
                          <div className="text-xs text-zinc-500">validation</div>
                          <div className="text-lg font-bold text-white">
                            {selectedModel.metadata.dataset.counts.validation ?? '—'}
                          </div>
                        </div>
                        <div className="p-3 rounded-lg bg-zinc-800/50 border border-zinc-700/50">
                          <div className="text-xs text-zinc-500">test</div>
                          <div className="text-lg font-bold text-white">
                            {selectedModel.metadata.dataset.counts.test ?? '—'}
                          </div>
                        </div>
                        <div className="p-3 rounded-lg bg-zinc-800/50 border border-zinc-700/50">
                          <div className="text-xs text-zinc-500">total</div>
                          <div className="text-lg font-bold text-white">
                            {selectedModel.metadata.dataset.counts.total ?? '—'}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <div className="text-sm text-zinc-500">
                          このモデルは旧形式のため、データ内訳がメタデータに保存されていません。
                        </div>
                      </div>
                    )}
                        </>
                      );
                    })()}
                  </div>
                </div>

                {/* 訓練履歴グラフ */}
                <div>
                  <button
                    onClick={() => setShowHistory(!showHistory)}
                    className="flex items-center justify-between w-full p-3 rounded-lg bg-zinc-800/50 
                             hover:bg-zinc-800 transition-colors"
                  >
                    <span className="text-sm font-medium text-zinc-300">訓練履歴グラフ</span>
                    {showHistory ? (
                      <ChevronUp className="w-4 h-4 text-zinc-400" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-zinc-400" />
                    )}
                  </button>
                  
                  {showHistory && selectedModel.metadata.history && (
                    <div className="mt-4 space-y-4">
                      {/* 精度グラフ */}
                      <div className="p-4 rounded-lg bg-zinc-900/50">
                        <h6 className="text-xs font-medium text-zinc-400 mb-3">精度 (Accuracy)</h6>
                        <div className="h-32 flex items-end gap-1">
                          {selectedModel.metadata.history.accuracy.map((acc, idx) => (
                            <div
                              key={idx}
                              className="flex-1 bg-gradient-to-t from-violet-500 to-violet-400 rounded-t opacity-60"
                              style={{ height: `${acc * 100}%` }}
                              title={`Epoch ${idx + 1}: ${(acc * 100).toFixed(1)}%`}
                            />
                          ))}
                        </div>
                        <div className="flex justify-between mt-2 text-xs text-zinc-500">
                          <span>Epoch 1</span>
                          <span>Epoch {selectedModel.metadata.history.accuracy.length}</span>
                        </div>
                      </div>

                      {/* 損失グラフ */}
                      <div className="p-4 rounded-lg bg-zinc-900/50">
                        <h6 className="text-xs font-medium text-zinc-400 mb-3">損失 (Loss)</h6>
                        <div className="h-32 flex items-end gap-1">
                          {selectedModel.metadata.history.loss.map((loss, idx) => {
                            const maxLoss = Math.max(...selectedModel.metadata!.history.loss);
                            return (
                              <div
                                key={idx}
                                className="flex-1 bg-gradient-to-t from-amber-500 to-amber-400 rounded-t opacity-60"
                                style={{ height: `${(loss / maxLoss) * 100}%` }}
                                title={`Epoch ${idx + 1}: ${loss.toFixed(4)}`}
                              />
                            );
                          })}
                        </div>
                        <div className="flex justify-between mt-2 text-xs text-zinc-500">
                          <span>Epoch 1</span>
                          <span>Epoch {selectedModel.metadata.history.loss.length}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* モデル解析（Grad-CAM） */}
                <div>
                  <button
                    onClick={() => {
                      setShowAnalysis(!showAnalysis);
                      if (!showAnalysis && !selectedModel.analysisSummary && selectedModel.analysisStatus !== 'loading') {
                        loadAnalysisResults(selectedModel);
                      }
                    }}
                    className="flex items-center justify-between w-full p-3 rounded-lg bg-zinc-800/50 
                             hover:bg-zinc-800 transition-colors"
                  >
                    <span className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                      <Activity className="w-4 h-4" />
                      モデル解析（判定根拠の可視化）
                    </span>
                    {showAnalysis ? (
                      <ChevronUp className="w-4 h-4 text-zinc-400" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-zinc-400" />
                    )}
                  </button>

                  {showAnalysis && (
                    <div className="mt-4 space-y-4">
                      {/* 解析ステータス */}
                      {selectedModel.analysisStatus === 'loading' && (
                        <div className="flex items-center justify-center py-8">
                          <Loader2 className="w-6 h-6 text-violet-400 animate-spin" />
                          <span className="ml-2 text-zinc-400">解析結果を読み込み中...</span>
                        </div>
                      )}

                      {selectedModel.analysisStatus === 'not_found' && (
                        <div className="p-4 rounded-lg bg-zinc-800/50 border border-zinc-700/50">
                          <div className="flex items-center gap-3 mb-4">
                            <ImageIcon className="w-8 h-8 text-zinc-500" />
                            <div>
                              <p className="text-white font-medium">解析結果がありません</p>
                              <p className="text-sm text-zinc-500">
                                Grad-CAMやクラス別平均スペクトログラムを生成するには解析を実行してください
                              </p>
                            </div>
                          </div>
                          <button
                            onClick={() => startAnalysis(selectedModel)}
                            disabled={isStartingAnalysis || !startAnalysisUrl}
                            className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg 
                                     bg-violet-600 hover:bg-violet-500 text-white font-medium
                                     disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                          >
                            {isStartingAnalysis ? (
                              <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                解析を開始中...
                              </>
                            ) : (
                              <>
                                <Play className="w-4 h-4" />
                                解析を開始
                              </>
                            )}
                          </button>
                          {!startAnalysisUrl && (
                            <p className="text-xs text-amber-400 mt-2">
                              ※ 解析機能を使用するにはバックエンドのデプロイが必要です
                            </p>
                          )}
                        </div>
                      )}

                      {selectedModel.analysisStatus === 'available' && selectedModel.analysisSummary && (
                        <div className="space-y-4">
                          {/* 周波数帯別寄与度グラフ */}
                          <div className="p-4 rounded-lg bg-zinc-900/50">
                            <h6 className="text-xs font-medium text-zinc-400 mb-3 flex items-center gap-2">
                              <BarChart3 className="w-4 h-4" />
                              周波数帯別の寄与度
                            </h6>
                            {selectedModel.analysisPath && (
                              <div className="mb-3">
                                {(() => {
                                  const imgPath = `${selectedModel.analysisPath}/${selectedModel.analysisSummary?.output_files.frequency_importance}`;
                                  if (!analysisImages[imgPath]) {
                                    loadAnalysisImage(imgPath);
                                  }
                                  return analysisImages[imgPath] ? (
                                    <img 
                                      src={analysisImages[imgPath]} 
                                      alt="Frequency Importance"
                                      className="w-full rounded-lg cursor-pointer hover:opacity-90 transition-opacity"
                                      onClick={() => setSelectedAnalysisImage(analysisImages[imgPath])}
                                    />
                                  ) : (
                                    <div className="h-32 flex items-center justify-center bg-zinc-800 rounded-lg">
                                      <Loader2 className="w-6 h-6 text-zinc-500 animate-spin" />
                                    </div>
                                  );
                                })()}
                              </div>
                            )}
                            <p className="text-xs text-zinc-500">
                              各クラスがどの周波数帯を重視して判定しているかを示します
                            </p>
                          </div>

                          {/* クラス別平均Grad-CAM */}
                          <div className="p-4 rounded-lg bg-zinc-900/50">
                            <h6 className="text-xs font-medium text-zinc-400 mb-3 flex items-center gap-2">
                              <Target className="w-4 h-4" />
                              クラス別平均Grad-CAM
                            </h6>
                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                              {selectedModel.analysisSummary.output_files.class_avg_gradcams.map((imgFile) => {
                                const imgPath = `${selectedModel.analysisPath}/${imgFile}`;
                                const className = imgFile.replace('class_', '').replace('_avg_gradcam.png', '');
                                if (!analysisImages[imgPath]) {
                                  loadAnalysisImage(imgPath);
                                }
                                return (
                                  <div key={imgFile} className="space-y-1">
                                    <div 
                                      className="aspect-video bg-zinc-800 rounded overflow-hidden cursor-pointer hover:ring-2 ring-violet-500 transition-all"
                                      onClick={() => analysisImages[imgPath] && setSelectedAnalysisImage(analysisImages[imgPath])}
                                    >
                                      {analysisImages[imgPath] ? (
                                        <img src={analysisImages[imgPath]} alt={className} className="w-full h-full object-cover" />
                                      ) : (
                                        <div className="w-full h-full flex items-center justify-center">
                                          <Loader2 className="w-4 h-4 text-zinc-600 animate-spin" />
                                        </div>
                                      )}
                                    </div>
                                    <p className="text-xs text-zinc-500 text-center truncate">{className}</p>
                                  </div>
                                );
                              })}
                            </div>
                            <p className="text-xs text-zinc-500 mt-3">
                              赤い領域ほど判定に強く寄与しています
                            </p>
                          </div>

                          {/* クラス別平均スペクトログラム */}
                          <div className="p-4 rounded-lg bg-zinc-900/50">
                            <h6 className="text-xs font-medium text-zinc-400 mb-3 flex items-center gap-2">
                              <Layers className="w-4 h-4" />
                              クラス別平均スペクトログラム
                            </h6>
                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                              {selectedModel.analysisSummary.output_files.class_avg_spectrograms.map((imgFile) => {
                                const imgPath = `${selectedModel.analysisPath}/${imgFile}`;
                                const className = imgFile.replace('class_', '').replace('_avg_spectrogram.png', '');
                                if (!analysisImages[imgPath]) {
                                  loadAnalysisImage(imgPath);
                                }
                                return (
                                  <div key={imgFile} className="space-y-1">
                                    <div 
                                      className="aspect-video bg-zinc-800 rounded overflow-hidden cursor-pointer hover:ring-2 ring-violet-500 transition-all"
                                      onClick={() => analysisImages[imgPath] && setSelectedAnalysisImage(analysisImages[imgPath])}
                                    >
                                      {analysisImages[imgPath] ? (
                                        <img src={analysisImages[imgPath]} alt={className} className="w-full h-full object-cover" />
                                      ) : (
                                        <div className="w-full h-full flex items-center justify-center">
                                          <Loader2 className="w-4 h-4 text-zinc-600 animate-spin" />
                                        </div>
                                      )}
                                    </div>
                                    <p className="text-xs text-zinc-500 text-center truncate">{className}</p>
                                  </div>
                                );
                              })}
                            </div>
                            <p className="text-xs text-zinc-500 mt-3">
                              各クラスの音の平均的な特徴を示しています
                            </p>
                          </div>

                          {/* サンプルGrad-CAM */}
                          {selectedModel.analysisSummary.sample_results.length > 0 && (
                            <div className="p-4 rounded-lg bg-zinc-900/50">
                              <h6 className="text-xs font-medium text-zinc-400 mb-3 flex items-center gap-2">
                                <ImageIcon className="w-4 h-4" />
                                サンプル別Grad-CAM
                              </h6>
                              <div className="space-y-2 max-h-64 overflow-y-auto">
                                {selectedModel.analysisSummary.sample_results.map((sample, idx) => {
                                  const imgPath = `${selectedModel.analysisPath}/${sample.image}`;
                                  const isCorrect = sample.true_class === sample.pred_class;
                                  if (!analysisImages[imgPath]) {
                                    loadAnalysisImage(imgPath);
                                  }
                                  return (
                                    <div 
                                      key={idx} 
                                      className="flex items-center gap-3 p-2 rounded bg-zinc-800/50 hover:bg-zinc-800 cursor-pointer transition-colors"
                                      onClick={() => analysisImages[imgPath] && setSelectedAnalysisImage(analysisImages[imgPath])}
                                    >
                                      <div className="w-16 h-10 bg-zinc-700 rounded overflow-hidden flex-shrink-0">
                                        {analysisImages[imgPath] ? (
                                          <img src={analysisImages[imgPath]} alt={sample.filename} className="w-full h-full object-cover" />
                                        ) : (
                                          <div className="w-full h-full flex items-center justify-center">
                                            <Loader2 className="w-3 h-3 text-zinc-600 animate-spin" />
                                          </div>
                                        )}
                                      </div>
                                      <div className="flex-1 min-w-0">
                                        <p className="text-xs text-white truncate">{sample.filename}</p>
                                        <div className="flex items-center gap-2 text-xs">
                                          <span className="text-zinc-500">予測: {sample.pred_class}</span>
                                          <span className={isCorrect ? 'text-emerald-400' : 'text-red-400'}>
                                            ({(sample.confidence * 100).toFixed(1)}%)
                                          </span>
                                        </div>
                                      </div>
                                      {isCorrect ? (
                                        <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                                      ) : (
                                        <XCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
                                      )}
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* 解析結果がまだ取得されていない場合のボタン */}
                      {!selectedModel.analysisStatus && (
                        <button
                          onClick={() => loadAnalysisResults(selectedModel)}
                          className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg 
                                   bg-zinc-700 hover:bg-zinc-600 text-white transition-colors"
                        >
                          <RefreshCw className="w-4 h-4" />
                          解析結果を確認
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <AlertCircle className="w-12 h-12 text-zinc-600 mx-auto mb-4" />
                <p className="text-zinc-400">メタデータを読み込めませんでした</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 画像拡大モーダル */}
      {selectedAnalysisImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
          onClick={() => setSelectedAnalysisImage(null)}
        >
          <div className="relative max-w-4xl max-h-[90vh] w-full">
            <img 
              src={selectedAnalysisImage} 
              alt="Analysis" 
              className="w-full h-auto max-h-[90vh] object-contain rounded-lg"
              onClick={(e) => e.stopPropagation()}
            />
            <button
              onClick={() => setSelectedAnalysisImage(null)}
              className="absolute top-2 right-2 p-2 rounded-full bg-black/50 hover:bg-black/70 text-white transition-colors"
            >
              <XCircle className="w-6 h-6" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

