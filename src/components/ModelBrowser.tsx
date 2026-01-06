import { useState, useCallback, useEffect } from 'react';
import { list, downloadData } from 'aws-amplify/storage';
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
} from 'lucide-react';

interface ModelBrowserProps {
  userId: string;
}

// モデルメタデータ（訓練結果）
interface ModelMetadata {
  classes: string[];
  input_shape: number[];
  target_field: string;
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

// S3に保存されているモデル情報
interface SavedModel {
  path: string;
  name: string;
  createdAt: Date;
  metadata?: ModelMetadata;
  isLoadingMetadata?: boolean;
}

export function ModelBrowser({ userId }: ModelBrowserProps) {
  const [models, setModels] = useState<SavedModel[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<SavedModel | null>(null);
  const [showHistory, setShowHistory] = useState(false);

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
   */
  const loadModelMetadata = useCallback(async (model: SavedModel) => {
    // 既に読み込み済みの場合はスキップ
    if (model.metadata) {
      setSelectedModel(model);
      return;
    }

    // ローディング状態を設定
    setModels(prev => prev.map(m => 
      m.path === model.path ? { ...m, isLoadingMetadata: true } : m
    ));

    try {
      // model.path は models/userId/jobName/output
      const metadataPath = `${model.path}/model_metadata.json`;
      const result = await downloadData({ path: metadataPath }).result;
      const text = await result.body.text();
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
    </div>
  );
}

