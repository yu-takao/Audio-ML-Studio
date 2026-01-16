import { useMemo } from 'react';
import { CheckCircle2, XCircle, TrendingUp, Target, Award } from 'lucide-react';
import { ConfusionMatrix } from './ConfusionMatrix';

interface ClassMetric {
  class_name: string;
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

interface EvaluationMetrics {
  accuracy?: number;  // オプショナルに変更（回帰モデルでは許容範囲内精度として使用）
  precision?: number;
  recall?: number;
  f1_score?: number;
  confusion_matrix?: number[][];
  class_metrics?: ClassMetric[];
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

interface InferenceResultsProps {
  metrics: EvaluationMetrics;
  predictions?: FilePrediction[];
  classNames: string[];
}

/**
 * 評価結果を表示するコンポーネント
 */
export function InferenceResults({ metrics, predictions, classNames }: InferenceResultsProps) {
  // 全体のサマリー統計
  const summary = useMemo(() => {
    const totalSamples = predictions?.length || 0;
    const correctPredictions = predictions?.filter(p => p.correct).length || 0;
    const incorrectPredictions = totalSamples - correctPredictions;

    return {
      totalSamples,
      correctPredictions,
      incorrectPredictions,
      accuracy: totalSamples > 0 ? correctPredictions / totalSamples : 0,
    };
  }, [predictions]);

  const isRegression = metrics.problem_type === 'regression';

  return (
    <div className="space-y-6">
      {/* 全体メトリクス */}
      {isRegression ? (
        // 回帰問題の指標
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            icon={<Target className="w-5 h-5" />}
            label={`許容範囲内 (±${metrics.tolerance ?? 0})`}
            value={`${((metrics.accuracy ?? 0) * 100).toFixed(2)}%`}
            color="violet"
          />
          <MetricCard
            icon={<Award className="w-5 h-5" />}
            label="MAE (平均絶対誤差)"
            value={metrics.mae?.toFixed(4) || 'N/A'}
            color="blue"
          />
          <MetricCard
            icon={<CheckCircle2 className="w-5 h-5" />}
            label="RMSE (二乗平均平方根誤差)"
            value={metrics.rmse?.toFixed(4) || 'N/A'}
            color="green"
          />
          <MetricCard
            icon={<TrendingUp className="w-5 h-5" />}
            label="R² スコア"
            value={metrics.r2_score?.toFixed(4) || 'N/A'}
            color="cyan"
          />
        </div>
      ) : (
        // 分類問題の指標
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            icon={<Target className="w-5 h-5" />}
            label={metrics.tolerance && metrics.tolerance > 0 ? `精度 (許容範囲±${metrics.tolerance})` : "精度 (Accuracy)"}
            value={`${((metrics.accuracy_with_tolerance ?? metrics.accuracy ?? 0) * 100).toFixed(2)}%`}
            color="violet"
          />
          <MetricCard
            icon={<Award className="w-5 h-5" />}
            label="F1スコア"
            value={metrics.f1_score?.toFixed(4) || 'N/A'}
            color="blue"
          />
          <MetricCard
            icon={<CheckCircle2 className="w-5 h-5" />}
            label="適合率 (Precision)"
            value={metrics.precision?.toFixed(4) || 'N/A'}
            color="green"
          />
          <MetricCard
            icon={<TrendingUp className="w-5 h-5" />}
            label="再現率 (Recall)"
            value={metrics.recall?.toFixed(4) || 'N/A'}
            color="cyan"
          />
        </div>
      )}

      {/* サンプル数サマリー */}
      {predictions && (
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">予測サマリー</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-white">{summary.totalSamples}</div>
              <div className="text-sm text-zinc-400 mt-1">総サンプル数</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-400">{summary.correctPredictions}</div>
              <div className="text-sm text-zinc-400 mt-1">正解数</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-red-400">{summary.incorrectPredictions}</div>
              <div className="text-sm text-zinc-400 mt-1">誤り数</div>
            </div>
          </div>
        </div>
      )}

      {/* 混同行列（分類問題のみ） */}
      {!isRegression && metrics.confusion_matrix && (
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">混同行列 (Confusion Matrix)</h3>
          <ConfusionMatrix matrix={metrics.confusion_matrix} classNames={classNames} />
        </div>
      )}

      {/* クラス別メトリクス（分類問題のみ） */}
      {!isRegression && metrics.class_metrics && metrics.class_metrics.length > 0 && (
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">クラス別の性能</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-zinc-800">
                  <th className="text-left py-3 px-4 text-sm font-medium text-zinc-400">クラス</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-zinc-400">適合率</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-zinc-400">再現率</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-zinc-400">F1スコア</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-zinc-400">サポート</th>
                </tr>
              </thead>
              <tbody>
                {metrics.class_metrics.map((cm, idx) => (
                  <tr key={idx} className="border-b border-zinc-800/50 hover:bg-zinc-800/30">
                    <td className="py-3 px-4 text-sm text-white font-medium">{cm.class_name}</td>
                    <td className="py-3 px-4 text-sm text-zinc-300 text-right font-mono">
                      {cm.precision.toFixed(4)}
                    </td>
                    <td className="py-3 px-4 text-sm text-zinc-300 text-right font-mono">
                      {cm.recall.toFixed(4)}
                    </td>
                    <td className="py-3 px-4 text-sm text-zinc-300 text-right font-mono">
                      {cm.f1_score.toFixed(4)}
                    </td>
                    <td className="py-3 px-4 text-sm text-zinc-400 text-right">{cm.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ファイル別の予測結果（オプション） */}
      {predictions && predictions.length > 0 && (
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">ファイル別の予測結果（最初の100件）</h3>
          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <table className="w-full">
              <thead className="sticky top-0 bg-zinc-900">
                <tr className="border-b border-zinc-800">
                  <th className="text-left py-3 px-4 text-sm font-medium text-zinc-400">ファイル名</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-zinc-400">実際</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-zinc-400">予測</th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-zinc-400">信頼度</th>
                  <th className="text-center py-3 px-4 text-sm font-medium text-zinc-400">結果</th>
                </tr>
              </thead>
              <tbody>
                {predictions.slice(0, 100).map((pred, idx) => (
                  <tr key={idx} className="border-b border-zinc-800/50 hover:bg-zinc-800/30">
                    <td className="py-2 px-4 text-xs text-zinc-300 font-mono max-w-xs truncate" title={pred.filename}>
                      {pred.filename}
                    </td>
                    <td className="py-2 px-4 text-xs text-zinc-300">{pred.true_label}</td>
                    <td className="py-2 px-4 text-xs text-zinc-300">{pred.predicted_label}</td>
                    <td className="py-2 px-4 text-xs text-zinc-400 text-right font-mono">
                      {pred.confidence}
                    </td>
                    <td className="py-2 px-4 text-center">
                      {pred.correct ? (
                        <CheckCircle2 className="w-4 h-4 text-green-400 mx-auto" />
                      ) : (
                        <XCircle className="w-4 h-4 text-red-400 mx-auto" />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {predictions.length > 100 && (
            <p className="text-xs text-zinc-500 mt-2">
              {predictions.length}件中100件を表示しています
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// メトリクスカードコンポーネント
interface MetricCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  color: 'violet' | 'blue' | 'green' | 'cyan';
}

function MetricCard({ icon, label, value, color }: MetricCardProps) {
  const colorClasses = {
    violet: 'bg-violet-500/10 border-violet-500/50 text-violet-400',
    blue: 'bg-blue-500/10 border-blue-500/50 text-blue-400',
    green: 'bg-green-500/10 border-green-500/50 text-green-400',
    cyan: 'bg-cyan-500/10 border-cyan-500/50 text-cyan-400',
  };

  return (
    <div className={`rounded-xl p-6 border ${colorClasses[color]}`}>
      <div className="flex items-center gap-3 mb-2">
        {icon}
        <span className="text-sm font-medium text-zinc-400">{label}</span>
      </div>
      <div className="text-3xl font-bold text-white">{value}</div>
    </div>
  );
}

