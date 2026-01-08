import { useMemo } from 'react';
import {
  BarChart3,
  Target,
  CheckCircle2,
  TrendingUp,
} from 'lucide-react';

interface EvaluationResult {
  accuracy: number;
  loss: number;
  confusionMatrix: number[][];
  classNames: string[];
  perClassMetrics: {
    className: string;
    precision: number;
    recall: number;
    f1Score: number;
    support: number;
  }[];
  predictions: number[];
  actuals: number[];
}

interface ModelEvaluationProps {
  result: EvaluationResult;
}

export function ModelEvaluation({ result }: ModelEvaluationProps) {
  const { accuracy, confusionMatrix, classNames, perClassMetrics } = result;

  // 平均メトリクスを計算
  const avgMetrics = useMemo(() => {
    const totalSupport = perClassMetrics.reduce((sum, m) => sum + m.support, 0);
    const weightedPrecision = perClassMetrics.reduce(
      (sum, m) => sum + m.precision * m.support,
      0
    ) / totalSupport;
    const weightedRecall = perClassMetrics.reduce(
      (sum, m) => sum + m.recall * m.support,
      0
    ) / totalSupport;
    const weightedF1 = perClassMetrics.reduce(
      (sum, m) => sum + m.f1Score * m.support,
      0
    ) / totalSupport;

    return {
      precision: weightedPrecision,
      recall: weightedRecall,
      f1Score: weightedF1,
    };
  }, [perClassMetrics]);

  // 混同行列の最大値を取得（色の正規化用）
  const maxConfusionValue = useMemo(() => {
    let max = 0;
    confusionMatrix.forEach((row) => {
      row.forEach((val) => {
        if (val > max) max = val;
      });
    });
    return max || 1;
  }, [confusionMatrix]);

  return (
    <div className="space-y-6">
      {/* 全体サマリー */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-700">
          <div className="flex items-center gap-2 text-sm text-zinc-500 mb-1">
            <Target className="w-4 h-4" />
            テスト精度
          </div>
          <div className="text-3xl font-bold text-emerald-400">
            {(accuracy * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-700">
          <div className="flex items-center gap-2 text-sm text-zinc-500 mb-1">
            <TrendingUp className="w-4 h-4" />
            適合率 (Precision)
          </div>
          <div className="text-2xl font-bold text-white">
            {(avgMetrics.precision * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-700">
          <div className="flex items-center gap-2 text-sm text-zinc-500 mb-1">
            <CheckCircle2 className="w-4 h-4" />
            再現率 (Recall)
          </div>
          <div className="text-2xl font-bold text-white">
            {(avgMetrics.recall * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-700">
          <div className="flex items-center gap-2 text-sm text-zinc-500 mb-1">
            <BarChart3 className="w-4 h-4" />
            F1スコア
          </div>
          <div className="text-2xl font-bold text-violet-400">
            {(avgMetrics.f1Score * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* 混同行列 */}
      <div className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-violet-500/20">
            <BarChart3 className="w-5 h-5 text-violet-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">混同行列</h3>
            <p className="text-sm text-zinc-400">行: 実際のクラス, 列: 予測クラス</p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                <th className="p-2 text-left text-xs text-zinc-500">実際 ＼ 予測</th>
                {classNames.map((name) => (
                  <th key={name} className="p-2 text-center text-xs text-zinc-400 min-w-[60px]">
                    <div className="truncate max-w-[80px]" title={name}>
                      {name}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {confusionMatrix.map((row, i) => (
                <tr key={i}>
                  <td className="p-2 text-xs text-zinc-400 font-medium">
                    <div className="truncate max-w-[100px]" title={classNames[i]}>
                      {classNames[i]}
                    </div>
                  </td>
                  {row.map((value, j) => {
                    const isCorrect = i === j;
                    const intensity = value / maxConfusionValue;
                    return (
                      <td key={j} className="p-1">
                        <div
                          className={`
                            w-full h-12 rounded flex items-center justify-center text-sm font-medium
                            ${isCorrect
                              ? 'bg-emerald-500'
                              : value > 0
                                ? 'bg-red-500'
                                : 'bg-zinc-800'
                            }
                          `}
                          style={{
                            opacity: value > 0 ? 0.3 + intensity * 0.7 : 1,
                          }}
                        >
                          <span className={value > 0 ? 'text-white' : 'text-zinc-600'}>
                            {value}
                          </span>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* クラス別メトリクス */}
      <div className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-cyan-500/20">
            <Target className="w-5 h-5 text-cyan-400" />
          </div>
          <h3 className="text-lg font-semibold text-white">クラス別メトリクス</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-zinc-700">
                <th className="p-3 text-left text-sm text-zinc-400">クラス</th>
                <th className="p-3 text-center text-sm text-zinc-400">適合率</th>
                <th className="p-3 text-center text-sm text-zinc-400">再現率</th>
                <th className="p-3 text-center text-sm text-zinc-400">F1スコア</th>
                <th className="p-3 text-center text-sm text-zinc-400">サンプル数</th>
              </tr>
            </thead>
            <tbody>
              {perClassMetrics.map((metric, i) => (
                <tr key={i} className="border-b border-zinc-800">
                  <td className="p-3 text-white font-medium">
                    <div className="truncate max-w-[150px]" title={metric.className}>
                      {metric.className}
                    </div>
                  </td>
                  <td className="p-3 text-center">
                    <span className={`font-medium ${metric.precision >= 0.8 ? 'text-emerald-400' : metric.precision >= 0.6 ? 'text-amber-400' : 'text-red-400'}`}>
                      {(metric.precision * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="p-3 text-center">
                    <span className={`font-medium ${metric.recall >= 0.8 ? 'text-emerald-400' : metric.recall >= 0.6 ? 'text-amber-400' : 'text-red-400'}`}>
                      {(metric.recall * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="p-3 text-center">
                    <span className={`font-medium ${metric.f1Score >= 0.8 ? 'text-emerald-400' : metric.f1Score >= 0.6 ? 'text-amber-400' : 'text-red-400'}`}>
                      {(metric.f1Score * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="p-3 text-center text-zinc-400">
                    {metric.support}
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="border-t border-zinc-600 bg-zinc-900/30">
                <td className="p-3 text-white font-semibold">加重平均</td>
                <td className="p-3 text-center font-semibold text-white">
                  {(avgMetrics.precision * 100).toFixed(1)}%
                </td>
                <td className="p-3 text-center font-semibold text-white">
                  {(avgMetrics.recall * 100).toFixed(1)}%
                </td>
                <td className="p-3 text-center font-semibold text-violet-400">
                  {(avgMetrics.f1Score * 100).toFixed(1)}%
                </td>
                <td className="p-3 text-center text-zinc-400">
                  {perClassMetrics.reduce((sum, m) => sum + m.support, 0)}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>

      {/* 評価の解説 */}
      <div className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-4">
        <div className="text-sm text-zinc-400 space-y-2">
          <div className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
            <span><strong className="text-white">適合率 (Precision)</strong>: 予測が正しかった割合。「このクラスと予測したうち、実際に正しかった割合」</span>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
            <span><strong className="text-white">再現率 (Recall)</strong>: 実際のデータを正しく予測できた割合。「実際にこのクラスだったデータのうち、正しく予測できた割合」</span>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-violet-400 mt-0.5 flex-shrink-0" />
            <span><strong className="text-white">F1スコア</strong>: 適合率と再現率の調和平均。両方のバランスを見る指標。</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export type { EvaluationResult };



