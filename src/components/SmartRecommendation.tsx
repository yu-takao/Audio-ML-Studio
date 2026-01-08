import { useMemo } from 'react';
import { Lightbulb, AlertTriangle, CheckCircle2, Info } from 'lucide-react';

interface DatasetStats {
  totalSamples: number;
  numClasses: number;
  minSamplesPerClass: number;
  maxSamplesPerClass: number;
  imbalanceRatio: number;
}

interface RecommendedParams {
  epochs: { value: number; reason: string };
  batchSize: { value: number; reason: string };
  learningRate: { value: number; reason: string };
  validationSplit: { value: number; reason: string };
  testSplit: { value: number; reason: string };
}

interface SmartRecommendationProps {
  stats: DatasetStats | null;
  currentParams: {
    epochs: number;
    batchSize: number;
    learningRate: number;
    validationSplit: number;
    testSplit: number;
  };
  onApplyRecommendation: (params: Partial<{
    epochs: number;
    batchSize: number;
    learningRate: number;
    validationSplit: number;
    testSplit: number;
  }>) => void;
}

export function SmartRecommendation({
  stats,
  currentParams,
  onApplyRecommendation,
}: SmartRecommendationProps) {
  const recommendations = useMemo<RecommendedParams | null>(() => {
    if (!stats) return null;

    const { totalSamples, numClasses, minSamplesPerClass, imbalanceRatio } = stats;

    // エポック数の推奨
    let epochsValue: number;
    let epochsReason: string;
    
    if (totalSamples < 500) {
      epochsValue = 30;
      epochsReason = 'データ数が少ない（<500）ため、過学習を防ぐために少なめに設定';
    } else if (totalSamples < 1000) {
      epochsValue = 50;
      epochsReason = 'データ数が中程度（500-1000）のため、標準的な値';
    } else if (totalSamples < 5000) {
      epochsValue = 80;
      epochsReason = 'データ数が多め（1000-5000）のため、十分な学習が可能';
    } else {
      epochsValue = 100;
      epochsReason = 'データ数が十分（>5000）のため、より多くの学習が可能';
    }

    // バッチサイズの推奨
    let batchSizeValue: number;
    let batchSizeReason: string;
    
    if (totalSamples < 500) {
      batchSizeValue = 16;
      batchSizeReason = 'データ数が少ないため、細かく更新して学習効率を上げる';
    } else if (totalSamples < 2000) {
      batchSizeValue = 32;
      batchSizeReason = '標準的なデータ量に適した値';
    } else {
      batchSizeValue = 64;
      batchSizeReason = 'データ数が多いため、大きめのバッチで安定した学習が可能';
    }
    
    // クラス不均衡がある場合はバッチサイズを小さめに
    if (imbalanceRatio > 3) {
      batchSizeValue = Math.max(16, batchSizeValue - 16);
      batchSizeReason += '（クラス不均衡があるため調整）';
    }

    // 学習率の推奨
    let learningRateValue: number;
    let learningRateReason: string;
    
    if (numClasses <= 3) {
      learningRateValue = 0.001;
      learningRateReason = 'クラス数が少ない（≤3）シンプルなタスクのため標準値';
    } else if (numClasses <= 10) {
      learningRateValue = 0.001;
      learningRateReason = '中程度のクラス数に適した標準値';
    } else {
      learningRateValue = 0.0005;
      learningRateReason = 'クラス数が多い（>10）複雑なタスクのため慎重に学習';
    }

    // 検証データ割合の推奨
    let validationSplitValue: number;
    let validationSplitReason: string;
    
    if (totalSamples < 500) {
      validationSplitValue = 0.15;
      validationSplitReason = 'データ数が少ないため、訓練データを確保しつつ過学習監視';
    } else if (minSamplesPerClass < 50) {
      validationSplitValue = 0.15;
      validationSplitReason = '少数クラスのサンプルが少ないため控えめに';
    } else {
      validationSplitValue = 0.2;
      validationSplitReason = 'データ数が十分なため標準的な割合';
    }

    // テストデータ割合の推奨
    let testSplitValue: number;
    let testSplitReason: string;
    
    if (totalSamples < 500) {
      testSplitValue = 0.1;
      testSplitReason = 'データ数が少ないため、訓練データを確保しつつ評価';
    } else if (minSamplesPerClass < 30) {
      testSplitValue = 0.1;
      testSplitReason = '少数クラスの評価サンプルを確保するため控えめに';
    } else {
      testSplitValue = 0.15;
      testSplitReason = 'データ数が十分なため信頼性の高い評価が可能';
    }

    return {
      epochs: { value: epochsValue, reason: epochsReason },
      batchSize: { value: batchSizeValue, reason: batchSizeReason },
      learningRate: { value: learningRateValue, reason: learningRateReason },
      validationSplit: { value: validationSplitValue, reason: validationSplitReason },
      testSplit: { value: testSplitValue, reason: testSplitReason },
    };
  }, [stats]);

  if (!stats || !recommendations) {
    return null;
  }

  // 現在の設定と推奨値の差異をチェック
  const hasDifferences = 
    currentParams.epochs !== recommendations.epochs.value ||
    currentParams.batchSize !== recommendations.batchSize.value ||
    Math.abs(currentParams.learningRate - recommendations.learningRate.value) > 0.0001 ||
    Math.abs(currentParams.validationSplit - recommendations.validationSplit.value) > 0.01 ||
    Math.abs(currentParams.testSplit - recommendations.testSplit.value) > 0.01;

  return (
    <div className="bg-gradient-to-r from-amber-500/10 to-orange-500/10 rounded-xl border border-amber-500/30 p-4">
      <div className="flex items-center gap-2 mb-3">
        <Lightbulb className="w-5 h-5 text-amber-400" />
        <h3 className="font-semibold text-white">データに基づく推奨設定</h3>
      </div>

      {/* データセット統計 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4 text-sm">
        <div className="bg-zinc-900/50 rounded-lg p-2">
          <div className="text-zinc-500 text-xs">総データ数</div>
          <div className="text-white font-medium">{stats.totalSamples}</div>
        </div>
        <div className="bg-zinc-900/50 rounded-lg p-2">
          <div className="text-zinc-500 text-xs">クラス数</div>
          <div className="text-white font-medium">{stats.numClasses}</div>
        </div>
        <div className="bg-zinc-900/50 rounded-lg p-2">
          <div className="text-zinc-500 text-xs">最小サンプル/クラス</div>
          <div className="text-white font-medium">{stats.minSamplesPerClass}</div>
        </div>
        <div className="bg-zinc-900/50 rounded-lg p-2">
          <div className="text-zinc-500 text-xs">不均衡度</div>
          <div className={`font-medium ${stats.imbalanceRatio > 3 ? 'text-amber-400' : 'text-emerald-400'}`}>
            {stats.imbalanceRatio.toFixed(1)}x
          </div>
        </div>
      </div>

      {/* 推奨値テーブル */}
      <div className="space-y-2 mb-4">
        {[
          { key: 'epochs', label: 'エポック数', current: currentParams.epochs, rec: recommendations.epochs },
          { key: 'batchSize', label: 'バッチサイズ', current: currentParams.batchSize, rec: recommendations.batchSize },
          { key: 'learningRate', label: '学習率', current: currentParams.learningRate, rec: recommendations.learningRate },
          { key: 'validationSplit', label: '検証データ', current: `${(currentParams.validationSplit * 100).toFixed(0)}%`, rec: { value: `${(recommendations.validationSplit.value * 100).toFixed(0)}%`, reason: recommendations.validationSplit.reason } },
          { key: 'testSplit', label: 'テストデータ', current: `${(currentParams.testSplit * 100).toFixed(0)}%`, rec: { value: `${(recommendations.testSplit.value * 100).toFixed(0)}%`, reason: recommendations.testSplit.reason } },
        ].map((item) => {
          const currentVal = typeof item.current === 'number' ? item.current : item.current;
          const recVal = typeof item.rec.value === 'number' ? item.rec.value : item.rec.value;
          const isMatch = currentVal.toString() === recVal.toString();
          
          return (
            <div key={item.key} className="flex items-center gap-3 p-2 bg-zinc-900/30 rounded-lg">
              {isMatch ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
              ) : (
                <Info className="w-4 h-4 text-amber-400 flex-shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-zinc-300">{item.label}:</span>
                  <span className={`text-sm font-medium ${isMatch ? 'text-emerald-400' : 'text-white'}`}>
                    {currentVal}
                  </span>
                  {!isMatch && (
                    <span className="text-xs text-amber-400">
                      → 推奨: {recVal}
                    </span>
                  )}
                </div>
                <div className="text-xs text-zinc-500 truncate" title={item.rec.reason}>
                  {item.rec.reason}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* 一括適用ボタン */}
      {hasDifferences && (
        <button
          onClick={() => onApplyRecommendation({
            epochs: recommendations.epochs.value,
            batchSize: recommendations.batchSize.value,
            learningRate: recommendations.learningRate.value,
            validationSplit: recommendations.validationSplit.value,
            testSplit: recommendations.testSplit.value,
          })}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/50 text-amber-300 rounded-lg transition-colors"
        >
          <Lightbulb className="w-4 h-4" />
          推奨設定を一括適用
        </button>
      )}

      {!hasDifferences && (
        <div className="flex items-center justify-center gap-2 text-sm text-emerald-400">
          <CheckCircle2 className="w-4 h-4" />
          現在の設定は推奨値と一致しています
        </div>
      )}

      {/* 注意書き */}
      <div className="mt-3 flex items-start gap-2 text-xs text-zinc-500">
        <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
        <span>
          これらは統計的な推奨値です。実際の最適値は訓練結果を見ながら調整してください。
        </span>
      </div>
    </div>
  );
}

export type { DatasetStats };



