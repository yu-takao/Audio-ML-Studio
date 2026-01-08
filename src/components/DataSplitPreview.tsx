/**
 * データ分割プレビューコンポーネント
 * Train/Validation/Testの分割結果を視覚的に表示
 */

import { BarChart3, Database, FlaskConical, TestTube2 } from 'lucide-react';
import type { SplitStats } from '../utils/dataSplitter';

interface DataSplitPreviewProps {
  stats: SplitStats;
  classes: string[];
}

export function DataSplitPreview({ stats, classes }: DataSplitPreviewProps) {
  const total = stats.train.total + stats.validation.total + stats.test.total;
  
  const getPercent = (count: number) => ((count / total) * 100).toFixed(1);
  
  // 各セットの色
  const colors = {
    train: { bg: 'bg-emerald-500', text: 'text-emerald-400', border: 'border-emerald-500/30' },
    validation: { bg: 'bg-amber-500', text: 'text-amber-400', border: 'border-amber-500/30' },
    test: { bg: 'bg-sky-500', text: 'text-sky-400', border: 'border-sky-500/30' },
  };
  
  return (
    <div className="space-y-4">
      {/* 全体のバー表示 */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-zinc-400">データ分割</span>
          <span className="text-zinc-500">合計 {total} ファイル</span>
        </div>
        <div className="h-4 rounded-full overflow-hidden flex bg-zinc-700">
          <div
            className={`${colors.train.bg} transition-all`}
            style={{ width: `${getPercent(stats.train.total)}%` }}
            title={`Train: ${stats.train.total}`}
          />
          <div
            className={`${colors.validation.bg} transition-all`}
            style={{ width: `${getPercent(stats.validation.total)}%` }}
            title={`Validation: ${stats.validation.total}`}
          />
          <div
            className={`${colors.test.bg} transition-all`}
            style={{ width: `${getPercent(stats.test.total)}%` }}
            title={`Test: ${stats.test.total}`}
          />
        </div>
      </div>
      
      {/* 各セットの詳細 */}
      <div className="grid grid-cols-3 gap-3">
        {/* Train */}
        <div className={`rounded-lg border ${colors.train.border} bg-zinc-900/50 p-3`}>
          <div className="flex items-center gap-2 mb-2">
            <Database className={`w-4 h-4 ${colors.train.text}`} />
            <span className={`text-sm font-medium ${colors.train.text}`}>Train</span>
          </div>
          <div className="text-2xl font-bold text-white">{stats.train.total}</div>
          <div className="text-xs text-zinc-500">{getPercent(stats.train.total)}%</div>
          <div className="mt-2 text-xs text-zinc-400">
            拡張対象（クラス分布調整可能）
          </div>
        </div>
        
        {/* Validation */}
        <div className={`rounded-lg border ${colors.validation.border} bg-zinc-900/50 p-3`}>
          <div className="flex items-center gap-2 mb-2">
            <FlaskConical className={`w-4 h-4 ${colors.validation.text}`} />
            <span className={`text-sm font-medium ${colors.validation.text}`}>Validation</span>
          </div>
          <div className="text-2xl font-bold text-white">{stats.validation.total}</div>
          <div className="text-xs text-zinc-500">{getPercent(stats.validation.total)}%</div>
          <div className="mt-2 text-xs text-zinc-400">
            学習中の評価用（元データのみ）
          </div>
        </div>
        
        {/* Test */}
        <div className={`rounded-lg border ${colors.test.border} bg-zinc-900/50 p-3`}>
          <div className="flex items-center gap-2 mb-2">
            <TestTube2 className={`w-4 h-4 ${colors.test.text}`} />
            <span className={`text-sm font-medium ${colors.test.text}`}>Test</span>
          </div>
          <div className="text-2xl font-bold text-white">{stats.test.total}</div>
          <div className="text-xs text-zinc-500">{getPercent(stats.test.total)}%</div>
          <div className="mt-2 text-xs text-zinc-400">
            最終評価用（完全未知データ）
          </div>
        </div>
      </div>
      
      {/* クラス別の分布（折りたたみ可能） */}
      <details className="group">
        <summary className="flex items-center gap-2 cursor-pointer text-sm text-zinc-400 hover:text-zinc-300">
          <BarChart3 className="w-4 h-4" />
          クラス別の分布を表示
        </summary>
        <div className="mt-3 space-y-2 max-h-48 overflow-y-auto">
          {classes.map((cls) => {
            const trainCount = stats.train.byClass.get(cls) || 0;
            const valCount = stats.validation.byClass.get(cls) || 0;
            const testCount = stats.test.byClass.get(cls) || 0;
            const classTotal = trainCount + valCount + testCount;
            
            return (
              <div key={cls} className="bg-zinc-800/50 rounded-lg p-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm text-white truncate" title={cls}>{cls}</span>
                  <span className="text-xs text-zinc-500">{classTotal}</span>
                </div>
                <div className="h-2 rounded-full overflow-hidden flex bg-zinc-700">
                  <div
                    className={colors.train.bg}
                    style={{ width: `${(trainCount / classTotal) * 100}%` }}
                  />
                  <div
                    className={colors.validation.bg}
                    style={{ width: `${(valCount / classTotal) * 100}%` }}
                  />
                  <div
                    className={colors.test.bg}
                    style={{ width: `${(testCount / classTotal) * 100}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-zinc-500 mt-1">
                  <span>T: {trainCount}</span>
                  <span>V: {valCount}</span>
                  <span>Te: {testCount}</span>
                </div>
              </div>
            );
          })}
        </div>
      </details>
    </div>
  );
}




