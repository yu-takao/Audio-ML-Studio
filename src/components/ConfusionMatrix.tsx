import { useMemo } from 'react';

interface ConfusionMatrixProps {
  matrix: number[][];
  classNames: string[];
}

/**
 * 混同行列を視覚化するコンポーネント
 */
export function ConfusionMatrix({ matrix, classNames }: ConfusionMatrixProps) {
  // 最大値を計算（色の正規化用）
  const maxValue = useMemo(() => {
    return Math.max(...matrix.flat());
  }, [matrix]);

  // 色の強度を計算（0-1）
  const getColorIntensity = (value: number) => {
    if (maxValue === 0) return 0;
    return value / maxValue;
  };

  // 背景色を取得（正解は緑系、不正解は赤系）
  const getCellColor = (row: number, col: number, value: number) => {
    const intensity = getColorIntensity(value);
    
    if (value === 0) {
      return 'bg-zinc-800';
    }
    
    if (row === col) {
      // 対角線（正解）: 緑系
      const greenShade = Math.min(900, Math.max(700, Math.floor(700 + intensity * 200)));
      return `bg-green-${greenShade}/20 border-green-500/30`;
    } else {
      // 非対角線（誤り）: 赤系
      const redShade = Math.min(900, Math.max(700, Math.floor(700 + intensity * 200)));
      return `bg-red-${redShade}/20 border-red-500/30`;
    }
  };

  // テキスト色を取得
  const getTextColor = (row: number, col: number, value: number) => {
    if (value === 0) return 'text-zinc-600';
    return row === col ? 'text-green-300' : 'text-red-300';
  };

  return (
    <div className="w-full overflow-auto">
      <div className="inline-block min-w-full">
        {/* ヘッダー */}
        <div className="mb-4">
          <p className="text-sm text-zinc-400">縦軸: 実際のクラス / 横軸: 予測されたクラス</p>
        </div>

        {/* テーブル */}
        <div className="bg-zinc-900/50 rounded-lg p-4 border border-zinc-800">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="p-2 text-xs font-medium text-zinc-500"></th>
                <th 
                  colSpan={classNames.length} 
                  className="p-2 text-xs font-medium text-zinc-400 border-b border-zinc-700"
                >
                  予測
                </th>
              </tr>
              <tr>
                <th className="p-2 text-xs font-medium text-zinc-500"></th>
                {classNames.map((name, idx) => (
                  <th 
                    key={idx} 
                    className="p-2 text-xs font-medium text-zinc-400 border-b border-zinc-700"
                    style={{ minWidth: '80px' }}
                  >
                    <div className="truncate max-w-[100px]" title={name}>
                      {name}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  {rowIdx === 0 && (
                    <th 
                      rowSpan={matrix.length}
                      className="p-2 text-xs font-medium text-zinc-400 border-r border-zinc-700 align-middle"
                      style={{ writingMode: 'vertical-rl', textOrientation: 'mixed' }}
                    >
                      実際
                    </th>
                  )}
                  <th 
                    className="p-2 text-xs font-medium text-zinc-400 border-r border-zinc-700 text-right"
                    style={{ minWidth: '80px' }}
                  >
                    <div className="truncate max-w-[100px]" title={classNames[rowIdx]}>
                      {classNames[rowIdx]}
                    </div>
                  </th>
                  {row.map((value, colIdx) => (
                    <td
                      key={colIdx}
                      className={`p-3 text-center border border-zinc-800/50 ${getCellColor(rowIdx, colIdx, value)}`}
                      style={{ minWidth: '80px' }}
                    >
                      <div className={`font-mono text-sm font-bold ${getTextColor(rowIdx, colIdx, value)}`}>
                        {value}
                      </div>
                      {value > 0 && (
                        <div className="text-xs text-zinc-500 mt-1">
                          {((value / row.reduce((a, b) => a + b, 0)) * 100).toFixed(1)}%
                        </div>
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* 凡例 */}
        <div className="mt-4 flex gap-6 text-xs text-zinc-400">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-700/20 border border-green-500/30 rounded"></div>
            <span>正解</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-700/20 border border-red-500/30 rounded"></div>
            <span>誤り</span>
          </div>
        </div>
      </div>
    </div>
  );
}

