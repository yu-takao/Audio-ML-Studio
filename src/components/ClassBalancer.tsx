import { useState, useCallback } from 'react';
import {
  Scale,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Sparkles,
  BarChart3,
} from 'lucide-react';
import {
  applyTimeShift,
  applyGain,
  applyPitchShift,
  samplesToWavBlob,
} from '../utils/audioAugmentation';

interface ClassInfo {
  name: string;
  count: number;
  files: FileInfo[];
}

interface FileInfo {
  file: File;
  path: string;
  folderName: string;
}

interface ClassBalancerProps {
  classDistribution: Map<string, number>;
  filesByClass: Map<string, FileInfo[]>;
  dataFolder: FileSystemDirectoryHandle | null;
  onBalanceComplete: () => void;
}

interface BalancingConfig {
  targetMode: 'max' | 'median' | 'custom';
  customTarget: number;
  augmentationTypes: {
    timeShift: boolean;
    gain: boolean;
    pitch: boolean;
    combined: boolean;
  };
}

interface BalancingProgress {
  isProcessing: boolean;
  currentClass: string;
  processedClasses: number;
  totalClasses: number;
  generatedFiles: number;
  status: string;
}

export function ClassBalancer({
  classDistribution,
  filesByClass,
  dataFolder,
  onBalanceComplete,
}: ClassBalancerProps) {
  const [config, setConfig] = useState<BalancingConfig>({
    targetMode: 'max',
    customTarget: 1000,
    augmentationTypes: {
      timeShift: true,
      gain: true,
      pitch: false,
      combined: true,
    },
  });

  const [progress, setProgress] = useState<BalancingProgress>({
    isProcessing: false,
    currentClass: '',
    processedClasses: 0,
    totalClasses: 0,
    generatedFiles: 0,
    status: '',
  });

  const [completedAction, setCompletedAction] = useState<{
    count: number;
  } | null>(null);

  // クラス情報を計算
  const classes: ClassInfo[] = Array.from(classDistribution.entries())
    .map(([name, count]) => ({
      name,
      count,
      files: filesByClass.get(name) || [],
    }))
    .sort((a, b) => b.count - a.count);

  const maxCount = Math.max(...classes.map((c) => c.count));
  const minCount = Math.min(...classes.map((c) => c.count));
  const medianCount = classes.length > 0
    ? classes[Math.floor(classes.length / 2)].count
    : 0;
  const totalSamples = classes.reduce((sum, c) => sum + c.count, 0);

  // 不均衡度を計算（最大/最小の比率）
  const imbalanceRatio = minCount > 0 ? maxCount / minCount : Infinity;
  const isImbalanced = imbalanceRatio > 1.5;

  // ターゲット数を計算
  const getTargetCount = () => {
    switch (config.targetMode) {
      case 'max':
        return maxCount;
      case 'median':
        return medianCount;
      case 'custom':
        return config.customTarget;
      default:
        return maxCount;
    }
  };

  const targetCount = getTargetCount();

  // 必要な拡張数を計算
  const augmentationPlan = classes.map((cls) => {
    const needed = Math.max(0, targetCount - cls.count);
    return {
      ...cls,
      needed,
      augmentationsPerFile: cls.count > 0 ? Math.ceil(needed / cls.count) : 0,
    };
  });

  const totalNeeded = augmentationPlan.reduce((sum, c) => sum + c.needed, 0);

  // WAVファイルをAudioBufferに変換
  const fileToAudioBuffer = async (file: File): Promise<AudioBuffer> => {
    const arrayBuffer = await file.arrayBuffer();
    const audioContext = new AudioContext();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    await audioContext.close();
    return audioBuffer;
  };

  // AudioBufferをモノラルFloat32Arrayに変換
  const audioBufferToMono = (buffer: AudioBuffer): Float32Array => {
    if (buffer.numberOfChannels === 1) {
      return buffer.getChannelData(0).slice();
    }
    const left = buffer.getChannelData(0);
    const right = buffer.getChannelData(1);
    const mono = new Float32Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) {
      mono[i] = (left[i] + right[i]) / 2;
    }
    return mono;
  };

  // ランダム値生成
  const randomInRange = (min: number, max: number) => min + Math.random() * (max - min);

  // パスからディレクトリハンドルを取得
  const getDirectoryHandle = async (
    rootHandle: FileSystemDirectoryHandle,
    path: string
  ): Promise<FileSystemDirectoryHandle> => {
    const parts = path.split('/').filter(p => p.length > 0);
    let currentHandle = rootHandle;
    
    // 最後の要素（ファイル名）を除いたパスを辿る
    for (let i = 0; i < parts.length - 1; i++) {
      currentHandle = await currentHandle.getDirectoryHandle(parts[i], { create: true });
    }
    
    return currentHandle;
  };

  // バランシングを実行
  const executeBalancing = useCallback(async () => {
    if (!dataFolder) return;

    setProgress({
      isProcessing: true,
      currentClass: '',
      processedClasses: 0,
      totalClasses: augmentationPlan.filter((c) => c.needed > 0).length,
      generatedFiles: 0,
      status: '開始中...',
    });
    setCompletedAction(null);

    let totalGenerated = 0;
    let processedClasses = 0;

    for (const classInfo of augmentationPlan) {
      if (classInfo.needed <= 0) continue;

      setProgress((prev) => ({
        ...prev,
        currentClass: classInfo.name,
        status: `${classInfo.name} を処理中...`,
      }));

      let classGenerated = 0;
      const filesInClass = classInfo.files;
      let fileIndex = 0;

      while (classGenerated < classInfo.needed && filesInClass.length > 0) {
        const fileInfo = filesInClass[fileIndex % filesInClass.length];
        
        try {
          const audioBuffer = await fileToAudioBuffer(fileInfo.file);
          const samples = audioBufferToMono(audioBuffer);
          const sampleRate = audioBuffer.sampleRate;
          const baseName = fileInfo.file.name.replace(/\.wav$/i, '');

          // 拡張を生成
          const augmentations: { name: string; samples: Float32Array }[] = [];

          if (config.augmentationTypes.timeShift) {
            const shiftSamples = Math.round(randomInRange(-sampleRate * 0.05, sampleRate * 0.05));
            augmentations.push({
              name: `${baseName}_bal_ts_${classGenerated + 1}.wav`,
              samples: applyTimeShift(samples, shiftSamples),
            });
          }

          if (config.augmentationTypes.gain) {
            const gainDb = randomInRange(-3, 3);
            augmentations.push({
              name: `${baseName}_bal_gain_${classGenerated + 1}.wav`,
              samples: applyGain(samples, gainDb),
            });
          }

          if (config.augmentationTypes.pitch) {
            const semitones = randomInRange(-0.5, 0.5);
            augmentations.push({
              name: `${baseName}_bal_pitch_${classGenerated + 1}.wav`,
              samples: applyPitchShift(samples, semitones),
            });
          }

          if (config.augmentationTypes.combined) {
            let augmented = samples.slice();
            augmented = applyTimeShift(augmented, Math.round(randomInRange(-sampleRate * 0.03, sampleRate * 0.03)));
            augmented = applyGain(augmented, randomInRange(-2, 2));
            augmentations.push({
              name: `${baseName}_bal_comb_${classGenerated + 1}.wav`,
              samples: augmented,
            });
          }

          // ファイルを元のフォルダに保存
          for (const aug of augmentations) {
            if (classGenerated >= classInfo.needed) break;

            try {
              // 元のファイルと同じディレクトリに保存
              const dirHandle = await getDirectoryHandle(dataFolder, fileInfo.path);
              
              const blob = samplesToWavBlob(aug.samples, sampleRate);
              const fileHandle = await dirHandle.getFileHandle(aug.name, { create: true });
              const writable = await fileHandle.createWritable();
              await writable.write(blob);
              await writable.close();

              classGenerated++;
              totalGenerated++;

              setProgress((prev) => ({
                ...prev,
                generatedFiles: totalGenerated,
              }));
            } catch (writeErr) {
              console.warn(`Failed to write file: ${aug.name}`, writeErr);
            }
          }
        } catch (err) {
          console.warn(`Failed to process file: ${fileInfo.file.name}`, err);
        }

        fileIndex++;
      }

      processedClasses++;
      setProgress((prev) => ({
        ...prev,
        processedClasses,
      }));
    }

    setProgress((prev) => ({
      ...prev,
      isProcessing: false,
      status: '完了！',
    }));

    setCompletedAction({ count: totalGenerated });

    onBalanceComplete();
  }, [dataFolder, augmentationPlan, config, onBalanceComplete]);

  return (
    <div className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-amber-500/20">
          <Scale className="w-5 h-5 text-amber-400" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">クラスバランシング</h2>
          <p className="text-sm text-zinc-400">
            データ拡張でクラス分布の偏りを解消します
          </p>
        </div>
      </div>

      {/* 不均衡度の警告 */}
      {isImbalanced && (
        <div className="mb-4 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0" />
          <div>
            <div className="text-amber-300 font-medium">クラス不均衡を検出</div>
            <div className="text-sm text-amber-400/80">
              最大/最小比率: {imbalanceRatio.toFixed(1)}x（{maxCount} / {minCount}）
            </div>
          </div>
        </div>
      )}

      {/* 現在の分布 */}
      <div className="mb-4">
        <div className="text-sm text-zinc-400 mb-2 flex items-center gap-2">
          <BarChart3 className="w-4 h-4" />
          現在のクラス分布
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 max-h-48 overflow-y-auto">
          {classes.map((cls) => {
            const percentage = ((cls.count / maxCount) * 100).toFixed(0);
            const isMin = cls.count === minCount;
            const isMax = cls.count === maxCount;
            return (
              <div
                key={cls.name}
                className={`
                  bg-zinc-900/50 rounded-lg p-2 border
                  ${isMin ? 'border-red-500/50' : isMax ? 'border-emerald-500/50' : 'border-zinc-700'}
                `}
              >
                <div className="text-xs text-zinc-400 truncate" title={cls.name}>
                  {cls.name}
                </div>
                <div className="flex items-baseline gap-1">
                  <span className={`text-lg font-bold ${isMin ? 'text-red-400' : isMax ? 'text-emerald-400' : 'text-white'}`}>
                    {cls.count}
                  </span>
                </div>
                <div className="mt-1 h-1 bg-zinc-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${isMin ? 'bg-red-500' : isMax ? 'bg-emerald-500' : 'bg-violet-500'}`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 設定 */}
      <div className="mb-4 p-4 bg-zinc-900/50 rounded-lg space-y-4">
        {/* ターゲットモード */}
        <div>
          <label className="text-sm text-zinc-400 mb-2 block">ターゲットサンプル数</label>
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => setConfig({ ...config, targetMode: 'max' })}
              className={`px-3 py-2 rounded-lg text-sm transition-all ${
                config.targetMode === 'max'
                  ? 'bg-violet-500 text-white'
                  : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
              }`}
            >
              最大値に合わせる ({maxCount})
            </button>
            <button
              onClick={() => setConfig({ ...config, targetMode: 'median' })}
              className={`px-3 py-2 rounded-lg text-sm transition-all ${
                config.targetMode === 'median'
                  ? 'bg-violet-500 text-white'
                  : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
              }`}
            >
              中央値に合わせる ({medianCount})
            </button>
            <button
              onClick={() => setConfig({ ...config, targetMode: 'custom' })}
              className={`px-3 py-2 rounded-lg text-sm transition-all ${
                config.targetMode === 'custom'
                  ? 'bg-violet-500 text-white'
                  : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
              }`}
            >
              カスタム
            </button>
          </div>
          {config.targetMode === 'custom' && (
            <input
              type="number"
              value={config.customTarget}
              onChange={(e) => setConfig({ ...config, customTarget: parseInt(e.target.value) || 0 })}
              className="mt-2 w-32 bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white"
              min={minCount}
            />
          )}
        </div>

        {/* 拡張タイプ */}
        <div>
          <label className="text-sm text-zinc-400 mb-2 block">使用する拡張タイプ</label>
          <div className="flex gap-2 flex-wrap">
            {[
              { key: 'timeShift', label: '時間シフト' },
              { key: 'gain', label: 'ゲイン変化' },
              { key: 'pitch', label: 'ピッチシフト' },
              { key: 'combined', label: '複合拡張' },
            ].map(({ key, label }) => (
              <button
                key={key}
                onClick={() =>
                  setConfig({
                    ...config,
                    augmentationTypes: {
                      ...config.augmentationTypes,
                      [key]: !config.augmentationTypes[key as keyof typeof config.augmentationTypes],
                    },
                  })
                }
                className={`px-3 py-2 rounded-lg text-sm transition-all ${
                  config.augmentationTypes[key as keyof typeof config.augmentationTypes]
                    ? 'bg-emerald-500 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 拡張計画 */}
      <div className="mb-4 p-4 bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10 rounded-lg border border-violet-500/30">
        <div className="flex items-center gap-2 mb-2">
          <Sparkles className="w-5 h-5 text-violet-400" />
          <span className="text-white font-medium">拡張計画</span>
        </div>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-zinc-400">ターゲット数:</span>
            <span className="text-white ml-2">{targetCount} / クラス</span>
          </div>
          <div>
            <span className="text-zinc-400">生成予定:</span>
            <span className="text-violet-300 ml-2 font-bold">{totalNeeded} ファイル</span>
          </div>
          <div>
            <span className="text-zinc-400">現在の合計:</span>
            <span className="text-white ml-2">{totalSamples} ファイル</span>
          </div>
          <div>
            <span className="text-zinc-400">バランス後:</span>
            <span className="text-emerald-300 ml-2 font-bold">{totalSamples + totalNeeded} ファイル</span>
          </div>
        </div>
      </div>

      {/* 進捗表示 */}
      {progress.isProcessing && (
        <div className="mb-4 p-4 bg-zinc-900/50 rounded-lg">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-zinc-400">{progress.status}</span>
            <span className="text-white">
              {progress.processedClasses} / {progress.totalClasses} クラス
            </span>
          </div>
          <div className="h-2 bg-zinc-700 rounded-full overflow-hidden mb-2">
            <div
              className="h-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all"
              style={{
                width: `${progress.totalClasses > 0 ? (progress.processedClasses / progress.totalClasses) * 100 : 0}%`,
              }}
            />
          </div>
          <div className="text-sm text-zinc-500">
            生成済み: {progress.generatedFiles} ファイル
          </div>
        </div>
      )}

      {/* 実行ボタン */}
      <button
        onClick={executeBalancing}
        disabled={!dataFolder || progress.isProcessing || totalNeeded === 0}
        className={`
          w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all
          ${
            !dataFolder || progress.isProcessing || totalNeeded === 0
              ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
              : 'bg-gradient-to-r from-amber-500 to-orange-500 text-white hover:from-amber-600 hover:to-orange-600 shadow-lg'
          }
        `}
      >
        {progress.isProcessing ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            バランシング中...
          </>
        ) : (
          <>
            <Scale className="w-5 h-5" />
            クラスバランシングを実行
          </>
        )}
      </button>

      {/* 完了メッセージ */}
      {!progress.isProcessing && completedAction && (
        <div className="mt-4 p-4 bg-emerald-500/10 border border-emerald-500/50 rounded-lg flex items-center gap-3">
          <CheckCircle2 className="w-6 h-6 text-emerald-400" />
          <div>
            <div className="text-white font-medium">バランシング完了！</div>
            <div className="text-sm text-emerald-400">
              {completedAction.count} ファイルを元のフォルダに追加しました
            </div>
          </div>
        </div>
      )}

      {/* 警告 */}
      {totalNeeded > 0 && (
        <div className="mt-4 bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-amber-300">
            <strong>注意:</strong> 拡張ファイルは元のフォルダに直接追加されます。
            重要なデータは事前にバックアップしてください。
          </div>
        </div>
      )}
    </div>
  );
}
