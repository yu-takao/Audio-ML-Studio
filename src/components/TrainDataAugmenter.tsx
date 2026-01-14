/**
 * 訓練データ拡張コンポーネント（統合版）
 * Train分割のみを対象にデータ拡張とクラスバランシングを統合的に行う
 */

import { useState, useCallback, useMemo } from 'react';
import {
  Sparkles,
  Scale,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  BarChart3,
  Info,
} from 'lucide-react';
import { AugmentationSettingsPanel } from './AugmentationSettings';
import { useNoiseFiles } from '../hooks/useNoiseFiles';
import type { AugmentationSettings, NoiseSample } from '../utils/audioAugmentation';
import {
  defaultSettings,
  audioBufferToMono,
  generateAugmentations,
  samplesToWavBlob,
} from '../utils/audioAugmentation';

export interface FileInfo {
  file: File;
  path: string;
  folderName: string;
}

interface TrainDataAugmenterProps {
  trainFiles: FileInfo[];
  getLabel: (file: FileInfo) => string;
  onAugmentationComplete: (augmentedFiles: FileInfo[]) => void;
}

interface AugmentationProgress {
  isProcessing: boolean;
  currentClass: string;
  currentFile: string;
  processedClasses: number;
  totalClasses: number;
  processedFiles: number;
  totalFiles: number;
  generatedFiles: number;
  status: string;
}

export function TrainDataAugmenter({
  trainFiles,
  getLabel,
  onAugmentationComplete,
}: TrainDataAugmenterProps) {
  // 統合された拡張設定
  const [settings, setSettings] = useState<AugmentationSettings>({
    ...defaultSettings,
    timeShift: { ...defaultSettings.timeShift, enabled: true, variations: 2 },
    gainVariation: { ...defaultSettings.gainVariation, enabled: true, variations: 2 },
    environmentNoise: { ...defaultSettings.environmentNoise, enabled: false },
    pitchShift: { ...defaultSettings.pitchShift, enabled: false },
    timeStretch: { ...defaultSettings.timeStretch, enabled: false },
  });

  // クラスバランシング設定
  const [enableBalancing, setEnableBalancing] = useState(true);
  const [balanceMode, setBalanceMode] = useState<'max' | 'median' | 'custom'>('max');
  const [customTarget, setCustomTarget] = useState(100);

  // 追加拡張設定
  const [enableExtraAugmentation, setEnableExtraAugmentation] = useState(true);
  const [applyExtraToBalanced, setApplyExtraToBalanced] = useState(false);

  // UI状態
  const [isExpanded, setIsExpanded] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [progress, setProgress] = useState<AugmentationProgress>({
    isProcessing: false,
    currentClass: '',
    currentFile: '',
    processedClasses: 0,
    totalClasses: 0,
    processedFiles: 0,
    totalFiles: 0,
    generatedFiles: 0,
    status: '',
  });
  const [isComplete, setIsComplete] = useState(false);

  // ノイズファイル
  const {
    filteredEntries,
    isReady: isNoiseReady,
    loadRandomSamples,
  } = useNoiseFiles();

  // クラス分布を計算
  const classDistribution = useMemo(() => {
    const dist = new Map<string, number>();
    for (const file of trainFiles) {
      const label = getLabel(file);
      dist.set(label, (dist.get(label) || 0) + 1);
    }
    return dist;
  }, [trainFiles, getLabel]);

  // ファイルをクラスごとにグループ化
  const filesByClass = useMemo(() => {
    const groups = new Map<string, FileInfo[]>();
    for (const file of trainFiles) {
      const label = getLabel(file);
      const group = groups.get(label) || [];
      group.push(file);
      groups.set(label, group);
    }
    return groups;
  }, [trainFiles, getLabel]);

  const classes = Array.from(classDistribution.keys()).sort();
  const counts = Array.from(classDistribution.values());
  const maxCount = counts.length > 0 ? Math.max(...counts) : 0;
  const minCount = counts.length > 0 ? Math.min(...counts) : 0;
  const medianCount = counts.length > 0
    ? [...counts].sort((a, b) => a - b)[Math.floor(counts.length / 2)]
    : 0;
  const imbalanceRatio = minCount > 0 ? maxCount / minCount : Infinity;
  const isImbalanced = imbalanceRatio > 1.5;

  // ターゲット数を計算
  const getTargetCount = () => {
    switch (balanceMode) {
      case 'max': return maxCount;
      case 'median': return medianCount;
      case 'custom': return customTarget;
      default: return maxCount;
    }
  };

  const targetCount = getTargetCount();

  // 拡張計画を計算
  const augmentationPlan = useMemo(() => {
    return classes.map(cls => {
      const currentCount = classDistribution.get(cls) || 0;
      const files = filesByClass.get(cls) || [];

      // バランス調整で必要な数
      const balanceNeeded = enableBalancing
        ? Math.max(0, targetCount - currentCount)
        : 0;

      // 追加拡張で生成される数
      let extraCount = 0;
      if (enableExtraAugmentation) {
        // 追加拡張の対象ファイル数を計算
        const baseFilesToAugment = applyExtraToBalanced
          ? currentCount + balanceNeeded  // バランス後の全ファイル
          : currentCount;                  // 元のファイルのみ

        // 各拡張タイプのバリエーション数を合計
        let variationsPerFile = 0;
        if (settings.timeShift.enabled) variationsPerFile += settings.timeShift.variations;
        if (settings.gainVariation.enabled) variationsPerFile += settings.gainVariation.variations;
        if (settings.environmentNoise.enabled && isNoiseReady) variationsPerFile += settings.environmentNoise.variations;
        if (settings.pitchShift.enabled) variationsPerFile += settings.pitchShift.variations;
        if (settings.timeStretch.enabled) variationsPerFile += settings.timeStretch.variations;

        extraCount = baseFilesToAugment * variationsPerFile;
      }

      return {
        className: cls,
        currentCount,
        balanceNeeded,
        extraCount,
        totalAfter: currentCount + balanceNeeded + extraCount,
        files,
      };
    });
  }, [
    classes,
    classDistribution,
    filesByClass,
    enableBalancing,
    targetCount,
    enableExtraAugmentation,
    applyExtraToBalanced,
    settings,
    isNoiseReady,
  ]);

  const totalBalanceNeeded = augmentationPlan.reduce((sum, p) => sum + p.balanceNeeded, 0);
  const totalExtraCount = augmentationPlan.reduce((sum, p) => sum + p.extraCount, 0);
  const totalGenerated = totalBalanceNeeded + totalExtraCount;
  const finalTotal = trainFiles.length + totalGenerated;

  // WAVファイルをAudioBufferに変換
  const fileToAudioBuffer = async (file: File): Promise<AudioBuffer> => {
    const arrayBuffer = await file.arrayBuffer();
    const audioContext = new AudioContext();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    await audioContext.close();
    return audioBuffer;
  };

  /**
   * 統合された拡張処理
   * クラスごとに、バランス調整と追加拡張を一度に実行
   */
  const executeAugmentation = useCallback(async () => {
    setProgress({
      isProcessing: true,
      currentClass: '',
      currentFile: '',
      processedClasses: 0,
      totalClasses: classes.length,
      processedFiles: 0,
      totalFiles: trainFiles.length,
      generatedFiles: 0,
      status: 'データ拡張を開始しています...',
    });
    setIsComplete(false);

    const resultFiles: FileInfo[] = [...trainFiles]; // 元データも含む
    let totalGenerated = 0;

    // ノイズサンプルを読み込む
    let noiseSamples: NoiseSample[] | undefined;
    if (settings.environmentNoise.enabled && isNoiseReady) {
      noiseSamples = await loadRandomSamples(20);
    }

    // クラスごとに処理
    for (let classIdx = 0; classIdx < augmentationPlan.length; classIdx++) {
      const plan = augmentationPlan[classIdx];
      const { className, balanceNeeded, files } = plan;

      if (files.length === 0) continue;

      setProgress(prev => ({
        ...prev,
        currentClass: className,
        processedClasses: classIdx,
        status: `クラス "${className}" を処理中...`,
      }));

      // フェーズ1: バランス調整（不足分のみ生成）
      const balancedFiles: FileInfo[] = [];
      if (enableBalancing && balanceNeeded > 0) {
        let classGenerated = 0;
        let fileIndex = 0;

        while (classGenerated < balanceNeeded && files.length > 0) {
          const fileInfo = files[fileIndex % files.length];

          setProgress(prev => ({
            ...prev,
            currentFile: fileInfo.file.name,
            processedFiles: prev.processedFiles + 1,
          }));

          try {
            const audioBuffer = await fileToAudioBuffer(fileInfo.file);
            const samples = audioBufferToMono(audioBuffer);
            const sampleRate = audioBuffer.sampleRate;
            const baseName = fileInfo.file.name.replace(/\.wav$/i, '');

            // バランス調整用の拡張を生成
            const augmentations = generateAugmentations(
              samples,
              sampleRate,
              settings,
              `${baseName}_bal${classGenerated}`,
              noiseSamples
            ).filter(a => !a.name.includes('_original')); // オリジナルを除外

            for (const aug of augmentations) {
              if (classGenerated >= balanceNeeded) break;

              const blob = samplesToWavBlob(aug.samples, sampleRate);
              const augFile = new File([blob], aug.name, { type: 'audio/wav' });

              const newFileInfo: FileInfo = {
                file: augFile,
                path: `${fileInfo.path.replace(/[^/]+$/, '')}${aug.name}`,
                folderName: fileInfo.folderName,
              };

              balancedFiles.push(newFileInfo);
              resultFiles.push(newFileInfo);

              classGenerated++;
              totalGenerated++;

              setProgress(prev => ({
                ...prev,
                generatedFiles: totalGenerated,
              }));
            }
          } catch (err) {
            console.warn(`Failed to augment ${fileInfo.file.name}:`, err);
          }

          fileIndex++;

          // 無限ループ防止
          if (fileIndex > files.length * 20 && classGenerated < balanceNeeded) {
            console.warn(`Could not generate enough augmentations for class "${className}"`);
            break;
          }
        }
      }

      // フェーズ2: 追加拡張（元ファイルまたはバランス後の全ファイル）
      if (enableExtraAugmentation) {
        const filesToAugment = applyExtraToBalanced
          ? [...files, ...balancedFiles]  // バランス後の全ファイル
          : files;                         // 元のファイルのみ

        for (const fileInfo of filesToAugment) {
          setProgress(prev => ({
            ...prev,
            currentFile: fileInfo.file.name,
            processedFiles: prev.processedFiles + 1,
          }));

          try {
            const audioBuffer = await fileToAudioBuffer(fileInfo.file);
            const samples = audioBufferToMono(audioBuffer);
            const sampleRate = audioBuffer.sampleRate;
            const baseName = fileInfo.file.name.replace(/\.wav$/i, '');

            const augmentations = generateAugmentations(
              samples,
              sampleRate,
              settings,
              baseName,
              noiseSamples
            ).filter(a => !a.name.includes('_original'));

            for (const aug of augmentations) {
              const blob = samplesToWavBlob(aug.samples, sampleRate);
              const augFile = new File([blob], aug.name, { type: 'audio/wav' });

              resultFiles.push({
                file: augFile,
                path: `${fileInfo.path.replace(/[^/]+$/, '')}${aug.name}`,
                folderName: fileInfo.folderName,
              });

              totalGenerated++;
              setProgress(prev => ({ ...prev, generatedFiles: totalGenerated }));
            }
          } catch (err) {
            console.warn(`Failed to extra-augment ${fileInfo.file.name}:`, err);
          }
        }
      }

      setProgress(prev => ({
        ...prev,
        processedClasses: classIdx + 1,
      }));
    }

    setProgress(prev => ({
      ...prev,
      isProcessing: false,
      status: `データ拡張が完了しました（${totalGenerated} ファイルを生成）`,
    }));

    setIsComplete(true);
    onAugmentationComplete(resultFiles);
  }, [
    trainFiles,
    classes,
    augmentationPlan,
    settings,
    isNoiseReady,
    loadRandomSamples,
    enableBalancing,
    enableExtraAugmentation,
    applyExtraToBalanced,
    onAugmentationComplete,
  ]);

  return (
    <div className="bg-zinc-800/50 rounded-xl border border-zinc-700 overflow-hidden">
      {/* ヘッダー */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-4 flex items-center justify-between hover:bg-zinc-700/30 transition-all"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-violet-500/20">
            <Sparkles className="w-5 h-5 text-violet-400" />
          </div>
          <div className="text-left">
            <h3 className="font-semibold text-white">訓練データの拡張</h3>
            <p className="text-sm text-zinc-400">
              クラス分布の調整とデータ拡張を統合的に実行
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {isComplete && (
            <div className="flex items-center gap-1 text-emerald-400 text-sm">
              <CheckCircle2 className="w-4 h-4" />
              完了
            </div>
          )}
          {isExpanded ? (
            <ChevronUp className="w-5 h-5 text-zinc-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-zinc-400" />
          )}
        </div>
      </button>

      {isExpanded && (
        <div className="p-4 pt-0 space-y-4">
          {/* クラス分布の警告 */}
          {isImbalanced && (
            <div className="flex items-start gap-3 p-3 rounded-lg bg-amber-500/10 border border-amber-500/30">
              <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm text-amber-300 font-medium">
                  クラス不均衡が検出されました（比率: {imbalanceRatio.toFixed(1)}x）
                </p>
                <p className="text-xs text-amber-400/80 mt-1">
                  最大 {maxCount} ファイル vs 最小 {minCount} ファイル
                </p>
              </div>
            </div>
          )}

          {/* 現在のクラス分布 */}
          <div className="p-4 rounded-lg bg-zinc-900/50 space-y-3">
            <div className="flex items-center gap-2 text-sm text-zinc-400">
              <BarChart3 className="w-4 h-4" />
              現在のクラス分布
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 max-h-48 overflow-y-auto">
              {augmentationPlan.map((plan) => {
                const percentage = maxCount > 0 ? ((plan.currentCount / maxCount) * 100).toFixed(0) : '0';
                const isMin = plan.currentCount === minCount;
                const isMax = plan.currentCount === maxCount;
                return (
                  <div
                    key={plan.className}
                    className={`
                      bg-zinc-800/50 rounded-lg p-2 border
                      ${isMin ? 'border-red-500/50' : isMax ? 'border-emerald-500/50' : 'border-zinc-700'}
                    `}
                  >
                    <div className="text-xs text-zinc-400 truncate" title={plan.className}>
                      {plan.className}
                    </div>
                    <div className="flex items-baseline gap-1 mt-1">
                      <span className={`text-lg font-bold ${isMin ? 'text-red-400' : isMax ? 'text-emerald-400' : 'text-white'}`}>
                        {plan.currentCount}
                      </span>
                      {plan.totalAfter > plan.currentCount && (
                        <span className="text-xs text-violet-400">
                          → {plan.totalAfter}
                        </span>
                      )}
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

          {/* クラスバランシング設定 */}
          <div className="p-4 rounded-lg bg-zinc-900/50 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Scale className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-medium text-white">クラス分布を揃える</span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableBalancing}
                  onChange={(e) => setEnableBalancing(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-zinc-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-500"></div>
              </label>
            </div>

            {enableBalancing && (
              <div className="space-y-3 pt-2">
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { mode: 'max' as const, label: '最大に揃える', value: maxCount },
                    { mode: 'median' as const, label: '中央値', value: medianCount },
                    { mode: 'custom' as const, label: 'カスタム', value: customTarget },
                  ].map((option) => (
                    <button
                      key={option.mode}
                      onClick={() => setBalanceMode(option.mode)}
                      className={`p-2 rounded-lg border transition-all text-center ${balanceMode === option.mode
                          ? 'border-emerald-500 bg-emerald-500/10 text-emerald-400'
                          : 'border-zinc-700 hover:border-zinc-600 text-zinc-400'
                        }`}
                    >
                      <div className="text-xs">{option.label}</div>
                      <div className="text-lg font-bold">{option.value}</div>
                    </button>
                  ))}
                </div>

                {balanceMode === 'custom' && (
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-zinc-400">目標数:</label>
                    <input
                      type="number"
                      value={customTarget}
                      onChange={(e) => setCustomTarget(Math.max(1, parseInt(e.target.value) || 1))}
                      min={1}
                      className="w-24 px-3 py-1 bg-zinc-800 border border-zinc-700 rounded-lg text-white text-sm"
                    />
                  </div>
                )}
              </div>
            )}
          </div>

          {/* 追加拡張設定 */}
          <div className="p-4 rounded-lg bg-zinc-900/50 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-sky-400" />
                <span className="text-sm font-medium text-white">追加のデータ拡張</span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableExtraAugmentation}
                  onChange={(e) => setEnableExtraAugmentation(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-zinc-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-sky-500"></div>
              </label>
            </div>

            {enableExtraAugmentation && (
              <div className="space-y-2">
                <div className="flex items-start gap-2 p-2 rounded bg-zinc-800/50">
                  <input
                    type="checkbox"
                    id="applyExtraToBalanced"
                    checked={applyExtraToBalanced}
                    onChange={(e) => setApplyExtraToBalanced(e.target.checked)}
                    className="mt-0.5 h-4 w-4 accent-sky-500"
                  />
                  <label htmlFor="applyExtraToBalanced" className="text-sm text-zinc-300 cursor-pointer">
                    バランス調整後のファイルにも適用
                    <span className="block text-xs text-zinc-500 mt-0.5">
                      オフ: 元のファイルのみ拡張 / オン: バランス後の全ファイルを拡張
                    </span>
                  </label>
                </div>
              </div>
            )}
          </div>

          {/* 拡張設定の詳細 */}
          <details className="group" open={showAdvanced}>
            <summary
              onClick={(e) => {
                e.preventDefault();
                setShowAdvanced(!showAdvanced);
              }}
              className="flex items-center gap-2 cursor-pointer text-sm text-zinc-400 hover:text-zinc-300 p-2 rounded hover:bg-zinc-800/50"
            >
              <Sparkles className="w-4 h-4" />
              拡張の詳細設定
              {showAdvanced ? <ChevronUp className="w-4 h-4 ml-auto" /> : <ChevronDown className="w-4 h-4 ml-auto" />}
            </summary>
            {showAdvanced && (
              <div className="mt-3">
                <AugmentationSettingsPanel
                  settings={settings}
                  onChange={setSettings}
                  noiseFileCount={filteredEntries.length}
                  isNoiseReady={isNoiseReady}
                />
              </div>
            )}
          </details>

          {/* 拡張計画のサマリー */}
          <div className="p-4 rounded-lg bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10 border border-violet-500/30">
            <div className="flex items-center gap-2 mb-3">
              <Info className="w-5 h-5 text-violet-400" />
              <span className="text-white font-medium">拡張計画</span>
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span className="text-zinc-400">元のファイル:</span>
                <span className="text-white ml-2 font-medium">{trainFiles.length}</span>
              </div>
              {enableBalancing && (
                <div>
                  <span className="text-zinc-400">バランス調整:</span>
                  <span className="text-emerald-400 ml-2 font-medium">+{totalBalanceNeeded}</span>
                </div>
              )}
              {enableExtraAugmentation && (
                <div>
                  <span className="text-zinc-400">追加拡張:</span>
                  <span className="text-sky-400 ml-2 font-medium">+{totalExtraCount}</span>
                </div>
              )}
              <div className="col-span-2 pt-2 border-t border-zinc-700">
                <span className="text-zinc-400">最終的な合計:</span>
                <span className="text-violet-300 ml-2 font-bold text-lg">{finalTotal} ファイル</span>
              </div>
            </div>
          </div>

          {/* 進捗表示 */}
          {progress.isProcessing && (
            <div className="p-4 rounded-lg bg-zinc-900/50 space-y-2">
              <div className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 text-violet-400 animate-spin" />
                <span className="text-sm text-white">{progress.status}</span>
              </div>
              <div className="text-xs text-zinc-500">
                クラス: {progress.currentClass} ({progress.processedClasses}/{progress.totalClasses})
              </div>
              {progress.currentFile && (
                <div className="text-xs text-zinc-500 truncate">
                  処理中: {progress.currentFile}
                </div>
              )}
              <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all"
                  style={{
                    width: `${progress.totalClasses > 0
                        ? (progress.processedClasses / progress.totalClasses) * 100
                        : 0
                      }%`,
                  }}
                />
              </div>
              <div className="text-xs text-zinc-400">
                生成済み: {progress.generatedFiles} / {totalGenerated} ファイル
              </div>
            </div>
          )}

          {/* 完了メッセージ */}
          {isComplete && !progress.isProcessing && (
            <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <span className="text-sm text-emerald-300">{progress.status}</span>
            </div>
          )}

          {/* 実行ボタン */}
          {!isComplete && (
            <button
              onClick={executeAugmentation}
              disabled={progress.isProcessing || trainFiles.length === 0}
              className={`w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${progress.isProcessing || trainFiles.length === 0
                  ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-violet-500 to-fuchsia-500 hover:from-violet-600 hover:to-fuchsia-600 text-white shadow-lg'
                }`}
            >
              {progress.isProcessing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  処理中...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  データ拡張を実行
                </>
              )}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
