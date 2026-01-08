/**
 * 訓練データ拡張コンポーネント
 * Train分割のみを対象にデータ拡張とクラスバランシングを行う
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
  PlusCircle,
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
  currentFile: string;
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
  // バランス調整（不足分の生成）用 拡張設定
  const [balanceSettings, setBalanceSettings] = useState<AugmentationSettings>({
    ...defaultSettings,
    environmentNoise: {
      ...defaultSettings.environmentNoise,
      enabled: false, // デフォルトはオフ
    },
  });

  // 追加拡張（Train全体へ一律）用 拡張設定
  const [extraSettings, setExtraSettings] = useState<AugmentationSettings>({
    ...defaultSettings,
    // 追加拡張は軽めがデフォルト
    timeShift: { ...defaultSettings.timeShift, enabled: true, variations: 1 },
    gainVariation: { ...defaultSettings.gainVariation, enabled: true, variations: 1 },
    environmentNoise: { ...defaultSettings.environmentNoise, enabled: false },
    pitchShift: { ...defaultSettings.pitchShift, enabled: false },
    timeStretch: { ...defaultSettings.timeStretch, enabled: false },
  });
  const [enableExtraAugmentation, setEnableExtraAugmentation] = useState(true);
  // 追加拡張の対象：既に拡張されたファイル（_aug/_timeshift等）を含めるか
  const [includeAlreadyAugmented, setIncludeAlreadyAugmented] = useState(false);
  
  // クラスバランシング設定
  const [enableBalancing, setEnableBalancing] = useState(true);
  const [balanceMode, setBalanceMode] = useState<'max' | 'median' | 'custom'>('max');
  const [customTarget, setCustomTarget] = useState(100);
  
  // UI状態
  const [isExpanded, setIsExpanded] = useState(true);
  const [progress, setProgress] = useState<AugmentationProgress>({
    isProcessing: false,
    currentFile: '',
    processedFiles: 0,
    totalFiles: 0,
    generatedFiles: 0,
    status: '',
  });
  const [phase, setPhase] = useState<'idle' | 'balanced' | 'extra_done'>('idle');
  const [isComplete, setIsComplete] = useState(false);
  const [balancedTrainFiles, setBalancedTrainFiles] = useState<FileInfo[] | null>(null);
  
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
  const maxCount = Math.max(...counts);
  const minCount = Math.min(...counts);
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
  
  // 必要な拡張数を計算
  const augmentationPlan = useMemo(() => {
    if (!enableBalancing) {
      // バランシング無効の場合は拡張のみ
      return classes.map(cls => ({
        className: cls,
        currentCount: classDistribution.get(cls) || 0,
        neededAugmentations: 0,
        files: filesByClass.get(cls) || [],
      }));
    }
    
    return classes.map(cls => {
      const currentCount = classDistribution.get(cls) || 0;
      const needed = Math.max(0, targetCount - currentCount);
      return {
        className: cls,
        currentCount,
        neededAugmentations: needed,
        files: filesByClass.get(cls) || [],
      };
    });
  }, [classes, classDistribution, filesByClass, enableBalancing, targetCount]);
  
  const totalNeeded = augmentationPlan.reduce((sum, p) => sum + p.neededAugmentations, 0);
  
  // WAVファイルをAudioBufferに変換
  const fileToAudioBuffer = async (file: File): Promise<AudioBuffer> => {
    const arrayBuffer = await file.arrayBuffer();
    const audioContext = new AudioContext();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    await audioContext.close();
    return audioBuffer;
  };
  
  // ランダム値生成
  // @ts-expect-error - Reserved for future use
  const randomInRange = (min: number, max: number) => min + Math.random() * (max - min);
  
  /**
   * バランス調整（不足分だけ生成）
   */
  const executeBalanceAugmentation = useCallback(async () => {
    setProgress({
      isProcessing: true,
      currentFile: '',
      processedFiles: 0,
      totalFiles: trainFiles.length,
      generatedFiles: 0,
      status: 'バランス調整（不足分の生成）を開始しています...',
    });
    setIsComplete(false);
    setPhase('idle');
    
    const augmentedFiles: FileInfo[] = [...trainFiles]; // 元データも含む
    let totalGenerated = 0;
    
    // ノイズサンプルを読み込む
    let noiseSamples: NoiseSample[] | undefined;
    if (balanceSettings.environmentNoise.enabled && isNoiseReady) {
      noiseSamples = await loadRandomSamples(20);
    }
    
    // クラスごとに処理
    for (const plan of augmentationPlan) {
      const { className, neededAugmentations, files } = plan;
      
      // バランス調整をしない場合はこのフェーズでは何もしない
      if (!enableBalancing || neededAugmentations <= 0) continue;
      
      if (files.length === 0) continue;
      
      setProgress(prev => ({
        ...prev,
        status: `クラス "${className}" を処理中（不足分を生成）...`,
      }));
      
      let classGenerated = 0;
      let fileIndex = 0;
      
      while (classGenerated < neededAugmentations) {
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
          
          // 拡張を生成
          const augmentations = generateAugmentations(
            samples,
            sampleRate,
            balanceSettings,
            `${baseName}_aug${classGenerated}`,
            noiseSamples
          );
          
          // オリジナルを除外（既にtrainFilesに含まれている）
          const newAugmentations = augmentations.filter(a => !a.name.includes('_original'));
          
          for (const aug of newAugmentations) {
            if (classGenerated >= neededAugmentations) break;
            
            const blob = samplesToWavBlob(aug.samples, sampleRate);
            const augFile = new File([blob], aug.name, { type: 'audio/wav' });
            
            augmentedFiles.push({
              file: augFile,
              path: `${fileInfo.path.replace(/[^/]+$/, '')}${aug.name}`,
              folderName: fileInfo.folderName,
            });
            
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
        if (fileIndex > files.length * 10 && classGenerated < neededAugmentations) {
          console.warn(`Could not generate enough augmentations for class "${className}"`);
          break;
        }
      }
    }
    
    setProgress(prev => ({
      ...prev,
      isProcessing: false,
      status: enableBalancing
        ? `バランス調整が完了しました（${totalGenerated} ファイルを生成）`
        : 'バランス調整はスキップしました',
    }));
    setBalancedTrainFiles(augmentedFiles);
    setPhase('balanced');
  }, [
    trainFiles,
    augmentationPlan,
    balanceSettings,
    isNoiseReady,
    loadRandomSamples,
    enableBalancing,
  ]);

  /**
   * 追加拡張（Train全体へ一律に適用）
   * - データリーク防止のため Train のみに実施
   */
  const executeExtraAugmentation = useCallback(async () => {
    const baseFiles = balancedTrainFiles || trainFiles;
    if (!enableExtraAugmentation) return;

    setProgress({
      isProcessing: true,
      currentFile: '',
      processedFiles: 0,
      totalFiles: baseFiles.length,
      generatedFiles: 0,
      status: '追加のデータ拡張を開始しています...',
    });
    setIsComplete(false);

    const resultFiles: FileInfo[] = [...baseFiles];
    let totalGenerated = 0;

    let noiseSamples: NoiseSample[] | undefined;
    if (extraSettings.environmentNoise.enabled && isNoiseReady) {
      noiseSamples = await loadRandomSamples(20);
    }

    const shouldSkipAsAlreadyAugmented = (name: string) => {
      if (includeAlreadyAugmented) return false;
      return /_(aug|timeshift|gain|envnoise|pitch|stretch)_/i.test(name) || /_aug\d+/i.test(name);
    };

    for (let i = 0; i < baseFiles.length; i++) {
      const fileInfo = baseFiles[i];

      setProgress(prev => ({
        ...prev,
        currentFile: fileInfo.file.name,
        processedFiles: i + 1,
      }));

      if (shouldSkipAsAlreadyAugmented(fileInfo.file.name)) {
        continue;
      }

      try {
        const audioBuffer = await fileToAudioBuffer(fileInfo.file);
        const samples = audioBufferToMono(audioBuffer);
        const sampleRate = audioBuffer.sampleRate;
        const baseName = fileInfo.file.name.replace(/\.wav$/i, '');

        const augmentations = generateAugmentations(
          samples,
          sampleRate,
          extraSettings,
          `${baseName}_extra`,
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

    setProgress(prev => ({
      ...prev,
      isProcessing: false,
      status: `追加拡張が完了しました（${totalGenerated} ファイルを生成）`,
    }));

    setPhase('extra_done');
    setIsComplete(true);
    onAugmentationComplete(resultFiles);
  }, [
    balancedTrainFiles,
    trainFiles,
    enableExtraAugmentation,
    extraSettings,
    isNoiseReady,
    loadRandomSamples,
    includeAlreadyAugmented,
    onAugmentationComplete,
  ]);

  const finalizeWithoutExtra = useCallback(() => {
    const baseFiles = balancedTrainFiles || trainFiles;
    setPhase('extra_done');
    setIsComplete(true);
    onAugmentationComplete(baseFiles);
  }, [balancedTrainFiles, trainFiles, onAugmentationComplete]);
  
  // 予想される拡張後のファイル数を計算
  const estimatedTotal = trainFiles.length + totalNeeded;
  
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
              Train分割のみを対象にデータ拡張・クラス分布調整
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
                  クラス不均衡が検出されました（比率: {imbalanceRatio.toFixed(1)}）
                </p>
                <p className="text-xs text-amber-400/80 mt-1">
                  クラス分布を揃えることで、モデルの偏りを防ぎます。
                </p>
              </div>
            </div>
          )}
          
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
                    { mode: 'median' as const, label: '中央値に揃える', value: medianCount },
                    { mode: 'custom' as const, label: 'カスタム', value: customTarget },
                  ].map((option) => (
                    <button
                      key={option.mode}
                      onClick={() => setBalanceMode(option.mode)}
                      className={`p-2 rounded-lg border transition-all text-center ${
                        balanceMode === option.mode
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
                
                <div className="text-sm text-zinc-400">
                  <span className="text-violet-400 font-medium">{totalNeeded}</span> ファイルを生成
                  → 合計 <span className="text-white font-medium">{estimatedTotal}</span> ファイル
                </div>
              </div>
            )}
          </div>
          
          {/* 拡張設定 */}
          <details className="group">
            <summary className="flex items-center gap-2 cursor-pointer text-sm text-zinc-400 hover:text-zinc-300 p-2">
              <Sparkles className="w-4 h-4" />
              拡張の詳細設定
            </summary>
            <div className="mt-3">
              <AugmentationSettingsPanel
                settings={balanceSettings}
                onChange={setBalanceSettings}
                noiseFileCount={filteredEntries.length}
                isNoiseReady={isNoiseReady}
              />
            </div>
          </details>

          {/* 追加拡張ステップ */}
          {phase === 'balanced' && (
            <div className="p-4 rounded-lg bg-zinc-900/50 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <PlusCircle className="w-4 h-4 text-sky-400" />
                  <span className="text-sm font-medium text-white">追加のデータ拡張（Train全体）</span>
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
                <>
                  <div className="flex items-center justify-between text-sm">
                    <label className="text-zinc-400">拡張済みファイルも対象にする</label>
                    <input
                      type="checkbox"
                      checked={includeAlreadyAugmented}
                      onChange={(e) => setIncludeAlreadyAugmented(e.target.checked)}
                      className="h-4 w-4 accent-sky-500"
                    />
                  </div>

                  <details className="group">
                    <summary className="flex items-center gap-2 cursor-pointer text-sm text-zinc-400 hover:text-zinc-300 p-2">
                      <Sparkles className="w-4 h-4" />
                      追加拡張の詳細設定
                    </summary>
                    <div className="mt-3">
                      <AugmentationSettingsPanel
                        settings={extraSettings}
                        onChange={setExtraSettings}
                        noiseFileCount={filteredEntries.length}
                        isNoiseReady={isNoiseReady}
                      />
                    </div>
                  </details>
                </>
              )}
            </div>
          )}
          
          {/* 進捗表示 */}
          {progress.isProcessing && (
            <div className="p-4 rounded-lg bg-zinc-900/50 space-y-2">
              <div className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 text-violet-400 animate-spin" />
                <span className="text-sm text-white">{progress.status}</span>
              </div>
              <div className="text-xs text-zinc-500">
                {progress.currentFile && `処理中: ${progress.currentFile}`}
              </div>
              <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-violet-500 transition-all"
                  style={{
                    width: `${
                      ((progress.processedFiles / Math.max(1, progress.totalFiles)) * 100)
                    }%`,
                  }}
                />
              </div>
              <div className="text-xs text-zinc-400">
                生成済み: {progress.generatedFiles} ファイル
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
          
          {/* 実行ボタン（2段階） */}
          {phase === 'idle' && (
            <button
              onClick={executeBalanceAugmentation}
              disabled={progress.isProcessing || trainFiles.length === 0}
              className={`w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${
                progress.isProcessing || trainFiles.length === 0
                  ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                  : 'bg-violet-500 hover:bg-violet-600 text-white'
              }`}
            >
              {progress.isProcessing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  処理中...
                </>
              ) : (
                <>
                  <Scale className="w-5 h-5" />
                  バランス調整を実行
                </>
              )}
            </button>
          )}

          {phase === 'balanced' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <button
                onClick={() => {
                  if (enableExtraAugmentation) void executeExtraAugmentation();
                  else finalizeWithoutExtra();
                }}
                disabled={progress.isProcessing}
                className={`w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${
                  progress.isProcessing
                    ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                    : 'bg-sky-500 hover:bg-sky-600 text-white'
                }`}
              >
                {progress.isProcessing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    処理中...
                  </>
                ) : enableExtraAugmentation ? (
                  <>
                    <Sparkles className="w-5 h-5" />
                    追加拡張を実行して完了
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-5 h-5" />
                    追加拡張なしで完了
                  </>
                )}
              </button>

              <button
                onClick={() => {
                  // バランス調整からやり直す
                  setBalancedTrainFiles(null);
                  setPhase('idle');
                  setIsComplete(false);
                }}
                disabled={progress.isProcessing}
                className={`w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${
                  progress.isProcessing
                    ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                    : 'bg-zinc-700 hover:bg-zinc-600 text-white'
                }`}
              >
                再設定（バランス調整から）
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

