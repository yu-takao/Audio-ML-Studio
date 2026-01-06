import { useState, useCallback } from 'react';
import { useFileSystem } from '../hooks/useFileSystem';
import { useNoiseFiles } from '../hooks/useNoiseFiles';
import { AugmentationSettingsPanel } from './AugmentationSettings';
import { ESC50FilterPanel } from './ESC50FilterPanel';
import type { AugmentationSettings } from '../utils/audioAugmentation';
import {
  defaultSettings,
  audioBufferToMono,
  generateAugmentations,
  samplesToWavBlob,
} from '../utils/audioAugmentation';
import {
  FolderOpen,
  FolderOutput,
  FileAudio,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Sparkles,
} from 'lucide-react';

interface ProcessingStatus {
  isProcessing: boolean;
  currentFile: string;
  currentFileIndex: number;
  totalFiles: number;
  processedFiles: number;
  generatedFiles: number;
  errors: string[];
}

export function DataAugmentation() {
  const {
    inputFolder,
    outputFolder,
    wavFiles,
    isLoading,
    error,
    selectInputFolder,
    selectOutputFolder,
  } = useFileSystem();

  const {
    allEntries,
    filteredEntries,
    filterOptions,
    setFilterOptions,
    isLoading: isLoadingNoise,
    isReady: isNoiseReady,
    loadRandomSamples,
  } = useNoiseFiles();

  const [settings, setSettings] = useState<AugmentationSettings>({
    ...defaultSettings,
    environmentNoise: {
      ...defaultSettings.environmentNoise,
      enabled: true,
    },
  });
  const [status, setStatus] = useState<ProcessingStatus>({
    isProcessing: false,
    currentFile: '',
    currentFileIndex: 0,
    totalFiles: 0,
    processedFiles: 0,
    generatedFiles: 0,
    errors: [],
  });

  const fileToAudioBuffer = async (file: File): Promise<AudioBuffer> => {
    const arrayBuffer = await file.arrayBuffer();
    const audioContext = new AudioContext();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    await audioContext.close();
    return audioBuffer;
  };

  const processFiles = useCallback(async () => {
    if (!outputFolder || wavFiles.length === 0) return;

    setStatus({
      isProcessing: true,
      currentFile: '',
      currentFileIndex: 0,
      totalFiles: wavFiles.length,
      processedFiles: 0,
      generatedFiles: 0,
      errors: [],
    });

    const errors: string[] = [];
    let totalGenerated = 0;

    let noiseSamples = undefined;
    if (settings.environmentNoise.enabled && isNoiseReady) {
      noiseSamples = await loadRandomSamples(20);
    }

    for (let i = 0; i < wavFiles.length; i++) {
      const wavFile = wavFiles[i];
      
      setStatus((prev) => ({
        ...prev,
        currentFile: wavFile.name,
        currentFileIndex: i,
      }));

      try {
        const audioBuffer = await fileToAudioBuffer(wavFile.file);
        const samples = audioBufferToMono(audioBuffer);
        
        const augmentedResults = generateAugmentations(
          samples,
          audioBuffer.sampleRate,
          settings,
          wavFile.name.split('/').pop() || wavFile.name,
          noiseSamples && noiseSamples.length > 0 ? noiseSamples : undefined
        );

        const pathParts = wavFile.name.split('/');
        let currentDir = outputFolder;

        for (let j = 0; j < pathParts.length - 1; j++) {
          currentDir = await currentDir.getDirectoryHandle(pathParts[j], { create: true });
        }

        for (const result of augmentedResults) {
          const blob = samplesToWavBlob(result.samples, audioBuffer.sampleRate);
          const fileHandle = await currentDir.getFileHandle(result.name, { create: true });
          const writable = await fileHandle.createWritable();
          await writable.write(blob);
          await writable.close();
          totalGenerated++;
        }

        setStatus((prev) => ({
          ...prev,
          processedFiles: i + 1,
          generatedFiles: totalGenerated,
        }));
      } catch (err) {
        errors.push(`${wavFile.name}: ${(err as Error).message}`);
      }
    }

    setStatus((prev) => ({
      ...prev,
      isProcessing: false,
      errors,
    }));
  }, [outputFolder, wavFiles, settings, isNoiseReady, loadRandomSamples]);

  const calculateTotalOutputFiles = () => {
    let perFile = 1;
    if (settings.timeShift.enabled) perFile += settings.timeShift.variations;
    if (settings.gainVariation.enabled) perFile += settings.gainVariation.variations;
    if (settings.environmentNoise.enabled && isNoiseReady) {
      perFile += settings.environmentNoise.variations;
    }
    if (settings.pitchShift.enabled) perFile += settings.pitchShift.variations;
    if (settings.timeStretch.enabled) perFile += settings.timeStretch.variations;
    return wavFiles.length * perFile;
  };

  return (
    <div className="space-y-8">
      {/* フォルダ選択セクション */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 入力フォルダ */}
        <div
          className={`
            rounded-xl border-2 border-dashed p-6 transition-all cursor-pointer
            ${inputFolder
              ? 'border-emerald-500/50 bg-emerald-500/5'
              : 'border-zinc-700 hover:border-violet-500/50 hover:bg-violet-500/5'
            }
          `}
          onClick={selectInputFolder}
        >
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-xl ${inputFolder ? 'bg-emerald-500/20' : 'bg-zinc-800'}`}>
              <FolderOpen className={`w-8 h-8 ${inputFolder ? 'text-emerald-400' : 'text-zinc-400'}`} />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-white">入力フォルダ</h3>
              {inputFolder ? (
                <div className="text-sm text-emerald-400 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4" />
                  {inputFolder.name}
                  <span className="text-zinc-500">({wavFiles.length} WAVファイル)</span>
                </div>
              ) : (
                <p className="text-sm text-zinc-500">クリックしてフォルダを選択</p>
              )}
            </div>
            {isLoading && <Loader2 className="w-5 h-5 text-violet-400 animate-spin" />}
          </div>
        </div>

        {/* 出力フォルダ */}
        <div
          className={`
            rounded-xl border-2 border-dashed p-6 transition-all cursor-pointer
            ${outputFolder
              ? 'border-emerald-500/50 bg-emerald-500/5'
              : 'border-zinc-700 hover:border-violet-500/50 hover:bg-violet-500/5'
            }
          `}
          onClick={selectOutputFolder}
        >
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-xl ${outputFolder ? 'bg-emerald-500/20' : 'bg-zinc-800'}`}>
              <FolderOutput className={`w-8 h-8 ${outputFolder ? 'text-emerald-400' : 'text-zinc-400'}`} />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-white">出力フォルダ</h3>
              {outputFolder ? (
                <div className="text-sm text-emerald-400 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4" />
                  {outputFolder.name}
                </div>
              ) : (
                <p className="text-sm text-zinc-500">クリックしてフォルダを選択</p>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* エラー表示 */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/50 rounded-xl p-4 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {/* ファイルリスト */}
      {wavFiles.length > 0 && (
        <section className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-4">
          <div className="flex items-center gap-2 mb-3">
            <FileAudio className="w-5 h-5 text-violet-400" />
            <h3 className="font-semibold text-white">検出されたWAVファイル</h3>
            <span className="text-sm text-zinc-500">({wavFiles.length}件)</span>
          </div>
          <div className="max-h-48 overflow-y-auto space-y-1">
            {wavFiles.slice(0, 50).map((file, index) => (
              <div
                key={index}
                className="text-sm text-zinc-400 px-3 py-1.5 rounded bg-zinc-800/50 truncate"
              >
                {file.name}
              </div>
            ))}
            {wavFiles.length > 50 && (
              <div className="text-sm text-zinc-500 px-3 py-1.5">
                ...他 {wavFiles.length - 50} ファイル
              </div>
            )}
          </div>
        </section>
      )}

      {/* ESC-50 フィルタパネル */}
      {settings.environmentNoise.enabled && (
        <ESC50FilterPanel
          filterOptions={filterOptions}
          onChange={setFilterOptions}
          totalCount={allEntries.length}
          filteredCount={filteredEntries.length}
        />
      )}

      {/* 設定パネル */}
      <AugmentationSettingsPanel 
        settings={settings} 
        onChange={setSettings}
        noiseFileCount={filteredEntries.length}
        isNoiseReady={isNoiseReady}
      />

      {/* ノイズ読み込み状態 */}
      <div className="flex items-center justify-center gap-2 text-sm">
        {isLoadingNoise ? (
          <span className="text-zinc-500 flex items-center gap-1">
            <Loader2 className="w-4 h-4 animate-spin" />
            ESC-50 読込中...
          </span>
        ) : isNoiseReady ? (
          <span className="text-emerald-400 flex items-center gap-1">
            <CheckCircle2 className="w-4 h-4" />
            ESC-50: {filteredEntries.length}/{allEntries.length} 使用可能
          </span>
        ) : (
          <span className="text-amber-400">ノイズなし</span>
        )}
      </div>

      {/* 実行ボタン */}
      <section className="flex flex-col items-center gap-4">
        <button
          onClick={processFiles}
          disabled={!inputFolder || !outputFolder || status.isProcessing || wavFiles.length === 0}
          className={`
            flex items-center gap-3 px-8 py-4 rounded-xl font-bold text-lg transition-all
            ${
              !inputFolder || !outputFolder || status.isProcessing || wavFiles.length === 0
                ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white hover:from-violet-600 hover:to-fuchsia-600 shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40'
            }
          `}
        >
          {status.isProcessing ? (
            <>
              <Loader2 className="w-6 h-6 animate-spin" />
              処理中...
            </>
          ) : (
            <>
              <Sparkles className="w-6 h-6" />
              データ拡張を実行
            </>
          )}
        </button>
        
        {wavFiles.length > 0 && !status.isProcessing && (
          <p className="text-sm text-zinc-500">
            {wavFiles.length} ファイル → 約 {calculateTotalOutputFiles()} ファイル生成予定
          </p>
        )}
      </section>

      {/* 進捗表示 */}
      {status.isProcessing && (
        <section className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
          <div className="space-y-4">
            <div className="flex justify-between text-sm">
              <span className="text-zinc-400">処理中: {status.currentFile}</span>
              <span className="text-white">
                {status.processedFiles} / {status.totalFiles}
              </span>
            </div>
            <div className="h-3 bg-zinc-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-300"
                style={{
                  width: `${(status.processedFiles / status.totalFiles) * 100}%`,
                }}
              />
            </div>
            <div className="flex justify-between text-sm text-zinc-500">
              <span>生成済み: {status.generatedFiles} ファイル</span>
              <span>{Math.round((status.processedFiles / status.totalFiles) * 100)}%</span>
            </div>
          </div>
        </section>
      )}

      {/* 完了メッセージ */}
      {!status.isProcessing && status.processedFiles > 0 && (
        <section className="bg-emerald-500/10 border border-emerald-500/50 rounded-xl p-6">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="w-8 h-8 text-emerald-400" />
            <div>
              <h3 className="font-bold text-white">処理完了!</h3>
              <p className="text-emerald-400">
                {status.processedFiles} ファイルから {status.generatedFiles} ファイルを生成しました
              </p>
            </div>
          </div>
          {status.errors.length > 0 && (
            <div className="mt-4 p-4 bg-red-500/10 rounded-lg">
              <p className="text-red-400 font-semibold mb-2">エラー ({status.errors.length}件):</p>
              <ul className="text-sm text-red-400 space-y-1">
                {status.errors.map((err, i) => (
                  <li key={i}>{err}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

