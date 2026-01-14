import { useState, useCallback, useEffect, useMemo } from 'react';
import { list, downloadData, uploadData } from 'aws-amplify/storage';
import {
  FolderOpen,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Brain,
  BarChart3,
  Settings,
  Cloud,
  Database,
  Upload,
  RefreshCw,
  ChevronRight,
  Calendar,
  HardDrive,
  Plus,
  Archive,
  Sparkles,
  Scissors,
} from 'lucide-react';
import { analyzeFilenames, generateClassLabel, type ParsedMetadata, type TargetFieldConfig, type AuxiliaryFieldConfig } from '../utils/metadataParser';
import { MetadataConfig } from './MetadataConfig';
import { ParameterHelp, PARAM_HELP } from './ParameterHelp';
import { SmartRecommendation, type DatasetStats } from './SmartRecommendation';
import { CloudTraining } from './CloudTraining';
import { ModelBrowser } from './ModelBrowser';
import { DataSplitPreview } from './DataSplitPreview';
import { TrainDataAugmenter } from './TrainDataAugmenter';
import { stratifiedSplit, groupFilesByClass, calculateSplitStats, type SplitResult, type SplitStats } from '../utils/dataSplitter';

interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
  testSplit: number;
}

interface DatasetInfo {
  totalFiles: number;
  classes: string[];
  samplesPerClass: Map<string, number>;
}

interface FileInfo {
  file: File;
  path: string;
  folderName: string;
}

// S3のデータセット情報
interface S3Dataset {
  path: string;
  name: string;
  fileCount: number;
  lastModified: Date;
  size: number;
  hasMetadata?: boolean;
}

// S3に保存するメタデータ
interface S3DatasetMetadata {
  metadata: ParsedMetadata;
  targetConfig: TargetFieldConfig | null;
  auxiliaryFields: AuxiliaryFieldConfig[];
  datasetInfo: DatasetInfo;
}

// ステップ定義（新フロー：データ選択 → 設定 → 分割＆拡張 → 訓練）
type Step = 'select-source' | 'configure' | 'augment' | 'training';

// メインタブ定義
type MainTab = 'new-training' | 'saved-models';

/**
 * データセットの統計情報から推奨パラメータを計算
 */
function calculateRecommendedParams(stats: DatasetStats): TrainingConfig {
  const { totalSamples, numClasses, minSamplesPerClass, imbalanceRatio } = stats;

  // エポック数の推奨
  let epochs: number;
  if (totalSamples < 500) {
    epochs = 30;
  } else if (totalSamples < 1000) {
    epochs = 50;
  } else if (totalSamples < 5000) {
    epochs = 80;
  } else {
    epochs = 100;
  }

  // バッチサイズの推奨
  let batchSize: number;
  if (totalSamples < 500) {
    batchSize = 16;
  } else if (totalSamples < 2000) {
    batchSize = 32;
  } else {
    batchSize = 64;
  }

  // クラス不均衡がある場合はバッチサイズを小さめに
  if (imbalanceRatio > 3) {
    batchSize = Math.max(16, batchSize - 16);
  }

  // 学習率の推奨
  let learningRate: number;
  if (numClasses <= 3) {
    learningRate = 0.001;
  } else if (numClasses <= 10) {
    learningRate = 0.001;
  } else {
    learningRate = 0.0005;
  }

  // 検証データ割合の推奨
  let validationSplit: number;
  if (totalSamples < 500) {
    validationSplit = 0.15;
  } else if (minSamplesPerClass < 50) {
    validationSplit = 0.15;
  } else {
    validationSplit = 0.2;
  }

  // テストデータ割合の推奨
  let testSplit: number;
  if (totalSamples < 500) {
    testSplit = 0.1;
  } else if (minSamplesPerClass < 30) {
    testSplit = 0.1;
  } else {
    testSplit = 0.15;
  }

  return { epochs, batchSize, learningRate, validationSplit, testSplit };
}

interface ModelTrainingProps {
  userId: string;
}

export function ModelTraining({ userId }: ModelTrainingProps) {
  // UI状態の永続化キー
  const UI_STATE_KEY = 'audio-ml-ui-state';

  // メインタブ管理
  const [mainTab, setMainTab] = useState<MainTab>('new-training');

  // ステップ管理
  const [currentStep, setCurrentStep] = useState<Step>('select-source');

  // データソース選択
  const [dataSource, setDataSource] = useState<'s3' | 'local' | null>(null);

  // S3データセット関連
  const [s3Datasets, setS3Datasets] = useState<S3Dataset[]>([]);
  const [selectedS3Dataset, setSelectedS3Dataset] = useState<S3Dataset | null>(null);
  const [isLoadingS3, setIsLoadingS3] = useState(false);

  // ローカルファイル関連
  const [dataFolder, setDataFolder] = useState<FileSystemDirectoryHandle | null>(null);
  const [isLoadingData, setIsLoadingData] = useState(false);

  // 共通
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  // メタデータ関連の状態
  const [metadata, setMetadata] = useState<ParsedMetadata | null>(null);
  const [targetConfig, setTargetConfig] = useState<TargetFieldConfig | null>(null);
  const [auxiliaryFields, setAuxiliaryFields] = useState<AuxiliaryFieldConfig[]>([]);
  const [problemType, setProblemType] = useState<'classification' | 'regression'>('classification');
  const [tolerance, setTolerance] = useState<number>(0);
  const [fileInfoList, setFileInfoList] = useState<FileInfo[]>([]);

  // 訓練設定
  const [config, setConfig] = useState<TrainingConfig>({
    epochs: 50,
    batchSize: 32,
    learningRate: 0.001,
    validationSplit: 0.2,
    testSplit: 0.15,
  });

  // 訓練ジョブ状態
  const [isTrainingStarted, setIsTrainingStarted] = useState(false);

  // データ分割状態
  const [dataSplit, setDataSplit] = useState<SplitResult | null>(null);
  const [splitStats, setSplitStats] = useState<SplitStats | null>(null);

  // 拡張後の訓練データ
  const [augmentedTrainFiles, setAugmentedTrainFiles] = useState<FileInfo[] | null>(null);
  const [isAugmentationComplete, setIsAugmentationComplete] = useState(false);

  /**
   * UI状態を保存
   */
  useEffect(() => {
    try {
      localStorage.setItem(
        UI_STATE_KEY,
        JSON.stringify({
          mainTab,
          currentStep,
        })
      );
    } catch (err) {
      console.error('Failed to save UI state', err);
    }
  }, [mainTab, currentStep]);

  /**
   * UI状態を復元
   */
  useEffect(() => {
    try {
      const saved = localStorage.getItem(UI_STATE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved) as { mainTab?: MainTab; currentStep?: Step };
        if (parsed.mainTab === 'new-training' || parsed.mainTab === 'saved-models') {
          setMainTab(parsed.mainTab);
        }
        if (parsed.currentStep === 'select-source' || parsed.currentStep === 'configure' || parsed.currentStep === 'augment' || parsed.currentStep === 'training') {
          setCurrentStep(parsed.currentStep);
        }
      }
    } catch (err) {
      console.error('Failed to restore UI state', err);
    }
  }, []);

  /**
   * S3からデータセット一覧を取得
   */
  const loadS3Datasets = useCallback(async () => {
    setIsLoadingS3(true);
    setError(null);

    try {
      // public/training-data/ 以下のフォルダを取得
      const result = await list({
        path: 'public/training-data/',
        options: {
          listAll: true,
        },
      });

      // フォルダごとにグループ化
      const datasetMap = new Map<string, { files: number; size: number; lastModified: Date }>();

      for (const item of result.items) {
        // パスからデータセット名を抽出 (public/training-data/userId/timestamp/...)
        const parts = item.path.split('/');
        if (parts.length >= 4) {
          const datasetPath = parts.slice(0, 4).join('/');
          const existing = datasetMap.get(datasetPath) || { files: 0, size: 0, lastModified: new Date(0) };
          existing.files++;
          existing.size += item.size || 0;
          if (item.lastModified && item.lastModified > existing.lastModified) {
            existing.lastModified = item.lastModified;
          }
          datasetMap.set(datasetPath, existing);
        }
      }

      // データセット一覧に変換
      const datasets: S3Dataset[] = [];
      datasetMap.forEach((info, path) => {
        const parts = path.split('/');
        const timestamp = parts[3];
        datasets.push({
          path,
          name: `データセット ${new Date(parseInt(timestamp)).toLocaleString('ja-JP')}`,
          fileCount: info.files,
          lastModified: info.lastModified,
          size: info.size,
        });
      });

      // 日付順にソート
      datasets.sort((a, b) => b.lastModified.getTime() - a.lastModified.getTime());

      setS3Datasets(datasets);
    } catch (err) {
      console.error('Failed to load S3 datasets:', err);
      setError('S3からデータセット一覧を取得できませんでした');
    } finally {
      setIsLoadingS3(false);
    }
  }, []);

  /**
   * S3データセットを選択してメタデータを読み込む
   */
  const selectS3Dataset = useCallback(async (dataset: S3Dataset) => {
    setSelectedS3Dataset(dataset);
    setIsLoadingData(true);
    setError(null);

    try {
      // メタデータファイルを読み込む
      const metadataPath = `${dataset.path}/metadata.json`;
      const downloadResult = await downloadData({
        path: metadataPath,
      }).result;

      const text = await downloadResult.body.text();
      const savedMetadata: S3DatasetMetadata = JSON.parse(text);

      // samplesPerClassをMapに変換（JSONではオブジェクトとして保存される）
      let samplesPerClass: Map<string, number>;
      if (savedMetadata.datasetInfo.samplesPerClass instanceof Map) {
        samplesPerClass = savedMetadata.datasetInfo.samplesPerClass;
      } else {
        // オブジェクトからMapに変換
        samplesPerClass = new Map(Object.entries(savedMetadata.datasetInfo.samplesPerClass as unknown as Record<string, number>));
      }

      // 状態を復元
      setMetadata(savedMetadata.metadata);
      setTargetConfig(savedMetadata.targetConfig);
      setAuxiliaryFields(savedMetadata.auxiliaryFields);
      setDatasetInfo({
        ...savedMetadata.datasetInfo,
        samplesPerClass,
      });

      console.log('S3 metadata loaded:', savedMetadata);
    } catch (err) {
      console.warn('Metadata not found for dataset, will need to configure manually:', err);
      // メタデータがない場合は手動設定が必要
      setMetadata(null);
      setTargetConfig(null);
      setAuxiliaryFields([]);
      setDatasetInfo(null);
      setError('このデータセットにはメタデータがありません。新規アップロードを行ってください。');
    } finally {
      setIsLoadingData(false);
    }
  }, []);

  /**
   * メタデータをS3に保存
   */
  const saveMetadataToS3 = useCallback(async (dataPath: string) => {
    if (!metadata || !datasetInfo) return;

    // MapをオブジェクトにJSON変換（Mapはそのままではシリアライズできない）
    const datasetInfoForSave = {
      ...datasetInfo,
      samplesPerClass: Object.fromEntries(datasetInfo.samplesPerClass),
    };

    const metadataToSave = {
      metadata,
      targetConfig,
      auxiliaryFields,
      datasetInfo: datasetInfoForSave,
    };

    try {
      await uploadData({
        path: `${dataPath}/metadata.json`,
        data: JSON.stringify(metadataToSave, null, 2),
        options: {
          contentType: 'application/json',
        },
      }).result;
      console.log('Metadata saved to S3');
    } catch (err) {
      console.error('Failed to save metadata:', err);
    }
  }, [metadata, targetConfig, auxiliaryFields, datasetInfo]);

  /**
   * ローカルフォルダをスキャンしてデータを読み込む
   */
  const scanFolderAndLoad = useCallback(async (
    handle: FileSystemDirectoryHandle,
  ) => {
    setIsLoadingData(true);
    setError(null);

    const filenames: string[] = [];
    const folderNames: string[] = [];
    const fileInfos: FileInfo[] = [];

    // 再帰的にファイルを探索
    async function scanDirectory(
      dir: FileSystemDirectoryHandle,
      path: string,
      topLevelFolder: string
    ) {
      for await (const entry of dir.values()) {
        if (entry.kind === 'directory') {
          const subDir = await dir.getDirectoryHandle(entry.name);
          const newPath = path ? `${path}/${entry.name}` : entry.name;
          const folder = topLevelFolder || entry.name;
          await scanDirectory(subDir, newPath, folder);
        } else if (entry.kind === 'file' && entry.name.toLowerCase().endsWith('.wav')) {
          filenames.push(entry.name);
          folderNames.push(topLevelFolder);

          const file = await (entry as FileSystemFileHandle).getFile();
          fileInfos.push({
            file,
            path: path ? `${path}/${entry.name}` : entry.name,
            folderName: topLevelFolder,
          });
        }
      }
    }

    await scanDirectory(handle, '', '');

    // メタデータを解析
    const parsedMetadata = analyzeFilenames(filenames, folderNames);
    setMetadata(parsedMetadata);
    setFileInfoList(fileInfos);

    // デフォルトのターゲット設定
    const numericField = parsedMetadata.fields.find(f => f.isNumeric);
    if (numericField) {
      setTargetConfig({
        fieldIndex: numericField.index,
        fieldName: numericField.label,
        useAsTarget: true,
        groupingMode: 'individual',
      });
    }

    setIsLoadingData(false);
  }, []);

  /**
   * ローカルフォルダを選択
   */
  const selectDataFolder = useCallback(async () => {
    try {
      // クラス分布の揃え（増幅）でファイルを書き込むため readwrite を要求
      const handle = await window.showDirectoryPicker({ mode: 'readwrite' });
      setDataFolder(handle);
      await scanFolderAndLoad(handle);
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setError('フォルダの読み込みに失敗しました: ' + (err as Error).message);
      }
      setIsLoadingData(false);
    }
  }, [scanFolderAndLoad]);

  /**
   * ターゲット設定が変更されたらデータセット情報を更新（ローカルデータの場合のみ）
   */
  useEffect(() => {
    // S3データセットの場合は、selectS3Datasetで設定済みなのでスキップ
    if (dataSource === 's3') {
      return;
    }

    // ローカルデータの場合
    if (!metadata || !targetConfig || fileInfoList.length === 0) {
      setDatasetInfo(null);
      return;
    }

    // クラス分布を計算
    const samplesPerClass = new Map<string, number>();

    fileInfoList.forEach((info) => {
      const label = generateClassLabel(
        info.file.name,
        info.folderName,
        targetConfig,
        metadata.separator
      );
      samplesPerClass.set(label, (samplesPerClass.get(label) || 0) + 1);
    });

    const classes = [...samplesPerClass.keys()].sort();

    setDatasetInfo({
      totalFiles: fileInfoList.length,
      classes,
      samplesPerClass,
    });
  }, [metadata, targetConfig, fileInfoList, dataSource]);

  /**
   * データセット統計を計算
   */
  const datasetStats = useMemo<DatasetStats | null>(() => {
    if (!datasetInfo) return null;

    // samplesPerClassからカウントを取得（MapまたはObjectに対応）
    let counts: number[];
    if (datasetInfo.samplesPerClass instanceof Map) {
      if (datasetInfo.samplesPerClass.size === 0) return null;
      counts = Array.from(datasetInfo.samplesPerClass.values());
    } else {
      // オブジェクトの場合
      const values = Object.values(datasetInfo.samplesPerClass as unknown as Record<string, number>);
      if (values.length === 0) return null;
      counts = values;
    }

    const minSamples = Math.min(...counts);
    const maxSamples = Math.max(...counts);

    return {
      totalSamples: datasetInfo.totalFiles,
      numClasses: datasetInfo.classes.length,
      minSamplesPerClass: minSamples,
      maxSamplesPerClass: maxSamples,
      imbalanceRatio: minSamples > 0 ? maxSamples / minSamples : Infinity,
    };
  }, [datasetInfo]);

  /**
   * データセット統計が変更されたら推奨パラメータを自動適用
   */
  useEffect(() => {
    if (datasetStats) {
      const recommended = calculateRecommendedParams(datasetStats);
      setConfig(recommended);
    }
  }, [datasetStats]);

  /**
   * S3データセット用の統計情報
   */
  const s3DatasetStats = useMemo<DatasetStats | null>(() => {
    if (!selectedS3Dataset) return null;

    return {
      totalSamples: selectedS3Dataset.fileCount,
      numClasses: 2, // デフォルト値（クラス数は不明）
      minSamplesPerClass: Math.floor(selectedS3Dataset.fileCount / 2),
      maxSamplesPerClass: Math.ceil(selectedS3Dataset.fileCount / 2),
      imbalanceRatio: 1,
    };
  }, [selectedS3Dataset]);

  /**
   * S3データセットが選択されたら、ファイル数に基づいて推奨パラメータを適用
   */
  useEffect(() => {
    if (s3DatasetStats) {
      const recommended = calculateRecommendedParams(s3DatasetStats);
      setConfig(recommended);
    }
  }, [s3DatasetStats]);

  // 有効な統計情報（ローカルまたはS3）
  const effectiveStats = datasetStats || s3DatasetStats;

  /**
   * フィールドラベルを変更
   */
  const handleFieldLabelChange = useCallback((index: number, label: string) => {
    if (!metadata) return;

    const updatedFields = metadata.fields.map((field) =>
      field.index === index ? { ...field, label } : field
    );

    const updatedFolderField = metadata.folderField && index === -1
      ? { ...metadata.folderField, label }
      : metadata.folderField;

    setMetadata({
      ...metadata,
      fields: updatedFields,
      folderField: updatedFolderField,
    });

    if (targetConfig && targetConfig.fieldIndex === index) {
      setTargetConfig({
        ...targetConfig,
        fieldName: label,
      });
    }
  }, [metadata, targetConfig]);

  // バイトを人間が読める形式に変換
  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  // データソースが選択されているか
  const hasDataSource = dataSource === 's3' ? selectedS3Dataset !== null : fileInfoList.length > 0;

  // 設定が完了しているか（S3の場合もメタデータが必要）
  const isConfigComplete = hasDataSource && datasetInfo && targetConfig;

  // ラベル生成関数（分割・拡張で使用）
  const getLabelForFile = useCallback((file: FileInfo): string => {
    if (!targetConfig || !metadata) return 'unknown';
    return generateClassLabel(
      file.file.name,
      file.folderName,
      targetConfig,
      metadata.separator
    );
  }, [targetConfig, metadata]);

  // データ分割を実行
  const executeDataSplit = useCallback(() => {
    if (fileInfoList.length === 0 || !targetConfig || !metadata) return;

    const filesByClass = groupFilesByClass(fileInfoList, getLabelForFile);
    const split = stratifiedSplit(filesByClass, {
      validationSplit: config.validationSplit,
      testSplit: config.testSplit,
      seed: 42, // 再現性のため固定シード
    });

    setDataSplit(split);
    setSplitStats(calculateSplitStats(split, getLabelForFile));
    setAugmentedTrainFiles(null); // 分割が変わったら拡張結果をリセット
    setIsAugmentationComplete(false);
  }, [fileInfoList, targetConfig, metadata, config.validationSplit, config.testSplit, getLabelForFile]);

  // 拡張完了時のハンドラ
  const handleAugmentationComplete = useCallback((augmentedFiles: FileInfo[]) => {
    setAugmentedTrainFiles(augmentedFiles);
    setIsAugmentationComplete(true);
  }, []);

  // 最終的な訓練用ファイルリスト（拡張済みTrain + 元Val + 元Test）
  const finalFileInfoList = useMemo(() => {
    if (!dataSplit) return fileInfoList;

    const trainFiles = augmentedTrainFiles || dataSplit.train;
    // 訓練時はTrain/Val/Testを区別するためにパスにプレフィックスを付ける
    const withPrefix = (files: FileInfo[], prefix: string) =>
      files.map(f => ({
        ...f,
        path: `${prefix}/${f.path}`,
      }));

    return [
      ...withPrefix(trainFiles, 'train'),
      ...withPrefix(dataSplit.validation, 'validation'),
      ...withPrefix(dataSplit.test, 'test'),
    ];
  }, [dataSplit, augmentedTrainFiles, fileInfoList]);

  // ステップのレンダリング
  const renderStepIndicator = () => {
    const steps = [
      { id: 'select-source', label: 'データ選択', icon: Database },
      { id: 'configure', label: '設定', icon: Settings },
      { id: 'augment', label: '分割＆拡張', icon: Sparkles },
      { id: 'training', label: '訓練', icon: Brain },
    ];

    const stepOrder = steps.map(s => s.id);
    const currentIndex = stepOrder.indexOf(currentStep);

    return (
      <div className="flex items-center justify-center gap-2 mb-8">
        {steps.map((step, index) => {
          const isActive = currentStep === step.id;
          const isPast = index < currentIndex;
          const Icon = step.icon;

          return (
            <div key={step.id} className="flex items-center">
              {index > 0 && (
                <ChevronRight className={`w-5 h-5 mx-2 ${isPast ? 'text-emerald-500' : 'text-zinc-600'}`} />
              )}
              <button
                onClick={() => {
                  if (isPast) setCurrentStep(step.id as Step);
                }}
                disabled={!isPast && !isActive}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${isActive
                    ? 'bg-sky-500 text-white'
                    : isPast
                      ? 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 cursor-pointer'
                      : 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
                  }`}
              >
                <Icon className="w-4 h-4" />
                <span className="text-sm font-medium hidden md:inline">{step.label}</span>
              </button>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* メインタブ */}
      <div className="flex gap-2 p-1 bg-zinc-800/50 rounded-xl border border-zinc-700/50">
        <button
          onClick={() => setMainTab('new-training')}
          className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium transition-all ${mainTab === 'new-training'
              ? 'bg-violet-500 text-white'
              : 'text-zinc-400 hover:text-white hover:bg-zinc-700/50'
            }`}
        >
          <Plus className="w-5 h-5" />
          新規訓練
        </button>
        <button
          onClick={() => setMainTab('saved-models')}
          className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium transition-all ${mainTab === 'saved-models'
              ? 'bg-violet-500 text-white'
              : 'text-zinc-400 hover:text-white hover:bg-zinc-700/50'
            }`}
        >
          <Archive className="w-5 h-5" />
          保存済みモデル
        </button>
      </div>

      {/* 保存済みモデルタブ */}
      {mainTab === 'saved-models' && (
        <ModelBrowser userId={userId} />
      )}

      {/* 新規訓練タブ */}
      {mainTab === 'new-training' && (
        <>
          {/* ステップインジケーター */}
          {renderStepIndicator()}

          {/* エラー表示 */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/50 rounded-xl p-4 flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
              <p className="text-red-400">{error}</p>
            </div>
          )}

          {/* Step 1: データソース選択 */}
          {currentStep === 'select-source' && (
            <div className="space-y-6">
              <section className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 rounded-lg bg-sky-500/20">
                    <Database className="w-5 h-5 text-sky-400" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">データセットを選択</h2>
                    <p className="text-sm text-zinc-400">S3から既存のデータを選ぶか、新しくアップロードします</p>
                  </div>
                </div>

                {/* 選択オプション */}
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <button
                    onClick={() => {
                      setDataSource('s3');
                      loadS3Datasets();
                    }}
                    className={`p-6 rounded-xl border-2 transition-all text-left ${dataSource === 's3'
                        ? 'border-sky-500 bg-sky-500/10'
                        : 'border-zinc-700 hover:border-zinc-600 bg-zinc-900/50'
                      }`}
                  >
                    <Cloud className={`w-8 h-8 mb-3 ${dataSource === 's3' ? 'text-sky-400' : 'text-zinc-500'}`} />
                    <h3 className={`font-semibold mb-1 ${dataSource === 's3' ? 'text-white' : 'text-zinc-300'}`}>
                      S3から選択
                    </h3>
                    <p className="text-sm text-zinc-500">
                      以前アップロードしたデータセットを使用
                    </p>
                  </button>

                  <button
                    onClick={() => setDataSource('local')}
                    className={`p-6 rounded-xl border-2 transition-all text-left ${dataSource === 'local'
                        ? 'border-emerald-500 bg-emerald-500/10'
                        : 'border-zinc-700 hover:border-zinc-600 bg-zinc-900/50'
                      }`}
                  >
                    <Upload className={`w-8 h-8 mb-3 ${dataSource === 'local' ? 'text-emerald-400' : 'text-zinc-500'}`} />
                    <h3 className={`font-semibold mb-1 ${dataSource === 'local' ? 'text-white' : 'text-zinc-300'}`}>
                      新規アップロード
                    </h3>
                    <p className="text-sm text-zinc-500">
                      ローカルからデータをアップロード
                    </p>
                  </button>
                </div>

                {/* S3データセット一覧 */}
                {dataSource === 's3' && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-white font-medium">既存のデータセット</h3>
                      <button
                        onClick={loadS3Datasets}
                        disabled={isLoadingS3}
                        className="flex items-center gap-2 px-3 py-1.5 text-sm bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg transition-all"
                      >
                        <RefreshCw className={`w-4 h-4 ${isLoadingS3 ? 'animate-spin' : ''}`} />
                        更新
                      </button>
                    </div>

                    {isLoadingS3 ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="w-6 h-6 text-sky-400 animate-spin" />
                      </div>
                    ) : s3Datasets.length === 0 ? (
                      <div className="text-center py-8 text-zinc-500">
                        <Database className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p>アップロード済みのデータセットがありません</p>
                        <button
                          onClick={() => setDataSource('local')}
                          className="mt-3 text-sky-400 hover:text-sky-300 text-sm"
                        >
                          新規アップロードに切り替える
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-2 max-h-64 overflow-y-auto">
                        {s3Datasets.map((dataset) => (
                          <button
                            key={dataset.path}
                            onClick={() => selectS3Dataset(dataset)}
                            disabled={isLoadingData}
                            className={`w-full p-4 rounded-lg border transition-all text-left ${selectedS3Dataset?.path === dataset.path
                                ? 'border-sky-500 bg-sky-500/10'
                                : 'border-zinc-700 hover:border-zinc-600 bg-zinc-900/50'
                              } ${isLoadingData ? 'opacity-50 cursor-not-allowed' : ''}`}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-3">
                                {isLoadingData && selectedS3Dataset?.path === dataset.path ? (
                                  <Loader2 className="w-5 h-5 text-sky-400 animate-spin" />
                                ) : selectedS3Dataset?.path === dataset.path ? (
                                  <CheckCircle2 className="w-5 h-5 text-sky-400" />
                                ) : (
                                  <FolderOpen className="w-5 h-5 text-zinc-500" />
                                )}
                                <div>
                                  <div className="font-medium text-white">{dataset.name}</div>
                                  <div className="flex items-center gap-4 text-xs text-zinc-500 mt-1">
                                    <span className="flex items-center gap-1">
                                      <HardDrive className="w-3 h-3" />
                                      {dataset.fileCount} ファイル
                                    </span>
                                    <span>{formatBytes(dataset.size)}</span>
                                    <span className="flex items-center gap-1">
                                      <Calendar className="w-3 h-3" />
                                      {dataset.lastModified.toLocaleDateString('ja-JP')}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* ローカルフォルダ選択 */}
                {dataSource === 'local' && (
                  <div className="space-y-4">
                    <div
                      className={`
            rounded-xl border-2 border-dashed p-6 transition-all cursor-pointer
            ${dataFolder
                          ? 'border-emerald-500/50 bg-emerald-500/5'
                          : 'border-zinc-700 hover:border-emerald-500/50 hover:bg-emerald-500/5'
                        }
          `}
                      onClick={selectDataFolder}
                    >
                      <div className="flex items-center gap-4">
                        <div className={`p-3 rounded-xl ${dataFolder ? 'bg-emerald-500/20' : 'bg-zinc-800'}`}>
                          <FolderOpen className={`w-8 h-8 ${dataFolder ? 'text-emerald-400' : 'text-zinc-400'}`} />
                        </div>
                        <div className="flex-1">
                          <h3 className="font-semibold text-white">データフォルダを選択</h3>
                          {dataFolder ? (
                            <div className="text-sm text-emerald-400 flex items-center gap-2">
                              <CheckCircle2 className="w-4 h-4" />
                              {dataFolder.name}
                              {metadata && (
                                <span className="text-zinc-500">
                                  ({metadata.sampleCount} ファイル, {metadata.fields.length} フィールド)
                                </span>
                              )}
                            </div>
                          ) : (
                            <p className="text-sm text-zinc-500">クリックしてフォルダを選択</p>
                          )}
                        </div>
                        {isLoadingData && <Loader2 className="w-5 h-5 text-emerald-400 animate-spin" />}
                      </div>
                    </div>
                  </div>
                )}
              </section>

              {/* 次へボタン */}
              <div className="flex justify-end">
                <button
                  onClick={() => setCurrentStep('configure')}
                  disabled={!hasDataSource}
                  className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${hasDataSource
                      ? 'bg-sky-500 hover:bg-sky-600 text-white'
                      : 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                    }`}
                >
                  次へ：設定
                  <ChevronRight className="w-5 h-5" />
                </button>
              </div>
            </div>
          )}

          {/* Step 2: 設定 */}
          {currentStep === 'configure' && (
            <div className="space-y-6">
              {/* メタデータ設定（ローカルアップロードの場合） */}
              {dataSource === 'local' && metadata && (
                <MetadataConfig
                  metadata={metadata}
                  onFieldLabelChange={handleFieldLabelChange}
                  targetConfig={targetConfig}
                  onTargetConfigChange={setTargetConfig}
                  auxiliaryFields={auxiliaryFields}
                  onAuxiliaryFieldsChange={setAuxiliaryFields}
                  problemType={problemType}
                  onProblemTypeChange={setProblemType}
                  tolerance={tolerance}
                  onToleranceChange={setTolerance}
                />
              )}

              {/* データセット情報（ローカルの場合） */}
              {dataSource === 'local' && datasetInfo && (
                <section className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 rounded-lg bg-emerald-500/20">
                      <BarChart3 className="w-5 h-5 text-emerald-400" />
                    </div>
                    <h2 className="text-lg font-semibold text-white">クラス分布</h2>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div className="bg-zinc-900/50 rounded-lg p-4">
                      <div className="text-sm text-zinc-500">総サンプル数</div>
                      <div className="text-2xl font-bold text-white">{datasetInfo.totalFiles}</div>
                    </div>
                    <div className="bg-zinc-900/50 rounded-lg p-4">
                      <div className="text-sm text-zinc-500">クラス数</div>
                      <div className="text-2xl font-bold text-white">{datasetInfo.classes.length}</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                    {datasetInfo.classes.map((cls) => {
                      const count = datasetInfo.samplesPerClass.get(cls) || 0;
                      const percentage = ((count / datasetInfo.totalFiles) * 100).toFixed(1);
                      return (
                        <div key={cls} className="bg-zinc-900/50 rounded-lg p-3">
                          <div className="text-sm text-white font-medium truncate" title={cls}>
                            {cls}
                          </div>
                          <div className="flex items-baseline gap-2">
                            <span className="text-lg font-bold text-violet-400">{count}</span>
                            <span className="text-xs text-zinc-500">({percentage}%)</span>
                          </div>
                          <div className="mt-1 h-1 bg-zinc-700 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-violet-500"
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </section>
              )}

              {/* データ分割設定 */}
              <section className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-2 rounded-lg bg-amber-500/20">
                    <Scissors className="w-5 h-5 text-amber-400" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">データ分割設定</h2>
                    <p className="text-sm text-zinc-400">
                      Train/Validation/Testに分割します。Trainデータのみ拡張対象になります。
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="flex items-center gap-1 mb-2">
                      <label className="text-sm text-zinc-400">検証データ: {(config.validationSplit * 100).toFixed(0)}%</label>
                      <ParameterHelp {...PARAM_HELP.validationSplit} />
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="0.3"
                      step="0.05"
                      value={config.validationSplit}
                      onChange={(e) => setConfig({ ...config, validationSplit: parseFloat(e.target.value) })}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <div className="flex items-center gap-1 mb-2">
                      <label className="text-sm text-zinc-400">テストデータ: {(config.testSplit * 100).toFixed(0)}%</label>
                      <ParameterHelp {...PARAM_HELP.testSplit} />
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="0.3"
                      step="0.05"
                      value={config.testSplit}
                      onChange={(e) => setConfig({ ...config, testSplit: parseFloat(e.target.value) })}
                      className="w-full"
                    />
                  </div>
                </div>

                <div className="mt-4 p-3 rounded-lg bg-zinc-900/50 text-sm text-zinc-400">
                  <p>
                    <span className="text-emerald-400 font-medium">Train:</span>{' '}
                    {((1 - config.validationSplit - config.testSplit) * 100).toFixed(0)}%（拡張対象）
                    {' | '}
                    <span className="text-amber-400 font-medium">Validation:</span>{' '}
                    {(config.validationSplit * 100).toFixed(0)}%（元データのみ）
                    {' | '}
                    <span className="text-sky-400 font-medium">Test:</span>{' '}
                    {(config.testSplit * 100).toFixed(0)}%（完全未知データ）
                  </p>
                </div>
              </section>

              {/* S3データセット情報 */}
              {dataSource === 's3' && selectedS3Dataset && (
                <section className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 rounded-lg bg-sky-500/20">
                      <Cloud className="w-5 h-5 text-sky-400" />
                    </div>
                    <h2 className="text-lg font-semibold text-white">選択中のデータセット</h2>
                  </div>

                  <div className="bg-zinc-900/50 rounded-lg p-4">
                    <div className="font-medium text-white mb-2">{selectedS3Dataset.name}</div>
                    <div className="flex items-center gap-4 text-sm text-zinc-400">
                      <span>{selectedS3Dataset.fileCount} ファイル</span>
                      <span>{formatBytes(selectedS3Dataset.size)}</span>
                      <span>最終更新: {selectedS3Dataset.lastModified.toLocaleString('ja-JP')}</span>
                    </div>
                    <div className="mt-2 text-xs text-zinc-500 font-mono">{selectedS3Dataset.path}</div>
                  </div>
                </section>
              )}

              {/* ナビゲーションボタン */}
              <div className="flex flex-col gap-3">
                {/* 設定が不完全な場合の警告 */}
                {!isConfigComplete && (
                  <div className="bg-amber-500/10 border border-amber-500/50 rounded-lg p-3 text-sm text-amber-400">
                    <p className="font-medium mb-1">次へ進むには以下が必要です：</p>
                    <ul className="list-disc list-inside space-y-0.5 text-amber-300">
                      {!hasDataSource && <li>データソースを選択してください</li>}
                      {!datasetInfo && <li>データセット情報が読み込まれていません</li>}
                      {!targetConfig && <li>ターゲットフィールドを選択してください</li>}
                    </ul>
                  </div>
                )}

                <div className="flex justify-between">
                  <button
                    onClick={() => setCurrentStep('select-source')}
                    className="flex items-center gap-2 px-6 py-3 bg-zinc-700 hover:bg-zinc-600 text-white rounded-xl font-semibold transition-all"
                  >
                    戻る
                  </button>
                  <button
                    onClick={() => {
                      executeDataSplit();
                      setCurrentStep('augment');
                    }}
                    disabled={!isConfigComplete}
                    className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${isConfigComplete
                        ? 'bg-sky-500 hover:bg-sky-600 text-white'
                        : 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                      }`}
                  >
                    次へ：分割＆拡張
                    <ChevronRight className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Step 3: 分割＆拡張 */}
          {currentStep === 'augment' && (
            <div className="space-y-6">
              {/* 分割結果のプレビュー */}
              {splitStats && datasetInfo && (
                <section className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 rounded-lg bg-sky-500/20">
                      <Scissors className="w-5 h-5 text-sky-400" />
                    </div>
                    <div>
                      <h2 className="text-lg font-semibold text-white">データ分割結果</h2>
                      <p className="text-sm text-zinc-400">
                        元データを Train / Validation / Test に分割しました
                      </p>
                    </div>
                  </div>

                  <DataSplitPreview stats={splitStats} classes={datasetInfo.classes} />
                </section>
              )}

              {/* 訓練データの拡張 */}
              {dataSplit && (
                <TrainDataAugmenter
                  trainFiles={dataSplit.train}
                  getLabel={getLabelForFile}
                  onAugmentationComplete={handleAugmentationComplete}
                />
              )}

              {/* 訓練設定 */}
              <section className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-2 rounded-lg bg-fuchsia-500/20">
                    <Settings className="w-5 h-5 text-fuchsia-400" />
                  </div>
                  <h2 className="text-lg font-semibold text-white">訓練パラメータ</h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <div className="flex items-center gap-1 mb-2">
                      <label className="text-sm text-zinc-400">エポック数: {config.epochs}</label>
                      <ParameterHelp {...PARAM_HELP.epochs} />
                    </div>
                    <input
                      type="range"
                      min="10"
                      max="200"
                      step="10"
                      value={config.epochs}
                      onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <div className="flex items-center gap-1 mb-2">
                      <label className="text-sm text-zinc-400">バッチサイズ: {config.batchSize}</label>
                      <ParameterHelp {...PARAM_HELP.batchSize} />
                    </div>
                    <input
                      type="range"
                      min="8"
                      max="128"
                      step="8"
                      value={config.batchSize}
                      onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) })}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <div className="flex items-center gap-1 mb-2">
                      <label className="text-sm text-zinc-400">学習率: {config.learningRate}</label>
                      <ParameterHelp {...PARAM_HELP.learningRate} />
                    </div>
                    <input
                      type="range"
                      min="0.0001"
                      max="0.01"
                      step="0.0001"
                      value={config.learningRate}
                      onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) })}
                      className="w-full"
                    />
                  </div>
                </div>

                {/* スマート推奨 */}
                {effectiveStats && (
                  <div className="mt-4">
                    <SmartRecommendation
                      stats={effectiveStats}
                      currentParams={config}
                      onApplyRecommendation={(params) => setConfig({ ...config, ...params })}
                    />
                  </div>
                )}
              </section>

              {/* ナビゲーションボタン */}
              <div className="flex justify-between">
                <button
                  onClick={() => setCurrentStep('configure')}
                  className="flex items-center gap-2 px-6 py-3 bg-zinc-700 hover:bg-zinc-600 text-white rounded-xl font-semibold transition-all"
                >
                  戻る
                </button>
                <button
                  onClick={() => setCurrentStep('training')}
                  disabled={!isAugmentationComplete && dataSource === 'local'}
                  className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${(isAugmentationComplete || dataSource === 's3')
                      ? 'bg-sky-500 hover:bg-sky-600 text-white'
                      : 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                    }`}
                >
                  {!isAugmentationComplete && dataSource === 'local'
                    ? '拡張を実行してください'
                    : '次へ：訓練'}
                  <ChevronRight className="w-5 h-5" />
                </button>
              </div>
            </div>
          )}

          {/* Step 4: クラウド訓練 */}
          {currentStep === 'training' && (
            <div className="space-y-6">
              <CloudTraining
                userId={userId}
                config={config}
                datasetInfo={datasetInfo}
                targetField={targetConfig?.fieldIndex?.toString() || '0'}
                auxiliaryFields={auxiliaryFields.map(f => f.fieldIndex.toString())}
                fieldLabels={metadata?.fields.map(f => ({ index: f.index, label: f.label })) || []}
                problemType={problemType}
                tolerance={tolerance}
                fileInfoList={finalFileInfoList}
                s3DatasetPath={dataSource === 's3' ? selectedS3Dataset?.path : undefined}
                onModelReady={(modelPath) => {
                  console.log('Cloud model ready:', modelPath);
                  setIsTrainingStarted(false);
                }}
                onTrainingStart={() => setIsTrainingStarted(true)}
                onUploadComplete={saveMetadataToS3}
              />

              {/* 戻るボタン（訓練中でなければ表示） */}
              {!isTrainingStarted && (
                <div className="flex justify-start">
                  <button
                    onClick={() => setCurrentStep('augment')}
                    className="flex items-center gap-2 px-6 py-3 bg-zinc-700 hover:bg-zinc-600 text-white rounded-xl font-semibold transition-all"
                  >
                    戻る
                  </button>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

