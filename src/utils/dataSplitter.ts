/**
 * データセット分割ユーティリティ
 * Train/Validation/Testにクラスごとに層化分割する
 */

export interface FileInfo {
  file: File;
  path: string;
  folderName: string;
}

export interface SplitResult {
  train: FileInfo[];
  validation: FileInfo[];
  test: FileInfo[];
}

export interface SplitConfig {
  validationSplit: number; // 0.0 - 1.0
  testSplit: number;       // 0.0 - 1.0
  seed?: number;           // ランダムシード（再現性のため）
}

/**
 * シードベースの疑似乱数生成器（再現性のため）
 */
function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = Math.sin(s) * 10000;
    return s - Math.floor(s);
  };
}

/**
 * 配列をシャッフル（破壊的でない）
 */
function shuffleArray<T>(array: T[], random: () => number): T[] {
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/**
 * ファイルをクラスごとにグループ化
 */
export function groupFilesByClass(
  files: FileInfo[],
  getLabel: (file: FileInfo) => string
): Map<string, FileInfo[]> {
  const groups = new Map<string, FileInfo[]>();
  
  for (const file of files) {
    const label = getLabel(file);
    const group = groups.get(label) || [];
    group.push(file);
    groups.set(label, group);
  }
  
  return groups;
}

/**
 * 層化分割を実行
 * 各クラスから指定の割合でデータを抽出してTrain/Val/Testに分ける
 */
export function stratifiedSplit(
  filesByClass: Map<string, FileInfo[]>,
  config: SplitConfig
): SplitResult {
  const random = seededRandom(config.seed ?? Date.now());
  
  const train: FileInfo[] = [];
  const validation: FileInfo[] = [];
  const test: FileInfo[] = [];
  
  for (const [className, files] of filesByClass) {
    // シャッフル
    const shuffled = shuffleArray(files, random);
    const total = shuffled.length;
    
    // 最低1サンプルずつ各セットに入れることを保証（可能な場合）
    const testCount = Math.max(1, Math.round(total * config.testSplit));
    const valCount = Math.max(1, Math.round(total * config.validationSplit));
    const trainCount = total - testCount - valCount;
    
    if (trainCount < 1) {
      // データが少なすぎる場合は警告を出しつつ、可能な限り分割
      console.warn(`Class "${className}" has only ${total} samples. Distribution may be uneven.`);
    }
    
    // 分割
    let idx = 0;
    
    // Test
    for (let i = 0; i < Math.min(testCount, total); i++) {
      test.push(shuffled[idx++]);
    }
    
    // Validation
    for (let i = 0; i < Math.min(valCount, total - idx); i++) {
      validation.push(shuffled[idx++]);
    }
    
    // Train（残り全部）
    while (idx < total) {
      train.push(shuffled[idx++]);
    }
  }
  
  return { train, validation, test };
}

/**
 * 分割統計を計算
 */
export interface SplitStats {
  train: { total: number; byClass: Map<string, number> };
  validation: { total: number; byClass: Map<string, number> };
  test: { total: number; byClass: Map<string, number> };
}

export function calculateSplitStats(
  split: SplitResult,
  getLabel: (file: FileInfo) => string
): SplitStats {
  const countByClass = (files: FileInfo[]): Map<string, number> => {
    const counts = new Map<string, number>();
    for (const file of files) {
      const label = getLabel(file);
      counts.set(label, (counts.get(label) || 0) + 1);
    }
    return counts;
  };
  
  return {
    train: {
      total: split.train.length,
      byClass: countByClass(split.train),
    },
    validation: {
      total: split.validation.length,
      byClass: countByClass(split.validation),
    },
    test: {
      total: split.test.length,
      byClass: countByClass(split.test),
    },
  };
}


