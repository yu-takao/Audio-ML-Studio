/**
 * ファイル名からメタデータを解析するユーティリティ
 */

export interface MetadataField {
  index: number;
  name: string;
  label: string;
  uniqueValues: string[];
  valueCount: number;
  isNumeric: boolean;
  minValue?: number;
  maxValue?: number;
}

export interface ParsedMetadata {
  fields: MetadataField[];
  folderField?: MetadataField; // フォルダ名もメタデータとして扱う
  separator: string;
  sampleCount: number;
  rawExamples: string[];
}

// デフォルトのフィールド名（汎用的な名前、ユーザーが変更可能）
// 特定のドメイン知識を前提としない汎用的な名前を使用

/**
 * 値が数値かどうかを判定
 */
function isNumericValue(value: string): boolean {
  return /^-?\d+(\.\d+)?$/.test(value);
}

/**
 * ファイル名リストからメタデータ構造を解析
 */
export function analyzeFilenames(
  filenames: string[],
  folderNames?: string[],
  separator: string = '_'
): ParsedMetadata {
  if (filenames.length === 0) {
    return {
      fields: [],
      separator,
      sampleCount: 0,
      rawExamples: [],
    };
  }

  // 拡張子を除去してファイル名を分割
  const parsedFiles = filenames.map((filename) => {
    const nameWithoutExt = filename.replace(/\.[^/.]+$/, '');
    return nameWithoutExt.split(separator);
  });

  // 最も一般的なフィールド数を見つける
  const fieldCounts = new Map<number, number>();
  parsedFiles.forEach((parts) => {
    fieldCounts.set(parts.length, (fieldCounts.get(parts.length) || 0) + 1);
  });

  let mostCommonFieldCount = 0;
  let maxCount = 0;
  fieldCounts.forEach((count, fieldCount) => {
    if (count > maxCount) {
      maxCount = count;
      mostCommonFieldCount = fieldCount;
    }
  });

  // 同じフィールド数のファイルのみを使用
  const validParsedFiles = parsedFiles.filter(
    (parts) => parts.length === mostCommonFieldCount
  );

  // 各フィールドの情報を収集
  const fields: MetadataField[] = [];
  
  for (let i = 0; i < mostCommonFieldCount; i++) {
    const values = validParsedFiles.map((parts) => parts[i]);
    const uniqueValues = [...new Set(values)].sort();
    const numericValues = values.filter(isNumericValue).map(Number);
    const isNumeric = numericValues.length === values.length;

    fields.push({
      index: i,
      name: `field_${i}`,
      label: `フィールド ${i + 1}`,  // ユーザーが変更可能な汎用的な名前
      uniqueValues,
      valueCount: uniqueValues.length,
      isNumeric,
      minValue: isNumeric ? Math.min(...numericValues) : undefined,
      maxValue: isNumeric ? Math.max(...numericValues) : undefined,
    });
  }

  // フォルダ名のメタデータ
  let folderField: MetadataField | undefined;
  if (folderNames && folderNames.length > 0) {
    const uniqueFolders = [...new Set(folderNames)].sort();
    folderField = {
      index: -1,
      name: 'folder',
      label: 'フォルダ (タイヤ種類)',
      uniqueValues: uniqueFolders,
      valueCount: uniqueFolders.length,
      isNumeric: false,
    };
  }

  return {
    fields,
    folderField,
    separator,
    sampleCount: validParsedFiles.length,
    rawExamples: filenames.slice(0, 5),
  };
}

/**
 * ファイル名から特定のフィールドの値を抽出
 */
export function extractFieldValue(
  filename: string,
  fieldIndex: number,
  separator: string = '_'
): string | undefined {
  const nameWithoutExt = filename.replace(/\.[^/.]+$/, '');
  const parts = nameWithoutExt.split(separator);
  return parts[fieldIndex];
}

/**
 * 数値フィールドを範囲でグループ化
 */
export function groupNumericValues(
  values: number[],
  ranges: { min: number; max: number; label: string }[]
): Map<string, number[]> {
  const groups = new Map<string, number[]>();
  
  ranges.forEach((range) => {
    groups.set(range.label, []);
  });
  groups.set('その他', []);

  values.forEach((value) => {
    let matched = false;
    for (const range of ranges) {
      if (value >= range.min && value < range.max) {
        groups.get(range.label)?.push(value);
        matched = true;
        break;
      }
    }
    if (!matched) {
      groups.get('その他')?.push(value);
    }
  });

  return groups;
}

/**
 * ターゲットフィールドの設定
 */
export interface TargetFieldConfig {
  fieldIndex: number; // -1 = folder
  fieldName: string;
  useAsTarget: boolean;
  groupingMode: 'individual' | 'range' | 'custom';
  ranges?: { min: number; max: number; label: string }[];
  customGroups?: { values: string[]; label: string }[];
  problemType?: 'classification' | 'regression';
  tolerance?: number;
}

/**
 * 補助パラメータの設定
 */
export interface AuxiliaryFieldConfig {
  fieldIndex: number;
  fieldName: string;
  normalize: boolean; // 数値を正規化するか
  minValue?: number;
  maxValue?: number;
}

/**
 * ターゲット設定に基づいてクラスラベルを生成
 */
export function generateClassLabel(
  filename: string,
  folderName: string | undefined,
  config: TargetFieldConfig,
  separator: string = '_'
): string {
  let value: string;

  if (config.fieldIndex === -1) {
    // フォルダ名を使用
    value = folderName || 'unknown';
  } else {
    // ファイル名のフィールドを使用
    value = extractFieldValue(filename, config.fieldIndex, separator) || 'unknown';
  }

  // グループ化モードに応じてラベルを生成
  switch (config.groupingMode) {
    case 'individual':
      return value;

    case 'range':
      if (config.ranges && isNumericValue(value)) {
        const numValue = parseFloat(value);
        for (const range of config.ranges) {
          if (numValue >= range.min && numValue < range.max) {
            return range.label;
          }
        }
        return 'その他';
      }
      return value;

    case 'custom':
      if (config.customGroups) {
        for (const group of config.customGroups) {
          if (group.values.includes(value)) {
            return group.label;
          }
        }
        return 'その他';
      }
      return value;

    default:
      return value;
  }
}

/**
 * デフォルトの空気圧範囲設定
 */
export const DEFAULT_PRESSURE_RANGES = [
  { min: 0, max: 200, label: '低圧 (<200kPa)' },
  { min: 200, max: 240, label: '正常 (200-240kPa)' },
  { min: 240, max: 999, label: '高圧 (>240kPa)' },
];

