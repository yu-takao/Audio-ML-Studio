import { useState } from 'react';
import {
  Settings,
  Tag,
  Hash,
  Folder,
  ChevronDown,
  ChevronUp,
  Target,
  Layers,
  Plus,
} from 'lucide-react';
import type { ParsedMetadata, MetadataField, TargetFieldConfig, AuxiliaryFieldConfig } from '../utils/metadataParser';

interface MetadataConfigProps {
  metadata: ParsedMetadata;
  onFieldLabelChange: (index: number, label: string) => void;
  targetConfig: TargetFieldConfig | null;
  onTargetConfigChange: (config: TargetFieldConfig | null) => void;
  auxiliaryFields: AuxiliaryFieldConfig[];
  onAuxiliaryFieldsChange: (fields: AuxiliaryFieldConfig[]) => void;
}

// ユーザーがカスタム範囲を設定できるようにするためのデフォルト値
const DEFAULT_RANGES = [
  { min: 0, max: 100, label: '範囲1' },
  { min: 100, max: 200, label: '範囲2' },
  { min: 200, max: 999, label: '範囲3' },
];

export function MetadataConfig({
  metadata,
  onFieldLabelChange,
  targetConfig,
  onTargetConfigChange,
  auxiliaryFields,
  onAuxiliaryFieldsChange,
}: MetadataConfigProps) {
  const [expandedField, setExpandedField] = useState<number | null>(null);

  const handleTargetSelect = (field: MetadataField) => {
    const isCurrentTarget = targetConfig?.fieldIndex === field.index;
    
    if (isCurrentTarget) {
      onTargetConfigChange(null);
    } else {
      // 補助パラメータから除外
      onAuxiliaryFieldsChange(auxiliaryFields.filter(f => f.fieldIndex !== field.index));
      
      onTargetConfigChange({
        fieldIndex: field.index,
        fieldName: field.label,
        useAsTarget: true,
        groupingMode: 'individual',
      });
    }
  };

  const handleAuxiliaryToggle = (field: MetadataField) => {
    const isAuxiliary = auxiliaryFields.some(f => f.fieldIndex === field.index);
    
    if (isAuxiliary) {
      onAuxiliaryFieldsChange(auxiliaryFields.filter(f => f.fieldIndex !== field.index));
    } else {
      // ターゲットとして選択されていたら解除
      if (targetConfig?.fieldIndex === field.index) {
        onTargetConfigChange(null);
      }
      
      onAuxiliaryFieldsChange([
        ...auxiliaryFields,
        {
          fieldIndex: field.index,
          fieldName: field.label,
          normalize: field.isNumeric,
          minValue: field.minValue,
          maxValue: field.maxValue,
        },
      ]);
    }
  };

  const handleGroupingModeChange = (mode: 'individual' | 'range') => {
    if (!targetConfig) return;
    onTargetConfigChange({
      ...targetConfig,
      groupingMode: mode,
      ranges: mode === 'range' ? DEFAULT_RANGES : undefined,
    });
  };

  const renderFieldCard = (field: MetadataField, isFolder: boolean = false) => {
    const isExpanded = expandedField === field.index;
    const isTarget = targetConfig?.fieldIndex === field.index;
    const isAuxiliary = auxiliaryFields.some(f => f.fieldIndex === field.index);
    const displayIndex = isFolder ? 'F' : field.index;

    return (
      <div
        key={field.index}
        className={`
          bg-zinc-900/50 rounded-lg border transition-all
          ${isTarget 
            ? 'border-violet-500 ring-1 ring-violet-500/50' 
            : isAuxiliary 
              ? 'border-cyan-500 ring-1 ring-cyan-500/50'
              : 'border-zinc-700'}
        `}
      >
        {/* ヘッダー */}
        <div
          className="flex items-center gap-3 p-3 cursor-pointer"
          onClick={() => setExpandedField(isExpanded ? null : field.index)}
        >
          <div className={`
            w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold
            ${isFolder ? 'bg-amber-500/20 text-amber-400' : 'bg-zinc-800 text-zinc-400'}
          `}>
            {displayIndex}
          </div>
          
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              {isFolder ? (
                <Folder className="w-4 h-4 text-amber-400" />
              ) : field.isNumeric ? (
                <Hash className="w-4 h-4 text-blue-400" />
              ) : (
                <Tag className="w-4 h-4 text-green-400" />
              )}
              <input
                type="text"
                value={field.label}
                onChange={(e) => onFieldLabelChange(field.index, e.target.value)}
                onClick={(e) => e.stopPropagation()}
                className="bg-transparent border-none text-white font-medium focus:outline-none focus:ring-1 focus:ring-violet-500 rounded px-1 -mx-1"
              />
            </div>
            <div className="text-xs text-zinc-500 mt-0.5">
              {field.valueCount} 種類の値
              {field.isNumeric && field.minValue !== undefined && (
                <span className="ml-2">
                  範囲: {field.minValue} - {field.maxValue}
                </span>
              )}
            </div>
          </div>

          {/* ボタン群 */}
          <div className="flex gap-2">
            {/* ターゲット選択ボタン */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleTargetSelect(field);
              }}
              className={`
                px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1
                ${isTarget
                  ? 'bg-violet-500 text-white'
                  : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-white'
                }
              `}
            >
              <Target className="w-3 h-3" />
              {isTarget ? 'ターゲット' : 'ターゲット'}
            </button>

            {/* 補助パラメータ選択ボタン */}
            {field.isNumeric && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleAuxiliaryToggle(field);
                }}
                className={`
                  px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1
                  ${isAuxiliary
                    ? 'bg-cyan-500 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-white'
                  }
                `}
              >
                <Plus className="w-3 h-3" />
                {isAuxiliary ? '補助' : '補助'}
              </button>
            )}
          </div>

          {isExpanded ? (
            <ChevronUp className="w-5 h-5 text-zinc-500" />
          ) : (
            <ChevronDown className="w-5 h-5 text-zinc-500" />
          )}
        </div>

        {/* 展開コンテンツ */}
        {isExpanded && (
          <div className="px-3 pb-3 border-t border-zinc-800 pt-3">
            {/* ラベル名入力 */}
            <div className="mb-3">
              <label className="text-xs text-zinc-500 mb-1 block">フィールド名を入力</label>
              <input
                type="text"
                value={field.label}
                onChange={(e) => onFieldLabelChange(field.index, e.target.value)}
                placeholder="例: 空気圧、サイズ、温度など"
                className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
              />
              <p className="text-xs text-zinc-500 mt-1">
                このフィールドが何を表すか、わかりやすい名前をつけてください
              </p>
            </div>

            {/* 値のプレビュー */}
            <div>
              <label className="text-xs text-zinc-500 mb-1 block">検出された値</label>
              <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto">
                {field.uniqueValues.slice(0, 20).map((value) => (
                  <span
                    key={value}
                    className="px-2 py-0.5 bg-zinc-800 rounded text-xs text-zinc-300"
                  >
                    {value}
                  </span>
                ))}
                {field.uniqueValues.length > 20 && (
                  <span className="px-2 py-0.5 text-xs text-zinc-500">
                    ...他 {field.uniqueValues.length - 20} 件
                  </span>
                )}
              </div>
            </div>

            {/* ターゲット設定（選択されている場合） */}
            {isTarget && (
              <div className="mt-3 p-3 bg-violet-500/10 rounded-lg border border-violet-500/30">
                <div className="flex items-center gap-2 mb-2">
                  <Layers className="w-4 h-4 text-violet-400" />
                  <span className="text-sm font-medium text-violet-300">クラス分類設定</span>
                </div>

                {/* グループ化モード */}
                <div className="flex gap-2 mb-2">
                  <button
                    onClick={() => handleGroupingModeChange('individual')}
                    className={`
                      flex-1 px-3 py-2 rounded text-sm transition-all
                      ${targetConfig?.groupingMode === 'individual'
                        ? 'bg-violet-500 text-white'
                        : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
                      }
                    `}
                  >
                    個別値
                  </button>
                  {field.isNumeric && (
                    <button
                      onClick={() => handleGroupingModeChange('range')}
                      className={`
                        flex-1 px-3 py-2 rounded text-sm transition-all
                        ${targetConfig?.groupingMode === 'range'
                          ? 'bg-violet-500 text-white'
                          : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
                        }
                      `}
                    >
                      範囲グループ
                    </button>
                  )}
                </div>

                {/* 範囲設定 */}
                {targetConfig?.groupingMode === 'range' && targetConfig.ranges && (
                  <div className="space-y-2">
                    <div className="text-xs text-zinc-400">範囲設定:</div>
                    {targetConfig.ranges.map((range, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        <span className="text-zinc-500">{range.min}-{range.max}:</span>
                        <span className="text-white">{range.label}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* 生成されるクラス数 */}
                <div className="mt-2 text-xs text-zinc-400">
                  生成されるクラス数:{' '}
                  <span className="text-violet-300 font-medium">
                    {targetConfig?.groupingMode === 'range' && targetConfig.ranges 
                      ? targetConfig.ranges.length 
                      : field.valueCount}
                  </span>
                </div>
              </div>
            )}

            {/* 補助パラメータ設定（選択されている場合） */}
            {isAuxiliary && (
              <div className="mt-3 p-3 bg-cyan-500/10 rounded-lg border border-cyan-500/30">
                <div className="flex items-center gap-2">
                  <Plus className="w-4 h-4 text-cyan-400" />
                  <span className="text-sm font-medium text-cyan-300">補助パラメータ</span>
                </div>
                <div className="text-xs text-zinc-400 mt-1">
                  この値はモデルに追加入力として与えられます
                </div>
                {field.isNumeric && (
                  <div className="text-xs text-cyan-300 mt-1">
                    正規化範囲: {field.minValue} - {field.maxValue}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-cyan-500/20">
          <Settings className="w-5 h-5 text-cyan-400" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">メタデータ設定</h2>
          <p className="text-sm text-zinc-400">
            ファイル名から {metadata.fields.length} 個のフィールドを検出しました
          </p>
        </div>
      </div>

      {/* 使い方の説明 */}
      <div className="mb-4 p-3 bg-zinc-900/50 rounded-lg border border-zinc-700">
        <div className="text-xs text-zinc-400 space-y-1">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 bg-violet-500 text-white rounded text-xs">ターゲット</span>
            <span>= 予測したい値（分類対象）</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 bg-cyan-500 text-white rounded text-xs">補助</span>
            <span>= モデルに追加で与える情報（数値フィールドのみ）</span>
          </div>
        </div>
      </div>

      {/* サンプルファイル名 */}
      <div className="mb-4 p-3 bg-zinc-900/50 rounded-lg">
        <div className="text-xs text-zinc-500 mb-1">サンプルファイル名:</div>
        <div className="text-sm text-zinc-300 font-mono break-all">
          {metadata.rawExamples[0] || '(なし)'}
        </div>
      </div>

      {/* フォルダフィールド */}
      {metadata.folderField && (
        <div className="mb-4">
          <div className="text-sm text-zinc-400 mb-2 flex items-center gap-2">
            <Folder className="w-4 h-4" />
            フォルダ情報
          </div>
          {renderFieldCard(metadata.folderField, true)}
        </div>
      )}

      {/* ファイル名フィールド */}
      <div>
        <div className="text-sm text-zinc-400 mb-2 flex items-center gap-2">
          <Tag className="w-4 h-4" />
          ファイル名フィールド (区切り: "{metadata.separator}")
        </div>
        <div className="space-y-2">
          {metadata.fields.map((field) => renderFieldCard(field))}
        </div>
      </div>

      {/* 選択されたターゲットと補助パラメータの概要 */}
      {(targetConfig || auxiliaryFields.length > 0) && (
        <div className="mt-4 space-y-3">
          {/* ターゲット */}
          {targetConfig && (
            <div className="p-4 bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10 rounded-lg border border-violet-500/30">
              <div className="flex items-center gap-2">
                <Target className="w-5 h-5 text-violet-400" />
                <span className="text-white font-medium">分類ターゲット:</span>
                <span className="text-violet-300">{targetConfig.fieldName}</span>
              </div>
              <div className="text-sm text-zinc-400 mt-1">
                モード: {targetConfig.groupingMode === 'range' ? '範囲グループ' : '個別値'}
              </div>
            </div>
          )}

          {/* 補助パラメータ */}
          {auxiliaryFields.length > 0 && (
            <div className="p-4 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 rounded-lg border border-cyan-500/30">
              <div className="flex items-center gap-2 mb-2">
                <Plus className="w-5 h-5 text-cyan-400" />
                <span className="text-white font-medium">補助パラメータ:</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {auxiliaryFields.map((field) => (
                  <span
                    key={field.fieldIndex}
                    className="px-3 py-1 bg-cyan-500/20 text-cyan-300 rounded-lg text-sm"
                  >
                    {field.fieldName}
                  </span>
                ))}
              </div>
              <div className="text-xs text-zinc-400 mt-2">
                これらの値は正規化されてモデルに入力されます
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
