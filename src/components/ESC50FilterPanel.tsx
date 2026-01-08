/**
 * ESC-50 フィルタ設定パネル
 */

import { useState } from 'react';
import type { ESC50CategoryKey, ESC50FilterOptions } from '../utils/esc50';
import {
  ESC50_CATEGORIES,
  CLASS_NAMES_JA,
} from '../utils/esc50';
import {
  Filter,
  ChevronDown,
  ChevronUp,
  Dog,
  TreePine,
  User,
  Home,
  Building2,
  CheckSquare,
  Square,
} from 'lucide-react';

interface Props {
  filterOptions: ESC50FilterOptions;
  onChange: (options: ESC50FilterOptions) => void;
  totalCount: number;
  filteredCount: number;
}

const CATEGORY_ICONS: Record<ESC50CategoryKey, React.ReactNode> = {
  animals: <Dog className="w-4 h-4" />,
  natural: <TreePine className="w-4 h-4" />,
  human: <User className="w-4 h-4" />,
  interior: <Home className="w-4 h-4" />,
  exterior: <Building2 className="w-4 h-4" />,
};

const CATEGORY_COLORS: Record<ESC50CategoryKey, string> = {
  animals: 'bg-amber-500/20 text-amber-400 border-amber-500/50',
  natural: 'bg-green-500/20 text-green-400 border-green-500/50',
  human: 'bg-blue-500/20 text-blue-400 border-blue-500/50',
  interior: 'bg-purple-500/20 text-purple-400 border-purple-500/50',
  exterior: 'bg-red-500/20 text-red-400 border-red-500/50',
};

export function ESC50FilterPanel({ filterOptions, onChange, totalCount, filteredCount }: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showClassSelector, setShowClassSelector] = useState(false);

  const toggleCategory = (key: ESC50CategoryKey) => {
    const newCategories = filterOptions.categories.includes(key)
      ? filterOptions.categories.filter((c) => c !== key)
      : [...filterOptions.categories, key];
    onChange({ ...filterOptions, categories: newCategories, selectedClasses: [] });
  };

  const toggleFold = (fold: number) => {
    const newFolds = filterOptions.folds.includes(fold)
      ? filterOptions.folds.filter((f) => f !== fold)
      : [...filterOptions.folds, fold];
    onChange({ ...filterOptions, folds: newFolds });
  };

  const toggleESC10Only = () => {
    onChange({ ...filterOptions, esc10Only: !filterOptions.esc10Only });
  };

  const toggleClass = (className: string) => {
    const newClasses = filterOptions.selectedClasses.includes(className)
      ? filterOptions.selectedClasses.filter((c) => c !== className)
      : [...filterOptions.selectedClasses, className];
    onChange({ ...filterOptions, selectedClasses: newClasses });
  };

  const selectAllCategories = () => {
    onChange({
      ...filterOptions,
      categories: Object.keys(ESC50_CATEGORIES) as ESC50CategoryKey[],
      selectedClasses: [],
    });
  };

  const clearAllCategories = () => {
    onChange({ ...filterOptions, categories: [], selectedClasses: [] });
  };

  // 全クラス一覧（カテゴリごと）
  // @ts-expect-error - Reserved for future use
  const allClasses = Object.entries(ESC50_CATEGORIES).flatMap(([key, cat]) =>
    cat.classes.map((cls) => ({ category: key as ESC50CategoryKey, className: cls }))
  );

  return (
    <div className="bg-zinc-800/50 rounded-xl border border-zinc-700 overflow-hidden">
      {/* ヘッダー */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-zinc-700/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Filter className="w-5 h-5 text-orange-400" />
          <span className="font-semibold text-white">ESC-50 フィルタ設定</span>
          <span className="text-sm text-zinc-500">
            ({filteredCount} / {totalCount} 使用)
          </span>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-zinc-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-zinc-400" />
        )}
      </button>

      {/* 展開コンテンツ */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-4">
          {/* カテゴリ選択 */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-zinc-300">カテゴリ</span>
              <div className="flex gap-2">
                <button
                  onClick={selectAllCategories}
                  className="text-xs text-violet-400 hover:text-violet-300"
                >
                  すべて選択
                </button>
                <span className="text-zinc-600">|</span>
                <button
                  onClick={clearAllCategories}
                  className="text-xs text-zinc-400 hover:text-zinc-300"
                >
                  クリア
                </button>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              {(Object.keys(ESC50_CATEGORIES) as ESC50CategoryKey[]).map((key) => {
                const cat = ESC50_CATEGORIES[key];
                const isSelected = filterOptions.categories.includes(key);
                return (
                  <button
                    key={key}
                    onClick={() => toggleCategory(key)}
                    className={`
                      flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm transition-all
                      ${isSelected ? CATEGORY_COLORS[key] : 'bg-zinc-700/50 text-zinc-500 border-zinc-600'}
                    `}
                  >
                    {CATEGORY_ICONS[key]}
                    <span>{cat.name}</span>
                    <span className="text-xs opacity-70">({cat.classes.length})</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* クラス詳細選択 */}
          <div>
            <button
              onClick={() => setShowClassSelector(!showClassSelector)}
              className="text-sm text-violet-400 hover:text-violet-300 flex items-center gap-1"
            >
              {showClassSelector ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              個別クラスを選択
              {filterOptions.selectedClasses.length > 0 && (
                <span className="ml-1 px-1.5 py-0.5 bg-violet-500/20 rounded text-xs">
                  {filterOptions.selectedClasses.length}個選択中
                </span>
              )}
            </button>
            
            {showClassSelector && (
              <div className="mt-2 max-h-64 overflow-y-auto bg-zinc-900/50 rounded-lg p-3 space-y-3">
                {(Object.keys(ESC50_CATEGORIES) as ESC50CategoryKey[]).map((catKey) => {
                  const cat = ESC50_CATEGORIES[catKey];
                  return (
                    <div key={catKey}>
                      <div className={`text-xs font-medium mb-1.5 ${CATEGORY_COLORS[catKey].split(' ')[1]}`}>
                        {cat.name}
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {cat.classes.map((cls) => {
                          const isSelected = filterOptions.selectedClasses.includes(cls);
                          return (
                            <button
                              key={cls}
                              onClick={() => toggleClass(cls)}
                              className={`
                                flex items-center gap-1 px-2 py-1 rounded text-xs transition-all
                                ${isSelected
                                  ? 'bg-violet-500/30 text-violet-300 border border-violet-500/50'
                                  : 'bg-zinc-700/50 text-zinc-400 hover:bg-zinc-700 border border-transparent'
                                }
                              `}
                            >
                              {isSelected ? <CheckSquare className="w-3 h-3" /> : <Square className="w-3 h-3" />}
                              {CLASS_NAMES_JA[cls] || cls}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* ESC-10 / Fold 設定 */}
          <div className="flex flex-wrap gap-4">
            {/* ESC-10のみ */}
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={filterOptions.esc10Only}
                onChange={toggleESC10Only}
                className="w-4 h-4 rounded bg-zinc-700 border-zinc-600 text-violet-500 focus:ring-violet-500"
              />
              <span className="text-sm text-zinc-300">ESC-10のみ</span>
              <span className="text-xs text-zinc-500">(10クラス, CC BY)</span>
            </label>

            {/* Fold選択 */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-zinc-400">Fold:</span>
              {[1, 2, 3, 4, 5].map((fold) => {
                const isSelected = filterOptions.folds.includes(fold);
                return (
                  <button
                    key={fold}
                    onClick={() => toggleFold(fold)}
                    className={`
                      w-7 h-7 rounded text-sm font-medium transition-all
                      ${isSelected
                        ? 'bg-violet-500/30 text-violet-300 border border-violet-500/50'
                        : 'bg-zinc-700/50 text-zinc-500 hover:bg-zinc-700 border border-transparent'
                      }
                    `}
                  >
                    {fold}
                  </button>
                );
              })}
            </div>
          </div>

          {/* 説明 */}
          <div className="text-xs text-zinc-500 bg-zinc-900/50 rounded-lg p-3">
            <p><strong>カテゴリ:</strong> 使用するノイズの種類を大カテゴリで選択</p>
            <p><strong>個別クラス:</strong> より細かく特定の音だけを選択（例: エンジン音のみ）</p>
            <p><strong>ESC-10:</strong> 10クラスのサブセット（CC BYライセンス）</p>
            <p><strong>Fold:</strong> クロスバリデーション用の分割（1〜5）</p>
          </div>
        </div>
      )}
    </div>
  );
}

