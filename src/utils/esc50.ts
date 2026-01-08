/**
 * ESC-50 データセットのメタデータ管理
 */

export interface ESC50Entry {
  filename: string;
  fold: number;
  target: number;
  category: string;
  esc10: boolean;
  src_file: string;
  take: string;
}

// 5つの大カテゴリとそれに含まれるtarget番号
export const ESC50_CATEGORIES = {
  animals: {
    name: '動物',
    nameEn: 'Animals',
    targets: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    classes: ['dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow'],
  },
  natural: {
    name: '自然音・水',
    nameEn: 'Natural soundscapes & water sounds',
    targets: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    classes: ['rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm'],
  },
  human: {
    name: '人間の音',
    nameEn: 'Human, non-speech sounds',
    targets: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    classes: ['crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping'],
  },
  interior: {
    name: '室内音',
    nameEn: 'Interior/domestic sounds',
    targets: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    classes: ['door_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking'],
  },
  exterior: {
    name: '屋外・都市音',
    nameEn: 'Exterior/urban noises',
    targets: [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    classes: ['helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'],
  },
} as const;

export type ESC50CategoryKey = keyof typeof ESC50_CATEGORIES;

// 全50クラスの日本語名
export const CLASS_NAMES_JA: Record<string, string> = {
  // Animals
  dog: '犬',
  rooster: '鶏（オンドリ）',
  pig: '豚',
  cow: '牛',
  frog: 'カエル',
  cat: '猫',
  hen: '鶏（メンドリ）',
  insects: '虫（飛行）',
  sheep: '羊',
  crow: 'カラス',
  // Natural
  rain: '雨',
  sea_waves: '波',
  crackling_fire: '焚き火',
  crickets: 'コオロギ',
  chirping_birds: '鳥のさえずり',
  water_drops: '水滴',
  wind: '風',
  pouring_water: '水を注ぐ音',
  toilet_flush: 'トイレ',
  thunderstorm: '雷雨',
  // Human
  crying_baby: '赤ちゃんの泣き声',
  sneezing: 'くしゃみ',
  clapping: '拍手',
  breathing: '呼吸',
  coughing: '咳',
  footsteps: '足音',
  laughing: '笑い声',
  brushing_teeth: '歯磨き',
  snoring: 'いびき',
  drinking_sipping: '飲む音',
  // Interior
  door_knock: 'ドアノック',
  mouse_click: 'マウスクリック',
  keyboard_typing: 'キーボード',
  door_wood_creaks: 'ドアのきしみ',
  can_opening: '缶を開ける',
  washing_machine: '洗濯機',
  vacuum_cleaner: '掃除機',
  clock_alarm: '目覚まし時計',
  clock_tick: '時計の音',
  glass_breaking: 'ガラスが割れる',
  // Exterior
  helicopter: 'ヘリコプター',
  chainsaw: 'チェーンソー',
  siren: 'サイレン',
  car_horn: 'クラクション',
  engine: 'エンジン',
  train: '電車',
  church_bells: '教会の鐘',
  airplane: '飛行機',
  fireworks: '花火',
  hand_saw: '手ノコギリ',
};

export interface ESC50FilterOptions {
  categories: ESC50CategoryKey[];
  esc10Only: boolean;
  folds: number[];
  selectedClasses: string[];
}

export const defaultFilterOptions: ESC50FilterOptions = {
  categories: ['animals', 'natural', 'human', 'interior', 'exterior'],
  esc10Only: false,
  folds: [1, 2, 3, 4, 5],
  selectedClasses: [],
};

/**
 * CSVをパース
 */
export function parseESC50CSV(csvText: string): ESC50Entry[] {
  const lines = csvText.trim().split('\n');
  const entries: ESC50Entry[] = [];
  
  // ヘッダーをスキップ
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(',');
    if (parts.length >= 7) {
      entries.push({
        filename: parts[0],
        fold: parseInt(parts[1], 10),
        target: parseInt(parts[2], 10),
        category: parts[3],
        esc10: parts[4] === 'True',
        src_file: parts[5],
        take: parts[6],
      });
    }
  }
  
  return entries;
}

/**
 * フィルタオプションに基づいてエントリをフィルタリング
 */
export function filterESC50Entries(
  entries: ESC50Entry[],
  options: ESC50FilterOptions
): ESC50Entry[] {
  return entries.filter((entry) => {
    // ESC-10のみ
    if (options.esc10Only && !entry.esc10) {
      return false;
    }
    
    // fold フィルタ
    if (!options.folds.includes(entry.fold)) {
      return false;
    }
    
    // 特定クラスが選択されている場合
    if (options.selectedClasses.length > 0) {
      if (!options.selectedClasses.includes(entry.category)) {
        return false;
      }
      return true;
    }
    
    // カテゴリフィルタ
    const categoryTargets = options.categories.flatMap(
      (cat) => ESC50_CATEGORIES[cat].targets
    );
    // @ts-expect-error - Type narrowing issue with readonly arrays
    if (!categoryTargets.includes(entry.target)) {
      return false;
    }
    
    return true;
  });
}

/**
 * カテゴリキーからターゲット番号を取得
 */
export function getCategoryFromTarget(target: number): ESC50CategoryKey | null {
  for (const [key, cat] of Object.entries(ESC50_CATEGORIES)) {
    // @ts-expect-error - Type narrowing issue with readonly arrays
    if (cat.targets.includes(target)) {
      return key as ESC50CategoryKey;
    }
  }
  return null;
}


