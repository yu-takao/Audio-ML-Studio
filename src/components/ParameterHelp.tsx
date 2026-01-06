import { useState } from 'react';
import { HelpCircle, X } from 'lucide-react';

interface ParameterHelpProps {
  title: string;
  description: string;
  recommendations: string[];
  defaultValue?: string;
  warningLow?: string;
  warningHigh?: string;
}

export function ParameterHelp({
  title,
  description,
  recommendations,
  defaultValue,
  warningLow,
  warningHigh,
}: ParameterHelpProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="p-1 text-zinc-500 hover:text-violet-400 transition-colors"
        title="パラメータの説明を表示"
      >
        <HelpCircle className="w-4 h-4" />
      </button>

      {/* モーダル */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* オーバーレイ */}
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setIsOpen(false)}
          />

          {/* モーダルコンテンツ */}
          <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl max-w-md w-full p-6 shadow-2xl">
            {/* 閉じるボタン */}
            <button
              onClick={() => setIsOpen(false)}
              className="absolute top-4 right-4 p-1 text-zinc-500 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            {/* タイトル */}
            <div className="flex items-center gap-2 mb-4">
              <div className="p-2 rounded-lg bg-violet-500/20">
                <HelpCircle className="w-5 h-5 text-violet-400" />
              </div>
              <h3 className="text-lg font-semibold text-white">{title}</h3>
            </div>

            {/* 説明 */}
            <div className="mb-4">
              <p className="text-zinc-300 text-sm leading-relaxed">{description}</p>
            </div>

            {/* デフォルト値 */}
            {defaultValue && (
              <div className="mb-4 p-3 bg-zinc-800 rounded-lg">
                <div className="text-xs text-zinc-500 mb-1">推奨デフォルト値</div>
                <div className="text-white font-medium">{defaultValue}</div>
              </div>
            )}

            {/* 推奨事項 */}
            <div className="mb-4">
              <div className="text-sm text-zinc-400 mb-2">調整のポイント</div>
              <ul className="space-y-2">
                {recommendations.map((rec, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <span className="text-violet-400 mt-1">•</span>
                    <span className="text-zinc-300">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* 警告 */}
            {(warningLow || warningHigh) && (
              <div className="space-y-2">
                {warningLow && (
                  <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                    <div className="text-xs text-amber-400 mb-1">⚠️ 小さすぎると...</div>
                    <div className="text-sm text-amber-300">{warningLow}</div>
                  </div>
                )}
                {warningHigh && (
                  <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                    <div className="text-xs text-amber-400 mb-1">⚠️ 大きすぎると...</div>
                    <div className="text-sm text-amber-300">{warningHigh}</div>
                  </div>
                )}
              </div>
            )}

            {/* 閉じるボタン */}
            <button
              onClick={() => setIsOpen(false)}
              className="w-full mt-4 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-white rounded-lg transition-colors"
            >
              閉じる
            </button>
          </div>
        </div>
      )}
    </>
  );
}

// パラメータ説明のプリセット
export const PARAM_HELP = {
  epochs: {
    title: 'エポック数',
    description: 'データセット全体を何回繰り返し学習するかを決めるパラメータです。1エポック = 全データを1回学習したことを意味します。',
    defaultValue: '50',
    recommendations: [
      'まずは50エポックで試してみる',
      '検証精度が上がり続けているなら増やす',
      '検証精度が途中から下がり始めたら減らす（過学習のサイン）',
      'データ量が少ない場合は少なめに設定',
    ],
    warningLow: '学習が不十分で精度が低くなる可能性があります',
    warningHigh: '過学習（訓練データに特化しすぎて汎用性が下がる）のリスクがあります',
  },
  batchSize: {
    title: 'バッチサイズ',
    description: '一度にモデルに入力するデータの数です。メモリ使用量と学習の安定性に影響します。',
    defaultValue: '32',
    recommendations: [
      '32が一般的なスタート値',
      'メモリ不足エラーが出たら小さくする（16や8）',
      'データ量が多い場合は大きくしても良い（64や128）',
      '小さいと学習が不安定だが細かく調整される',
      '大きいと学習が安定するが汎化性能が下がることも',
    ],
    warningLow: '学習が不安定になり、精度がばらつく可能性があります',
    warningHigh: 'メモリ不足になる可能性があります。また、汎化性能が下がることがあります',
  },
  learningRate: {
    title: '学習率',
    description: 'モデルが一度の更新でどれくらい大きく変化するかを決めるパラメータです。最も重要なハイパーパラメータの1つです。',
    defaultValue: '0.001',
    recommendations: [
      '0.001（1e-3）が一般的なスタート値',
      '学習が進まない場合は大きくする（0.01）',
      '学習が不安定（精度が乱高下）なら小さくする（0.0001）',
      '途中で学習率を下げる「学習率スケジューリング」も有効',
    ],
    warningLow: '学習が非常に遅くなり、良い解に到達しない可能性があります',
    warningHigh: '学習が不安定になり、精度が収束しない（乱高下する）可能性があります',
  },
  validationSplit: {
    title: '検証データ割合',
    description: '訓練中にモデルの性能を確認するためのデータの割合です。訓練には使わず、各エポック終了後の精度確認に使います。',
    defaultValue: '20%',
    recommendations: [
      '15-20%が一般的',
      'データ量が少ない場合は10%程度に抑える',
      '過学習の監視に必要なので、0にはしない',
      'クロスバリデーションを使う場合は不要なこともある',
    ],
    warningLow: '過学習を検出しにくくなります',
    warningHigh: '訓練に使えるデータが減り、モデルの性能が下がる可能性があります',
  },
  testSplit: {
    title: 'テストデータ割合',
    description: '訓練完了後にモデルの最終的な性能を評価するためのデータの割合です。訓練中には一切使用しません。',
    defaultValue: '15%',
    recommendations: [
      '10-20%が一般的',
      'データ量が十分にある場合は15-20%',
      'データ量が少ない場合は10%程度に抑える',
      '本番環境での性能を予測するための重要なデータ',
    ],
    warningLow: '評価の信頼性が下がります。特に少数クラスの評価が不正確になります',
    warningHigh: '訓練に使えるデータが減り、モデルの性能が下がる可能性があります',
  },
};



