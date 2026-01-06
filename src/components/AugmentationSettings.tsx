import type { AugmentationSettings as Settings } from '../utils/audioAugmentation';
import { Settings as SettingsIcon, Clock, Volume2, FileAudio2, Music, Timer, CheckCircle2 } from 'lucide-react';

interface Props {
  settings: Settings;
  onChange: (settings: Settings) => void;
  noiseFileCount?: number;
  isNoiseReady?: boolean;
}

interface SettingCardProps {
  title: string;
  icon: React.ReactNode;
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  children: React.ReactNode;
  color: string;
  warning?: boolean;
  disabled?: boolean;
  disabledMessage?: string;
}

function SettingCard({ title, icon, enabled, onToggle, children, color, warning, disabled, disabledMessage }: SettingCardProps) {
  return (
    <div
      className={`
        rounded-xl border-2 transition-all duration-300
        ${enabled && !disabled
          ? `${color} shadow-lg`
          : 'border-zinc-700 bg-zinc-800/50 opacity-60'
        }
      `}
    >
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={`p-2 rounded-lg ${enabled && !disabled ? 'bg-white/10' : 'bg-zinc-700'}`}>
              {icon}
            </div>
            <span className="font-semibold text-white">{title}</span>
            {warning && (
              <span className="text-xs px-2 py-0.5 rounded bg-amber-500/20 text-amber-400">
                慎重に
              </span>
            )}
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => onToggle(e.target.checked)}
              disabled={disabled}
              className="sr-only peer"
            />
            <div className={`w-11 h-6 bg-zinc-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-500 ${disabled ? 'cursor-not-allowed opacity-50' : ''}`}></div>
          </label>
        </div>
        
        {disabled && disabledMessage && (
          <p className="text-xs text-amber-400 mb-2">{disabledMessage}</p>
        )}
        
        <div className={`space-y-3 transition-opacity ${enabled && !disabled ? 'opacity-100' : 'opacity-40'}`}>
          {children}
        </div>
      </div>
    </div>
  );
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (value: number) => void;
  disabled?: boolean;
}

function Slider({ label, value, min, max, step, unit, onChange, disabled }: SliderProps) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-zinc-400">{label}</span>
        <span className="text-white font-mono">{value}{unit}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        className="w-full h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-current"
      />
    </div>
  );
}

export function AugmentationSettingsPanel({ 
  settings, 
  onChange, 
  noiseFileCount = 0,
  isNoiseReady = false,
}: Props) {
  const updateTimeShift = (updates: Partial<Settings['timeShift']>) => {
    onChange({
      ...settings,
      timeShift: { ...settings.timeShift, ...updates },
    });
  };

  const updateGainVariation = (updates: Partial<Settings['gainVariation']>) => {
    onChange({
      ...settings,
      gainVariation: { ...settings.gainVariation, ...updates },
    });
  };

  const updateEnvironmentNoise = (updates: Partial<Settings['environmentNoise']>) => {
    onChange({
      ...settings,
      environmentNoise: { ...settings.environmentNoise, ...updates },
    });
  };

  const updatePitchShift = (updates: Partial<Settings['pitchShift']>) => {
    onChange({
      ...settings,
      pitchShift: { ...settings.pitchShift, ...updates },
    });
  };

  const updateTimeStretch = (updates: Partial<Settings['timeStretch']>) => {
    onChange({
      ...settings,
      timeStretch: { ...settings.timeStretch, ...updates },
    });
  };

  // 生成されるファイル数を計算
  const calculateTotalVariations = () => {
    let total = 1; // オリジナル
    if (settings.timeShift.enabled) total += settings.timeShift.variations;
    if (settings.gainVariation.enabled) total += settings.gainVariation.variations;
    if (settings.environmentNoise.enabled && isNoiseReady) total += settings.environmentNoise.variations;
    if (settings.pitchShift.enabled) total += settings.pitchShift.variations;
    if (settings.timeStretch.enabled) total += settings.timeStretch.variations;
    return total;
  };

  const noiseDisabled = !isNoiseReady;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <SettingsIcon className="w-5 h-5 text-violet-400" />
        <h2 className="text-lg font-bold text-white">データ拡張設定</h2>
        <span className="ml-auto text-sm text-zinc-400">
          1ファイルあたり <span className="text-violet-400 font-bold">{calculateTotalVariations()}</span> 個生成
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* 時間シフト */}
        <SettingCard
          title="時間シフト"
          icon={<Clock className="w-5 h-5 text-blue-400" />}
          enabled={settings.timeShift.enabled}
          onToggle={(enabled) => updateTimeShift({ enabled })}
          color="border-blue-500/50 bg-blue-500/10"
        >
          <Slider
            label="最大シフト量"
            value={settings.timeShift.maxShiftMs}
            min={10}
            max={200}
            step={10}
            unit="ms"
            onChange={(maxShiftMs) => updateTimeShift({ maxShiftMs })}
            disabled={!settings.timeShift.enabled}
          />
          <Slider
            label="バリエーション数"
            value={settings.timeShift.variations}
            min={1}
            max={10}
            step={1}
            unit="個"
            onChange={(variations) => updateTimeShift({ variations })}
            disabled={!settings.timeShift.enabled}
          />
        </SettingCard>

        {/* ゲイン変化 */}
        <SettingCard
          title="ゲイン変化"
          icon={<Volume2 className="w-5 h-5 text-emerald-400" />}
          enabled={settings.gainVariation.enabled}
          onToggle={(enabled) => updateGainVariation({ enabled })}
          color="border-emerald-500/50 bg-emerald-500/10"
        >
          <Slider
            label="最小ゲイン"
            value={settings.gainVariation.minGainDb}
            min={-12}
            max={0}
            step={0.5}
            unit="dB"
            onChange={(minGainDb) => updateGainVariation({ minGainDb })}
            disabled={!settings.gainVariation.enabled}
          />
          <Slider
            label="最大ゲイン"
            value={settings.gainVariation.maxGainDb}
            min={0}
            max={12}
            step={0.5}
            unit="dB"
            onChange={(maxGainDb) => updateGainVariation({ maxGainDb })}
            disabled={!settings.gainVariation.enabled}
          />
          <Slider
            label="バリエーション数"
            value={settings.gainVariation.variations}
            min={1}
            max={10}
            step={1}
            unit="個"
            onChange={(variations) => updateGainVariation({ variations })}
            disabled={!settings.gainVariation.enabled}
          />
        </SettingCard>

        {/* 環境ノイズ */}
        <SettingCard
          title="環境ノイズ"
          icon={<FileAudio2 className="w-5 h-5 text-orange-400" />}
          enabled={settings.environmentNoise.enabled}
          onToggle={(enabled) => updateEnvironmentNoise({ enabled })}
          color="border-orange-500/50 bg-orange-500/10"
          disabled={noiseDisabled}
          disabledMessage="ノイズファイルを読み込み中..."
        >
          {/* ESC-50情報 */}
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-700/50 text-sm">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span className="text-zinc-300">ESC-50: {noiseFileCount}種類の環境音</span>
          </div>
          
          <Slider
            label="最小SNR"
            value={settings.environmentNoise.minSnrDb}
            min={5}
            max={30}
            step={1}
            unit="dB"
            onChange={(minSnrDb) => updateEnvironmentNoise({ minSnrDb })}
            disabled={!settings.environmentNoise.enabled || noiseDisabled}
          />
          <Slider
            label="最大SNR"
            value={settings.environmentNoise.maxSnrDb}
            min={10}
            max={50}
            step={1}
            unit="dB"
            onChange={(maxSnrDb) => updateEnvironmentNoise({ maxSnrDb })}
            disabled={!settings.environmentNoise.enabled || noiseDisabled}
          />
          <Slider
            label="バリエーション数"
            value={settings.environmentNoise.variations}
            min={1}
            max={10}
            step={1}
            unit="個"
            onChange={(variations) => updateEnvironmentNoise({ variations })}
            disabled={!settings.environmentNoise.enabled || noiseDisabled}
          />
        </SettingCard>

        {/* ピッチシフト */}
        <SettingCard
          title="ピッチシフト"
          icon={<Music className="w-5 h-5 text-pink-400" />}
          enabled={settings.pitchShift.enabled}
          onToggle={(enabled) => updatePitchShift({ enabled })}
          color="border-pink-500/50 bg-pink-500/10"
          warning
        >
          <Slider
            label="最小シフト"
            value={settings.pitchShift.minSemitones}
            min={-2}
            max={0}
            step={0.1}
            unit="半音"
            onChange={(minSemitones) => updatePitchShift({ minSemitones })}
            disabled={!settings.pitchShift.enabled}
          />
          <Slider
            label="最大シフト"
            value={settings.pitchShift.maxSemitones}
            min={0}
            max={2}
            step={0.1}
            unit="半音"
            onChange={(maxSemitones) => updatePitchShift({ maxSemitones })}
            disabled={!settings.pitchShift.enabled}
          />
          <Slider
            label="バリエーション数"
            value={settings.pitchShift.variations}
            min={1}
            max={5}
            step={1}
            unit="個"
            onChange={(variations) => updatePitchShift({ variations })}
            disabled={!settings.pitchShift.enabled}
          />
        </SettingCard>

        {/* タイムストレッチ */}
        <SettingCard
          title="タイムストレッチ"
          icon={<Timer className="w-5 h-5 text-cyan-400" />}
          enabled={settings.timeStretch.enabled}
          onToggle={(enabled) => updateTimeStretch({ enabled })}
          color="border-cyan-500/50 bg-cyan-500/10"
          warning
        >
          <Slider
            label="最小レート"
            value={settings.timeStretch.minRate * 100}
            min={80}
            max={100}
            step={1}
            unit="%"
            onChange={(v) => updateTimeStretch({ minRate: v / 100 })}
            disabled={!settings.timeStretch.enabled}
          />
          <Slider
            label="最大レート"
            value={settings.timeStretch.maxRate * 100}
            min={100}
            max={120}
            step={1}
            unit="%"
            onChange={(v) => updateTimeStretch({ maxRate: v / 100 })}
            disabled={!settings.timeStretch.enabled}
          />
          <Slider
            label="バリエーション数"
            value={settings.timeStretch.variations}
            min={1}
            max={5}
            step={1}
            unit="個"
            onChange={(variations) => updateTimeStretch({ variations })}
            disabled={!settings.timeStretch.enabled}
          />
        </SettingCard>
      </div>
    </div>
  );
}
