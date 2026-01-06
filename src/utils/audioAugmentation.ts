/**
 * 音声データ拡張ユーティリティ
 * WAVファイルに対して各種データ拡張を適用する
 */

export interface AugmentationSettings {
  // 時間シフト
  timeShift: {
    enabled: boolean;
    maxShiftMs: number; // 最大シフト量（ミリ秒）
    variations: number; // 生成するバリエーション数
  };
  // ゲイン変化
  gainVariation: {
    enabled: boolean;
    minGainDb: number; // 最小ゲイン（dB）
    maxGainDb: number; // 最大ゲイン（dB）
    variations: number;
  };
  // 環境ノイズ（実音声ファイル）
  environmentNoise: {
    enabled: boolean;
    minSnrDb: number; // 最小SNR（dB）
    maxSnrDb: number; // 最大SNR（dB）
    variations: number;
  };
  // ピッチシフト（慎重に使用）
  pitchShift: {
    enabled: boolean;
    minSemitones: number; // 最小シフト量（半音）
    maxSemitones: number; // 最大シフト量（半音）
    variations: number;
  };
  // タイムストレッチ（慎重に使用）
  timeStretch: {
    enabled: boolean;
    minRate: number; // 最小レート（1.0 = 変化なし）
    maxRate: number; // 最大レート
    variations: number;
  };
}

export const defaultSettings: AugmentationSettings = {
  timeShift: {
    enabled: true,
    maxShiftMs: 50,
    variations: 3,
  },
  gainVariation: {
    enabled: true,
    minGainDb: -2,
    maxGainDb: 2,
    variations: 3,
  },
  environmentNoise: {
    enabled: false,
    minSnrDb: 15,
    maxSnrDb: 30,
    variations: 3,
  },
  pitchShift: {
    enabled: false,
    minSemitones: -0.5,
    maxSemitones: 0.5,
    variations: 2,
  },
  timeStretch: {
    enabled: false,
    minRate: 0.95,
    maxRate: 1.05,
    variations: 2,
  },
};

/**
 * AudioBufferをFloat32Arrayに変換（モノラル化）
 */
export function audioBufferToMono(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels === 1) {
    return buffer.getChannelData(0).slice();
  }
  
  // ステレオの場合は平均を取る
  const left = buffer.getChannelData(0);
  const right = buffer.getChannelData(1);
  const mono = new Float32Array(buffer.length);
  
  for (let i = 0; i < buffer.length; i++) {
    mono[i] = (left[i] + right[i]) / 2;
  }
  
  return mono;
}

/**
 * 時間シフトを適用
 */
export function applyTimeShift(
  samples: Float32Array,
  shiftSamples: number
): Float32Array {
  const result = new Float32Array(samples.length);
  const absShift = Math.abs(shiftSamples);
  
  if (shiftSamples >= 0) {
    // 右にシフト（先頭にゼロを追加）
    result.fill(0, 0, absShift);
    result.set(samples.subarray(0, samples.length - absShift), absShift);
  } else {
    // 左にシフト（末尾にゼロを追加）
    result.set(samples.subarray(absShift));
    result.fill(0, samples.length - absShift);
  }
  
  return result;
}

/**
 * ゲイン変化を適用
 */
export function applyGain(samples: Float32Array, gainDb: number): Float32Array {
  const gain = Math.pow(10, gainDb / 20);
  const result = new Float32Array(samples.length);
  
  for (let i = 0; i < samples.length; i++) {
    result[i] = samples[i] * gain;
  }
  
  return result;
}

/**
 * 環境ノイズ（実音声ファイル）を適用
 * @param samples 元の信号
 * @param noiseSamples ノイズ素材
 * @param snrDb 信号対雑音比（dB）
 */
export function applyEnvironmentNoise(
  samples: Float32Array,
  noiseSamples: Float32Array,
  snrDb: number
): Float32Array {
  // 信号のRMSを計算
  let signalPower = 0;
  for (let i = 0; i < samples.length; i++) {
    signalPower += samples[i] * samples[i];
  }
  signalPower /= samples.length;
  const signalRms = Math.sqrt(signalPower);
  
  // ノイズのRMSを計算
  let noisePower = 0;
  for (let i = 0; i < noiseSamples.length; i++) {
    noisePower += noiseSamples[i] * noiseSamples[i];
  }
  noisePower /= noiseSamples.length;
  const noiseRms = Math.sqrt(noisePower);
  
  // 目標ノイズレベルを計算
  const snrLinear = Math.pow(10, snrDb / 20);
  const targetNoiseRms = signalRms / snrLinear;
  const noiseGain = noiseRms > 0 ? targetNoiseRms / noiseRms : 0;
  
  // ノイズ素材からランダムな開始位置を選択
  const maxNoiseStartIndex = Math.max(0, noiseSamples.length - samples.length);
  const noiseStartIndex = Math.floor(Math.random() * maxNoiseStartIndex);
  
  // 打音側でノイズを入れ始めるタイミングもランダムに（0〜50%の位置から開始）
  const maxSignalOffset = Math.floor(samples.length * 0.5);
  const signalOffset = Math.floor(Math.random() * maxSignalOffset);
  
  // ノイズの長さもランダムに（50%〜100%の範囲）
  const minNoiseLength = Math.floor(samples.length * 0.5);
  const noiseLength = minNoiseLength + Math.floor(Math.random() * (samples.length - signalOffset - minNoiseLength + 1));
  
  // ノイズを適用
  const result = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    result[i] = samples[i]; // まず元の信号をコピー
    
    // ノイズを入れる区間のみ加算
    if (i >= signalOffset && i < signalOffset + noiseLength) {
      const noiseIndex = (noiseStartIndex + (i - signalOffset)) % noiseSamples.length;
      result[i] += noiseSamples[noiseIndex] * noiseGain;
    }
  }
  
  return result;
}

/**
 * 簡易ピッチシフト（リサンプリング方式）
 * 注意: 長さが変わるので、元の長さにトリミング/パディング
 */
export function applyPitchShift(
  samples: Float32Array,
  semitones: number
): Float32Array {
  const ratio = Math.pow(2, semitones / 12);
  const newLength = Math.round(samples.length / ratio);
  const resampled = new Float32Array(newLength);
  
  // 線形補間でリサンプリング
  for (let i = 0; i < newLength; i++) {
    const srcIndex = i * ratio;
    const srcIndexFloor = Math.floor(srcIndex);
    const frac = srcIndex - srcIndexFloor;
    
    if (srcIndexFloor + 1 < samples.length) {
      resampled[i] =
        samples[srcIndexFloor] * (1 - frac) +
        samples[srcIndexFloor + 1] * frac;
    } else if (srcIndexFloor < samples.length) {
      resampled[i] = samples[srcIndexFloor];
    }
  }
  
  // 元の長さに合わせる
  const result = new Float32Array(samples.length);
  if (resampled.length >= samples.length) {
    result.set(resampled.subarray(0, samples.length));
  } else {
    result.set(resampled);
    // 残りはゼロパディング
  }
  
  return result;
}

/**
 * タイムストレッチ（簡易版：PSOLA風ではなく単純リサンプリング）
 */
export function applyTimeStretch(
  samples: Float32Array,
  rate: number
): Float32Array {
  const newLength = Math.round(samples.length / rate);
  const stretched = new Float32Array(newLength);
  
  for (let i = 0; i < newLength; i++) {
    const srcIndex = i * rate;
    const srcIndexFloor = Math.floor(srcIndex);
    const frac = srcIndex - srcIndexFloor;
    
    if (srcIndexFloor + 1 < samples.length) {
      stretched[i] =
        samples[srcIndexFloor] * (1 - frac) +
        samples[srcIndexFloor + 1] * frac;
    } else if (srcIndexFloor < samples.length) {
      stretched[i] = samples[srcIndexFloor];
    }
  }
  
  // 元の長さに合わせる
  const result = new Float32Array(samples.length);
  if (stretched.length >= samples.length) {
    result.set(stretched.subarray(0, samples.length));
  } else {
    result.set(stretched);
  }
  
  return result;
}

/**
 * ランダムな値を範囲内で生成
 */
function randomInRange(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

export interface AugmentedResult {
  name: string;
  samples: Float32Array;
  description: string;
}

/**
 * ノイズ素材の配列
 */
export interface NoiseSample {
  name: string;
  samples: Float32Array;
}

/**
 * 設定に基づいてすべての拡張を適用し、結果の配列を返す
 */
export function generateAugmentations(
  originalSamples: Float32Array,
  sampleRate: number,
  settings: AugmentationSettings,
  originalFileName: string,
  noiseSamples?: NoiseSample[]
): AugmentedResult[] {
  const results: AugmentedResult[] = [];
  const baseName = originalFileName.replace(/\.wav$/i, '');
  
  // オリジナルも含める
  results.push({
    name: `${baseName}_original.wav`,
    samples: originalSamples.slice(),
    description: 'オリジナル',
  });
  
  // 時間シフト
  if (settings.timeShift.enabled) {
    const maxShiftSamples = Math.round(
      (settings.timeShift.maxShiftMs / 1000) * sampleRate
    );
    
    for (let i = 0; i < settings.timeShift.variations; i++) {
      const shiftSamples = Math.round(
        randomInRange(-maxShiftSamples, maxShiftSamples)
      );
      const shiftMs = (shiftSamples / sampleRate) * 1000;
      
      results.push({
        name: `${baseName}_timeshift_${i + 1}.wav`,
        samples: applyTimeShift(originalSamples, shiftSamples),
        description: `時間シフト: ${shiftMs.toFixed(1)}ms`,
      });
    }
  }
  
  // ゲイン変化
  if (settings.gainVariation.enabled) {
    for (let i = 0; i < settings.gainVariation.variations; i++) {
      const gainDb = randomInRange(
        settings.gainVariation.minGainDb,
        settings.gainVariation.maxGainDb
      );
      
      results.push({
        name: `${baseName}_gain_${i + 1}.wav`,
        samples: applyGain(originalSamples, gainDb),
        description: `ゲイン: ${gainDb.toFixed(1)}dB`,
      });
    }
  }
  
  // 環境ノイズ
  if (settings.environmentNoise.enabled && noiseSamples && noiseSamples.length > 0) {
    for (let i = 0; i < settings.environmentNoise.variations; i++) {
      // ランダムにノイズ素材を選択
      const randomNoise = noiseSamples[Math.floor(Math.random() * noiseSamples.length)];
      const snrDb = randomInRange(
        settings.environmentNoise.minSnrDb,
        settings.environmentNoise.maxSnrDb
      );
      
      results.push({
        name: `${baseName}_envnoise_${i + 1}.wav`,
        samples: applyEnvironmentNoise(originalSamples, randomNoise.samples, snrDb),
        description: `環境ノイズ: ${randomNoise.name} (SNR ${snrDb.toFixed(1)}dB)`,
      });
    }
  }
  
  // ピッチシフト
  if (settings.pitchShift.enabled) {
    for (let i = 0; i < settings.pitchShift.variations; i++) {
      const semitones = randomInRange(
        settings.pitchShift.minSemitones,
        settings.pitchShift.maxSemitones
      );
      
      results.push({
        name: `${baseName}_pitch_${i + 1}.wav`,
        samples: applyPitchShift(originalSamples, semitones),
        description: `ピッチシフト: ${semitones.toFixed(2)}半音`,
      });
    }
  }
  
  // タイムストレッチ
  if (settings.timeStretch.enabled) {
    for (let i = 0; i < settings.timeStretch.variations; i++) {
      const rate = randomInRange(
        settings.timeStretch.minRate,
        settings.timeStretch.maxRate
      );
      
      results.push({
        name: `${baseName}_stretch_${i + 1}.wav`,
        samples: applyTimeStretch(originalSamples, rate),
        description: `タイムストレッチ: ${(rate * 100).toFixed(1)}%`,
      });
    }
  }
  
  return results;
}

/**
 * Float32ArrayをWAVファイルのBlobに変換
 */
export function samplesToWavBlob(
  samples: Float32Array,
  sampleRate: number
): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const headerSize = 44;
  const totalSize = headerSize + dataSize;
  
  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  
  // RIFFヘッダー
  writeString(view, 0, 'RIFF');
  view.setUint32(4, totalSize - 8, true);
  writeString(view, 8, 'WAVE');
  
  // fmtチャンク
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // fmtチャンクサイズ
  view.setUint16(20, 1, true); // PCMフォーマット
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  
  // dataチャンク
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);
  
  // サンプルデータ（16bit整数に変換）
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    view.setInt16(offset, intSample, true);
    offset += 2;
  }
  
  return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}
