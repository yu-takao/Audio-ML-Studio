/**
 * スペクトログラム生成ユーティリティ
 * 音声データをCNNに入力するための2D画像（スペクトログラム）に変換します
 */

// FFTサイズ（周波数解像度）
const FFT_SIZE = 512;
// ホップサイズ（時間解像度）
const HOP_SIZE = 256;
// メルフィルタバンク数
const N_MELS = 128;
// 最小周波数
const F_MIN = 0;
// 最大周波数（サンプルレートの半分まで）
const F_MAX = 8000;

/**
 * メルスケールに変換
 */
function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

/**
 * メルスケールからHzに変換
 */
function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

/**
 * メルフィルタバンクを生成
 */
function createMelFilterbank(
  numMels: number,
  fftSize: number,
  sampleRate: number,
  fMin: number,
  fMax: number
): number[][] {
  const numBins = Math.floor(fftSize / 2) + 1;
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(Math.min(fMax, sampleRate / 2));
  
  // メル周波数の等間隔ポイント
  const melPoints: number[] = [];
  for (let i = 0; i <= numMels + 1; i++) {
    melPoints.push(melMin + (melMax - melMin) * i / (numMels + 1));
  }
  
  // Hzに変換してFFTビンインデックスに変換
  const binPoints = melPoints.map((mel) => {
    const hz = melToHz(mel);
    return Math.floor((fftSize + 1) * hz / sampleRate);
  });
  
  // フィルタバンク行列を作成
  const filterbank: number[][] = [];
  for (let i = 0; i < numMels; i++) {
    const filter: number[] = new Array(numBins).fill(0);
    const start = binPoints[i];
    const center = binPoints[i + 1];
    const end = binPoints[i + 2];
    
    // 上り勾配
    for (let j = start; j < center && j < numBins; j++) {
      if (center !== start) {
        filter[j] = (j - start) / (center - start);
      }
    }
    
    // 下り勾配
    for (let j = center; j < end && j < numBins; j++) {
      if (end !== center) {
        filter[j] = (end - j) / (end - center);
      }
    }
    
    filterbank.push(filter);
  }
  
  return filterbank;
}

/**
 * ハン窓を生成
 */
function hanningWindow(size: number): Float32Array {
  const window = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return window;
}

/**
 * FFTを計算（Cooley-Tukey アルゴリズム）
 */
function fft(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  
  if (n <= 1) return;
  
  // ビット反転並び替え
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
    let k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }
  
  // FFT計算
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (2 * Math.PI) / len;
    const wReal = Math.cos(angle);
    const wImag = -Math.sin(angle);
    
    for (let i = 0; i < n; i += len) {
      let curReal = 1;
      let curImag = 0;
      
      for (let k = 0; k < halfLen; k++) {
        const idx1 = i + k;
        const idx2 = i + k + halfLen;
        
        const tReal = curReal * real[idx2] - curImag * imag[idx2];
        const tImag = curReal * imag[idx2] + curImag * real[idx2];
        
        real[idx2] = real[idx1] - tReal;
        imag[idx2] = imag[idx1] - tImag;
        real[idx1] = real[idx1] + tReal;
        imag[idx1] = imag[idx1] + tImag;
        
        const nextReal = curReal * wReal - curImag * wImag;
        const nextImag = curReal * wImag + curImag * wReal;
        curReal = nextReal;
        curImag = nextImag;
      }
    }
  }
}

/**
 * パワースペクトルを計算
 */
function computePowerSpectrum(
  samples: Float32Array,
  fftSize: number,
  window: Float32Array
): Float32Array {
  const numBins = Math.floor(fftSize / 2) + 1;
  const real = new Float32Array(fftSize);
  const imag = new Float32Array(fftSize);
  
  // 窓関数を適用
  for (let i = 0; i < fftSize && i < samples.length; i++) {
    real[i] = samples[i] * window[i];
  }
  
  fft(real, imag);
  
  // パワースペクトルを計算
  const power = new Float32Array(numBins);
  for (let i = 0; i < numBins; i++) {
    power[i] = real[i] * real[i] + imag[i] * imag[i];
  }
  
  return power;
}

export interface SpectrogramOptions {
  fftSize?: number;
  hopSize?: number;
  nMels?: number;
  fMin?: number;
  fMax?: number;
  normalize?: boolean;
}

export interface SpectrogramResult {
  data: Float32Array[];
  width: number;
  height: number;
  timeSteps: number;
  freqBins: number;
}

/**
 * メルスペクトログラムを生成
 */
export function generateMelSpectrogram(
  samples: Float32Array,
  sampleRate: number,
  options: SpectrogramOptions = {}
): SpectrogramResult {
  const fftSize = options.fftSize ?? FFT_SIZE;
  const hopSize = options.hopSize ?? HOP_SIZE;
  const nMels = options.nMels ?? N_MELS;
  const fMin = options.fMin ?? F_MIN;
  const fMax = options.fMax ?? F_MAX;
  const normalize = options.normalize ?? true;
  
  // FFTサイズが2の累乗でなければ調整
  const actualFftSize = Math.pow(2, Math.ceil(Math.log2(fftSize)));
  
  const window = hanningWindow(actualFftSize);
  const filterbank = createMelFilterbank(nMels, actualFftSize, sampleRate, fMin, fMax);
  
  // フレーム数を計算
  const numFrames = Math.max(1, Math.floor((samples.length - actualFftSize) / hopSize) + 1);
  
  const melSpectrogram: Float32Array[] = [];
  
  for (let frame = 0; frame < numFrames; frame++) {
    const start = frame * hopSize;
    const frameData = new Float32Array(actualFftSize);
    
    for (let i = 0; i < actualFftSize && start + i < samples.length; i++) {
      frameData[i] = samples[start + i];
    }
    
    const powerSpec = computePowerSpectrum(frameData, actualFftSize, window);
    
    // メルフィルタを適用
    const melFrame = new Float32Array(nMels);
    for (let i = 0; i < nMels; i++) {
      let sum = 0;
      for (let j = 0; j < powerSpec.length; j++) {
        sum += filterbank[i][j] * powerSpec[j];
      }
      // 対数スケールに変換（dB）
      melFrame[i] = Math.log10(Math.max(sum, 1e-10));
    }
    
    melSpectrogram.push(melFrame);
  }
  
  // 正規化
  if (normalize && melSpectrogram.length > 0) {
    let minVal = Infinity;
    let maxVal = -Infinity;
    
    for (const frame of melSpectrogram) {
      for (const val of frame) {
        minVal = Math.min(minVal, val);
        maxVal = Math.max(maxVal, val);
      }
    }
    
    const range = maxVal - minVal || 1;
    for (const frame of melSpectrogram) {
      for (let i = 0; i < frame.length; i++) {
        frame[i] = (frame[i] - minVal) / range;
      }
    }
  }
  
  return {
    data: melSpectrogram,
    width: numFrames,
    height: nMels,
    timeSteps: numFrames,
    freqBins: nMels,
  };
}

/**
 * スペクトログラムを固定サイズにリサイズ
 */
export function resizeSpectrogram(
  spectrogram: SpectrogramResult,
  targetWidth: number,
  targetHeight: number
): Float32Array {
  const result = new Float32Array(targetWidth * targetHeight);
  
  const scaleX = spectrogram.width / targetWidth;
  const scaleY = spectrogram.height / targetHeight;
  
  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const srcX = Math.min(Math.floor(x * scaleX), spectrogram.width - 1);
      const srcY = Math.min(Math.floor(y * scaleY), spectrogram.height - 1);
      
      // スペクトログラムは [時間][周波数] の順
      if (srcX < spectrogram.data.length && srcY < spectrogram.data[srcX].length) {
        result[y * targetWidth + x] = spectrogram.data[srcX][srcY];
      }
    }
  }
  
  return result;
}

/**
 * スペクトログラムをCanvasに描画
 */
export function drawSpectrogramToCanvas(
  spectrogram: SpectrogramResult,
  canvas: HTMLCanvasElement
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  canvas.width = spectrogram.width;
  canvas.height = spectrogram.height;
  
  const imageData = ctx.createImageData(spectrogram.width, spectrogram.height);
  
  for (let x = 0; x < spectrogram.width; x++) {
    for (let y = 0; y < spectrogram.height; y++) {
      const value = spectrogram.data[x]?.[spectrogram.height - 1 - y] ?? 0;
      const pixelIndex = (y * spectrogram.width + x) * 4;
      
      // Viridisカラーマップ風の色付け
      const r = Math.floor(value * 255 * (0.267004 + value * 0.329415));
      const g = Math.floor(value * 255 * (0.004874 + value * 0.815373));
      const b = Math.floor(255 * (0.329415 + value * (0.452525 - value * 0.781993)));
      
      imageData.data[pixelIndex] = Math.min(255, r);
      imageData.data[pixelIndex + 1] = Math.min(255, g);
      imageData.data[pixelIndex + 2] = Math.min(255, Math.max(0, b));
      imageData.data[pixelIndex + 3] = 255;
    }
  }
  
  ctx.putImageData(imageData, 0, 0);
}

