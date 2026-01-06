/**
 * ESC-50 ノイズファイルを管理するフック
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import type { NoiseSample } from '../utils/audioAugmentation';
import { audioBufferToMono } from '../utils/audioAugmentation';
import type { ESC50Entry, ESC50FilterOptions } from '../utils/esc50';
import {
  defaultFilterOptions,
  parseESC50CSV,
  filterESC50Entries,
} from '../utils/esc50';

export interface UseNoiseFilesResult {
  // データ
  allEntries: ESC50Entry[];
  filteredEntries: ESC50Entry[];
  noiseSamples: NoiseSample[];
  
  // フィルタ
  filterOptions: ESC50FilterOptions;
  setFilterOptions: (options: ESC50FilterOptions) => void;
  
  // 状態
  isLoading: boolean;
  isReady: boolean;
  error: string | null;
  
  // メソッド
  loadRandomSamples: (count: number) => Promise<NoiseSample[]>;
}

/**
 * WAVファイルをAudioBufferに変換
 */
async function fetchAudioBuffer(url: string): Promise<AudioBuffer> {
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  const audioContext = new AudioContext();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  await audioContext.close();
  return audioBuffer;
}

export function useNoiseFiles(): UseNoiseFilesResult {
  const [allEntries, setAllEntries] = useState<ESC50Entry[]>([]);
  const [filterOptions, setFilterOptions] = useState<ESC50FilterOptions>(defaultFilterOptions);
  const [noiseSamples, setNoiseSamples] = useState<NoiseSample[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 初期化時にESC-50メタデータを読み込む
  useEffect(() => {
    async function loadESC50Metadata() {
      try {
        const response = await fetch('/esc50.csv');
        if (!response.ok) {
          throw new Error('ESC-50メタデータの読み込みに失敗しました');
        }
        const csvText = await response.text();
        const entries = parseESC50CSV(csvText);
        setAllEntries(entries);
        setIsLoading(false);
      } catch (err) {
        setError((err as Error).message);
        setIsLoading(false);
      }
    }

    loadESC50Metadata();
  }, []);

  // フィルタリングされたエントリ
  const filteredEntries = useMemo(() => {
    return filterESC50Entries(allEntries, filterOptions);
  }, [allEntries, filterOptions]);

  /**
   * ランダムにノイズサンプルを読み込む
   * 処理時に呼び出して、必要な分だけ読み込む
   */
  const loadRandomSamples = useCallback(async (count: number): Promise<NoiseSample[]> => {
    if (filteredEntries.length === 0) {
      return [];
    }

    // ランダムにファイルを選択
    const shuffled = [...filteredEntries].sort(() => Math.random() - 0.5);
    const selectedEntries = shuffled.slice(0, Math.min(count, filteredEntries.length));

    const samples: NoiseSample[] = [];
    for (const entry of selectedEntries) {
      try {
        const audioBuffer = await fetchAudioBuffer(`/noise/${entry.filename}`);
        const mono = audioBufferToMono(audioBuffer);
        samples.push({
          name: entry.filename,
          samples: mono,
        });
      } catch (err) {
        console.warn(`Failed to load noise file: ${entry.filename}`, err);
      }
    }

    setNoiseSamples(samples);
    return samples;
  }, [filteredEntries]);

  return {
    allEntries,
    filteredEntries,
    noiseSamples,
    filterOptions,
    setFilterOptions,
    isLoading,
    isReady: filteredEntries.length > 0,
    error,
    loadRandomSamples,
  };
}
