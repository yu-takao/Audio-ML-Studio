/**
 * File System Access API を使用したフォルダ/ファイル操作フック
 */

import { useState, useCallback } from 'react';

export interface WavFile {
  name: string;
  handle: FileSystemFileHandle;
  file: File;
}

export interface UseFileSystemResult {
  inputFolder: FileSystemDirectoryHandle | null;
  outputFolder: FileSystemDirectoryHandle | null;
  wavFiles: WavFile[];
  isLoading: boolean;
  error: string | null;
  selectInputFolder: () => Promise<void>;
  selectOutputFolder: () => Promise<void>;
  isSupported: boolean;
}

export function useFileSystem(): UseFileSystemResult {
  const [inputFolder, setInputFolder] = useState<FileSystemDirectoryHandle | null>(null);
  const [outputFolder, setOutputFolder] = useState<FileSystemDirectoryHandle | null>(null);
  const [wavFiles, setWavFiles] = useState<WavFile[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // File System Access APIがサポートされているか
  const isSupported = 'showDirectoryPicker' in window;

  /**
   * フォルダ内のWAVファイルを再帰的に取得
   */
  const getWavFilesFromDirectory = async (
    dirHandle: FileSystemDirectoryHandle,
    path: string = ''
  ): Promise<WavFile[]> => {
    const files: WavFile[] = [];

    for await (const entry of dirHandle.values()) {
      if (entry.kind === 'file' && entry.name.toLowerCase().endsWith('.wav')) {
        // deletedフォルダは除外
        if (!path.includes('deleted')) {
          const file = await entry.getFile();
          files.push({
            name: path ? `${path}/${entry.name}` : entry.name,
            handle: entry,
            file,
          });
        }
      } else if (entry.kind === 'directory') {
        // サブフォルダも再帰的に探索（deletedフォルダは除外）
        if (entry.name !== 'deleted') {
          const subPath = path ? `${path}/${entry.name}` : entry.name;
          const subFiles = await getWavFilesFromDirectory(entry, subPath);
          files.push(...subFiles);
        }
      }
    }

    return files;
  };

  /**
   * 入力フォルダを選択
   */
  const selectInputFolder = useCallback(async () => {
    if (!isSupported) {
      setError('このブラウザはFile System Access APIをサポートしていません。Chrome/Edgeをお使いください。');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      const handle = await window.showDirectoryPicker({
        mode: 'read',
      });

      setInputFolder(handle);

      // WAVファイルを取得
      const files = await getWavFilesFromDirectory(handle);
      setWavFiles(files);

      if (files.length === 0) {
        setError('選択したフォルダにWAVファイルが見つかりませんでした。');
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setError(`フォルダの選択に失敗しました: ${(err as Error).message}`);
      }
    } finally {
      setIsLoading(false);
    }
  }, [isSupported]);

  /**
   * 出力フォルダを選択
   */
  const selectOutputFolder = useCallback(async () => {
    if (!isSupported) {
      setError('このブラウザはFile System Access APIをサポートしていません。Chrome/Edgeをお使いください。');
      return;
    }

    try {
      setError(null);

      const handle = await window.showDirectoryPicker({
        mode: 'readwrite',
      });

      setOutputFolder(handle);
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setError(`フォルダの選択に失敗しました: ${(err as Error).message}`);
      }
    }
  }, [isSupported]);

  return {
    inputFolder,
    outputFolder,
    wavFiles,
    isLoading,
    error,
    selectInputFolder,
    selectOutputFolder,
    isSupported,
  };
}

// File System Access API の型定義
declare global {
  interface Window {
    showDirectoryPicker: (options?: {
      mode?: 'read' | 'readwrite';
    }) => Promise<FileSystemDirectoryHandle>;
  }

  interface FileSystemDirectoryHandle {
    values(): AsyncIterableIterator<FileSystemHandle>;
    getDirectoryHandle(
      name: string,
      options?: { create?: boolean }
    ): Promise<FileSystemDirectoryHandle>;
    getFileHandle(
      name: string,
      options?: { create?: boolean }
    ): Promise<FileSystemFileHandle>;
  }

  interface FileSystemFileHandle {
    createWritable(): Promise<FileSystemWritableFileStream>;
  }

  interface FileSystemWritableFileStream extends WritableStream {
    write(data: BufferSource | Blob | string): Promise<void>;
    close(): Promise<void>;
  }
}

