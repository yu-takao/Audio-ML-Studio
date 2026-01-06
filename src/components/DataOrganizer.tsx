import { useState, useCallback } from 'react';
import {
  FolderOpen,
  FolderOutput,
  Search,
  Trash2,
  FolderSymlink,
  AlertCircle,
  CheckCircle2,
  Loader2,
  X,
  Plus,
  FileAudio,
  AlertTriangle,
} from 'lucide-react';

interface FileEntry {
  name: string;
  path: string;
  handle: FileSystemFileHandle;
  parentHandle: FileSystemDirectoryHandle;
  matches: string[]; // マッチしたキーワード
}

interface ProcessingStatus {
  isProcessing: boolean;
  processed: number;
  total: number;
  currentFile: string;
  action: 'move' | 'delete' | null;
}

export function DataOrganizer() {
  const [inputFolder, setInputFolder] = useState<FileSystemDirectoryHandle | null>(null);
  const [outputFolder, setOutputFolder] = useState<FileSystemDirectoryHandle | null>(null);
  const [keywords, setKeywords] = useState<string[]>(['']);
  const [allFiles, setAllFiles] = useState<FileEntry[]>([]);
  const [matchedFiles, setMatchedFiles] = useState<FileEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<ProcessingStatus>({
    isProcessing: false,
    processed: 0,
    total: 0,
    currentFile: '',
    action: null,
  });
  const [completedAction, setCompletedAction] = useState<{
    action: 'move' | 'delete';
    count: number;
  } | null>(null);

  /**
   * 入力フォルダを選択
   */
  const selectInputFolder = useCallback(async () => {
    try {
      const handle = await window.showDirectoryPicker();
      setInputFolder(handle);
      setIsLoading(true);
      setError(null);
      setCompletedAction(null);

      const files: FileEntry[] = [];

      // 再帰的にファイルを探索
      async function scanDirectory(
        dir: FileSystemDirectoryHandle,
        path: string
      ) {
        for await (const entry of dir.values()) {
          if (entry.kind === 'directory') {
            const subDir = await dir.getDirectoryHandle(entry.name);
            await scanDirectory(subDir, path ? `${path}/${entry.name}` : entry.name);
          } else if (entry.kind === 'file') {
            files.push({
              name: entry.name,
              path: path ? `${path}/${entry.name}` : entry.name,
              handle: entry as FileSystemFileHandle,
              parentHandle: dir,
              matches: [],
            });
          }
        }
      }

      await scanDirectory(handle, '');
      setAllFiles(files);
      setIsLoading(false);

      // キーワードでフィルタリング
      filterFiles(files, keywords);
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setError('フォルダの読み込みに失敗しました: ' + (err as Error).message);
      }
      setIsLoading(false);
    }
  }, [keywords]);

  /**
   * 出力フォルダを選択
   */
  const selectOutputFolder = useCallback(async () => {
    try {
      const handle = await window.showDirectoryPicker({ mode: 'readwrite' });
      setOutputFolder(handle);
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setError('フォルダの選択に失敗しました: ' + (err as Error).message);
      }
    }
  }, []);

  /**
   * キーワードでファイルをフィルタリング
   */
  const filterFiles = useCallback((files: FileEntry[], kws: string[]) => {
    const activeKeywords = kws.filter((kw) => kw.trim() !== '');
    
    if (activeKeywords.length === 0) {
      setMatchedFiles([]);
      return;
    }

    const matched = files
      .map((file) => {
        const matchedKeywords = activeKeywords.filter((kw) =>
          file.name.toLowerCase().includes(kw.toLowerCase())
        );
        return {
          ...file,
          matches: matchedKeywords,
        };
      })
      .filter((file) => file.matches.length > 0);

    setMatchedFiles(matched);
  }, []);

  /**
   * キーワードを更新
   */
  const updateKeyword = (index: number, value: string) => {
    const newKeywords = [...keywords];
    newKeywords[index] = value;
    setKeywords(newKeywords);
    filterFiles(allFiles, newKeywords);
  };

  /**
   * キーワードを追加
   */
  const addKeyword = () => {
    setKeywords([...keywords, '']);
  };

  /**
   * キーワードを削除
   */
  const removeKeyword = (index: number) => {
    if (keywords.length <= 1) return;
    const newKeywords = keywords.filter((_, i) => i !== index);
    setKeywords(newKeywords);
    filterFiles(allFiles, newKeywords);
  };

  /**
   * ファイルを分離（移動）
   */
  const moveFiles = useCallback(async () => {
    if (!outputFolder || matchedFiles.length === 0) return;

    setStatus({
      isProcessing: true,
      processed: 0,
      total: matchedFiles.length,
      currentFile: '',
      action: 'move',
    });
    setCompletedAction(null);

    let processed = 0;

    for (const file of matchedFiles) {
      setStatus((prev) => ({
        ...prev,
        currentFile: file.name,
      }));

      try {
        // ファイルを読み込み
        const fileData = await file.handle.getFile();
        const content = await fileData.arrayBuffer();

        // パス構造を保持して出力先に保存
        const pathParts = file.path.split('/');
        let currentDir = outputFolder;

        for (let i = 0; i < pathParts.length - 1; i++) {
          currentDir = await currentDir.getDirectoryHandle(pathParts[i], { create: true });
        }

        // ファイルを書き込み
        const newFileHandle = await currentDir.getFileHandle(file.name, { create: true });
        const writable = await newFileHandle.createWritable();
        await writable.write(content);
        await writable.close();

        // 元のファイルを削除
        await file.parentHandle.removeEntry(file.name);

        processed++;
        setStatus((prev) => ({
          ...prev,
          processed,
        }));
      } catch (err) {
        console.error(`Failed to move file: ${file.name}`, err);
      }
    }

    setStatus({
      isProcessing: false,
      processed: 0,
      total: 0,
      currentFile: '',
      action: null,
    });

    setCompletedAction({ action: 'move', count: processed });

    // ファイルリストを更新
    const remainingFiles = allFiles.filter(
      (f) => !matchedFiles.some((m) => m.path === f.path)
    );
    setAllFiles(remainingFiles);
    setMatchedFiles([]);
  }, [outputFolder, matchedFiles, allFiles]);

  /**
   * ファイルを完全削除
   */
  const deleteFiles = useCallback(async () => {
    if (matchedFiles.length === 0) return;

    // 確認ダイアログ
    const confirmed = window.confirm(
      `${matchedFiles.length} 件のファイルを完全に削除しますか？\nこの操作は取り消せません。`
    );
    if (!confirmed) return;

    setStatus({
      isProcessing: true,
      processed: 0,
      total: matchedFiles.length,
      currentFile: '',
      action: 'delete',
    });
    setCompletedAction(null);

    let processed = 0;

    for (const file of matchedFiles) {
      setStatus((prev) => ({
        ...prev,
        currentFile: file.name,
      }));

      try {
        await file.parentHandle.removeEntry(file.name);
        processed++;
        setStatus((prev) => ({
          ...prev,
          processed,
        }));
      } catch (err) {
        console.error(`Failed to delete file: ${file.name}`, err);
      }
    }

    setStatus({
      isProcessing: false,
      processed: 0,
      total: 0,
      currentFile: '',
      action: null,
    });

    setCompletedAction({ action: 'delete', count: processed });

    // ファイルリストを更新
    const remainingFiles = allFiles.filter(
      (f) => !matchedFiles.some((m) => m.path === f.path)
    );
    setAllFiles(remainingFiles);
    setMatchedFiles([]);
  }, [matchedFiles, allFiles]);

  return (
    <div className="space-y-6">
      {/* 入力フォルダ選択 */}
      <section className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-violet-500/20">
            <FolderOpen className="w-5 h-5 text-violet-400" />
          </div>
          <h2 className="text-lg font-semibold text-white">対象フォルダ</h2>
        </div>

        <div
          className={`
            rounded-xl border-2 border-dashed p-6 transition-all cursor-pointer
            ${inputFolder
              ? 'border-emerald-500/50 bg-emerald-500/5'
              : 'border-zinc-700 hover:border-violet-500/50 hover:bg-violet-500/5'
            }
          `}
          onClick={selectInputFolder}
        >
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-xl ${inputFolder ? 'bg-emerald-500/20' : 'bg-zinc-800'}`}>
              <FolderOpen className={`w-8 h-8 ${inputFolder ? 'text-emerald-400' : 'text-zinc-400'}`} />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-white">フォルダを選択</h3>
              {inputFolder ? (
                <div className="text-sm text-emerald-400 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4" />
                  {inputFolder.name}
                  <span className="text-zinc-500">({allFiles.length} ファイル)</span>
                </div>
              ) : (
                <p className="text-sm text-zinc-500">クリックしてフォルダを選択</p>
              )}
            </div>
            {isLoading && <Loader2 className="w-5 h-5 text-violet-400 animate-spin" />}
          </div>
        </div>
      </section>

      {/* キーワード設定 */}
      <section className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-cyan-500/20">
            <Search className="w-5 h-5 text-cyan-400" />
          </div>
          <h2 className="text-lg font-semibold text-white">キーワード</h2>
          <span className="text-sm text-zinc-500">
            ファイル名に含まれるキーワードを指定
          </span>
        </div>

        <div className="space-y-2">
          {keywords.map((keyword, index) => (
            <div key={index} className="flex items-center gap-2">
              <input
                type="text"
                value={keyword}
                onChange={(e) => updateKeyword(index, e.target.value)}
                placeholder="キーワードを入力..."
                className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2 text-white placeholder-zinc-500 focus:outline-none focus:border-violet-500"
              />
              {keywords.length > 1 && (
                <button
                  onClick={() => removeKeyword(index)}
                  className="p-2 text-zinc-500 hover:text-red-400 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              )}
            </div>
          ))}
          <button
            onClick={addKeyword}
            className="flex items-center gap-2 px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors"
          >
            <Plus className="w-4 h-4" />
            キーワードを追加
          </button>
        </div>
      </section>

      {/* マッチしたファイル */}
      {allFiles.length > 0 && (
        <section className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-amber-500/20">
              <FileAudio className="w-5 h-5 text-amber-400" />
            </div>
            <h2 className="text-lg font-semibold text-white">マッチしたファイル</h2>
            <span className={`text-sm font-medium ${matchedFiles.length > 0 ? 'text-amber-400' : 'text-zinc-500'}`}>
              {matchedFiles.length} / {allFiles.length} ファイル
            </span>
          </div>

          {matchedFiles.length > 0 ? (
            <div className="max-h-64 overflow-y-auto space-y-1">
              {matchedFiles.slice(0, 100).map((file, index) => (
                <div
                  key={index}
                  className="flex items-center gap-3 px-3 py-2 bg-zinc-900/50 rounded-lg"
                >
                  <FileAudio className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                  <span className="text-sm text-zinc-300 truncate flex-1">{file.path}</span>
                  <div className="flex gap-1">
                    {file.matches.map((kw, i) => (
                      <span
                        key={i}
                        className="px-2 py-0.5 bg-amber-500/20 text-amber-300 rounded text-xs"
                      >
                        {kw}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
              {matchedFiles.length > 100 && (
                <div className="text-sm text-zinc-500 px-3 py-2">
                  ...他 {matchedFiles.length - 100} ファイル
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-zinc-500 text-center py-8">
              {keywords.some((kw) => kw.trim() !== '')
                ? 'マッチするファイルがありません'
                : 'キーワードを入力してください'}
            </div>
          )}
        </section>
      )}

      {/* 出力フォルダ（分離用） */}
      {matchedFiles.length > 0 && (
        <section className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-emerald-500/20">
              <FolderOutput className="w-5 h-5 text-emerald-400" />
            </div>
            <h2 className="text-lg font-semibold text-white">分離先フォルダ（移動用）</h2>
          </div>

          <div
            className={`
              rounded-xl border-2 border-dashed p-4 transition-all cursor-pointer
              ${outputFolder
                ? 'border-emerald-500/50 bg-emerald-500/5'
                : 'border-zinc-700 hover:border-violet-500/50 hover:bg-violet-500/5'
              }
            `}
            onClick={selectOutputFolder}
          >
            <div className="flex items-center gap-3">
              <FolderOutput className={`w-6 h-6 ${outputFolder ? 'text-emerald-400' : 'text-zinc-400'}`} />
              <div>
                <div className="font-medium text-white">
                  {outputFolder ? outputFolder.name : 'クリックして選択'}
                </div>
                {outputFolder && (
                  <div className="text-sm text-emerald-400 flex items-center gap-1">
                    <CheckCircle2 className="w-4 h-4" />
                    選択済み
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* エラー表示 */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/50 rounded-xl p-4 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {/* 進捗表示 */}
      {status.isProcessing && (
        <div className="bg-zinc-800/50 rounded-xl border border-zinc-700 p-6">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-zinc-400">
              {status.action === 'move' ? '移動中' : '削除中'}: {status.currentFile}
            </span>
            <span className="text-white">
              {status.processed} / {status.total}
            </span>
          </div>
          <div className="h-2 bg-zinc-700 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all ${
                status.action === 'delete'
                  ? 'bg-red-500'
                  : 'bg-gradient-to-r from-violet-500 to-fuchsia-500'
              }`}
              style={{ width: `${(status.processed / status.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* 完了メッセージ */}
      {completedAction && (
        <div className={`rounded-xl p-4 flex items-center gap-3 ${
          completedAction.action === 'delete'
            ? 'bg-red-500/10 border border-red-500/50'
            : 'bg-emerald-500/10 border border-emerald-500/50'
        }`}>
          <CheckCircle2 className={`w-6 h-6 ${
            completedAction.action === 'delete' ? 'text-red-400' : 'text-emerald-400'
          }`} />
          <div>
            <div className="text-white font-medium">
              {completedAction.action === 'delete' ? '削除完了' : '移動完了'}
            </div>
            <div className={`text-sm ${
              completedAction.action === 'delete' ? 'text-red-400' : 'text-emerald-400'
            }`}>
              {completedAction.count} ファイルを
              {completedAction.action === 'delete' ? '削除' : '移動'}しました
            </div>
          </div>
        </div>
      )}

      {/* アクションボタン */}
      {matchedFiles.length > 0 && !status.isProcessing && (
        <section className="flex flex-wrap gap-4 justify-center">
          {/* 分離（移動） */}
          <button
            onClick={moveFiles}
            disabled={!outputFolder}
            className={`
              flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all
              ${
                !outputFolder
                  ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white hover:from-violet-600 hover:to-fuchsia-600 shadow-lg'
              }
            `}
          >
            <FolderSymlink className="w-5 h-5" />
            分離（{matchedFiles.length} ファイルを移動）
          </button>

          {/* 完全削除 */}
          <button
            onClick={deleteFiles}
            className="flex items-center gap-2 px-6 py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/50 rounded-xl font-semibold transition-all"
          >
            <Trash2 className="w-5 h-5" />
            完全削除
          </button>
        </section>
      )}

      {/* 警告 */}
      {matchedFiles.length > 0 && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-amber-300">
            <strong>注意:</strong> 分離・削除操作は元に戻せません。
            重要なデータは事前にバックアップしてください。
          </div>
        </div>
      )}
    </div>
  );
}





