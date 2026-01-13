#!/usr/bin/env python3
"""
SageMaker Processing Script for Model Evaluation
音声分類モデルの評価を行い、精度指標と混同行列を出力

使用方法:
- S3にアップロードして SageMaker Processing Job で実行
- TFJSモデルを読み込み、テストデータで評価

使用イメージ: tensorflow-training:2.13.0-cpu-py310-ubuntu20.04-sagemaker
（TensorFlow, numpy, scipy, scikit-learn がプリインストール済み）
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple
import csv

print("=" * 60)
print("Environment initialized successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print("=" * 60)

# SageMaker環境変数
INPUT_DATA_DIR = '/opt/ml/processing/input/data'
INPUT_MODEL_DIR = '/opt/ml/processing/input/model'
OUTPUT_DIR = '/opt/ml/processing/output'

# 環境変数から設定を取得
BUCKET_NAME = os.environ.get('BUCKET_NAME', '')
USER_ID = os.environ.get('USER_ID', '')
JOB_NAME = os.environ.get('JOB_NAME', '')
TARGET_FIELD = os.environ.get('TARGET_FIELD', '0')
AUXILIARY_FIELDS = json.loads(os.environ.get('AUXILIARY_FIELDS', '[]'))
CLASS_NAMES = json.loads(os.environ.get('CLASS_NAMES', '[]'))
INPUT_HEIGHT = int(os.environ.get('INPUT_HEIGHT', '128'))
INPUT_WIDTH = int(os.environ.get('INPUT_WIDTH', '128'))


def load_audio_file(file_path: str, target_sr: int = 22050) -> np.ndarray:
    """音声ファイルを読み込み"""
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio
    except ImportError:
        from scipy.io import wavfile
        sr, audio = wavfile.read(file_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples)
        return audio


def generate_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """メルスペクトログラムを生成"""
    try:
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except ImportError:
        # librosaがない場合は簡易版（非推奨）
        print("Warning: librosa not available, using basic spectrogram")
        from scipy import signal
        f, t, Sxx = signal.spectrogram(audio, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        return Sxx_db[:n_mels, :]


def parse_filename(filename: str, separator: str = '_') -> Dict[str, str]:
    """ファイル名からメタデータを抽出"""
    name_without_ext = filename.rsplit('.', 1)[0]
    parts = name_without_ext.split(separator)
    return {str(i): part for i, part in enumerate(parts)}


def generate_class_label(metadata: Dict[str, str], target_field: str, aux_fields: List[str]) -> str:
    """ターゲットと補助フィールドからクラスラベルを生成"""
    label_parts = [metadata.get(target_field, '')]
    for aux_field in aux_fields:
        if aux_field in metadata:
            label_parts.append(metadata[aux_field])
    return '_'.join(label_parts)


def load_tfjs_model(model_dir: str) -> tf.keras.Model:
    """モデルをロード（SavedModel形式またはTFJS形式）"""
    model_dir_path = Path(model_dir)
    
    print(f"Loading model from: {model_dir_path}")
    
    # model.tar.gzが存在する場合は展開
    model_tar_path = model_dir_path / 'model.tar.gz'
    if model_tar_path.exists():
        print(f"Found model.tar.gz, extracting...")
        import tarfile
        with tarfile.open(model_tar_path, 'r:gz') as tar:
            tar.extractall(path=model_dir_path)
        print(f"Extraction complete.")
    
    # ディレクトリ内のファイル一覧を表示
    print(f"Files in model directory:")
    for f in model_dir_path.rglob('*'):
        if f.is_file():
            print(f"  - {f}")
    
    # 方法1: TensorFlow SavedModel形式を探す（audio_classifier/saved_model.pb）
    saved_model_dirs = list(model_dir_path.glob('*/saved_model.pb'))
    if saved_model_dirs:
        saved_model_dir = saved_model_dirs[0].parent
        print(f"Found TensorFlow SavedModel at: {saved_model_dir}")
        try:
            model = tf.keras.models.load_model(str(saved_model_dir))
            print(f"Model loaded successfully (SavedModel): {type(model)}")
            return model
        except Exception as e:
            print(f"Error loading SavedModel: {e}")
            # 次の方法を試す
    
    # 方法2: TFJS形式を探す（model.json）
    model_json_path = model_dir_path / 'model.json'
    if model_json_path.exists():
        print(f"Found TensorFlow.js model at: {model_json_path}")
        try:
            import tensorflowjs as tfjs
            print("Loading model with tensorflowjs...")
            model = tfjs.converters.load_keras_model(str(model_dir_path))
            print(f"Model loaded successfully (TFJS): {type(model)}")
            return model
        except ImportError as e:
            print(f"Warning: tensorflowjs not available: {e}")
        except Exception as e:
            print(f"Error loading TFJS model: {e}")
    
    # どの形式も見つからない
    raise FileNotFoundError(
        f"No supported model format found in {model_dir_path}. "
        f"Expected either SavedModel (saved_model.pb) or TensorFlow.js (model.json)"
    )


def prepare_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """データセットを準備"""
    X_list = []
    y_list = []
    file_list = []
    label_list = []
    
    # WAVファイルを再帰的に探す
    wav_files = list(Path(data_dir).rglob('*.wav'))
    
    if len(wav_files) == 0:
        raise ValueError(f"No WAV files found in {data_dir}")
    
    print(f"Found {len(wav_files)} WAV files")
    
    # クラス名からインデックスへのマッピング
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    for wav_file in wav_files:
        try:
            # 音声読み込み
            audio = load_audio_file(str(wav_file))
            
            # スペクトログラム生成
            mel_spec = generate_mel_spectrogram(audio, n_mels=INPUT_HEIGHT)
            
            # リサイズ
            if mel_spec.shape[1] < INPUT_WIDTH:
                # パディング
                pad_width = INPUT_WIDTH - mel_spec.shape[1]
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # トリミング
                mel_spec = mel_spec[:, :INPUT_WIDTH]
            
            # 正規化
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            # (H, W, C) 形式に変換
            mel_spec = mel_spec[:INPUT_HEIGHT, :INPUT_WIDTH]
            mel_spec = np.expand_dims(mel_spec, axis=-1)
            
            # ファイル名からクラスラベルを取得
            metadata = parse_filename(wav_file.name)
            class_label = generate_class_label(metadata, TARGET_FIELD, AUXILIARY_FIELDS)
            
            if class_label in class_to_idx:
                X_list.append(mel_spec)
                y_list.append(class_to_idx[class_label])
                file_list.append(wav_file.name)
                label_list.append(class_label)
            else:
                print(f"Warning: Unknown class '{class_label}' for file {wav_file.name}")
                
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue
    
    if len(X_list) == 0:
        raise ValueError("No valid samples found")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    print(f"Prepared dataset: X shape={X.shape}, y shape={y.shape}")
    
    return X, y, file_list, label_list


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict:
    """評価指標を計算"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # クラスごとの指標
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # 混同行列
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'class_metrics': [
            {
                'class_name': class_names[i] if i < len(class_names) else f'Class_{i}',
                'precision': float(class_precision[i]),
                'recall': float(class_recall[i]),
                'f1_score': float(class_f1[i]),
                'support': int(class_support[i])
            }
            for i in range(len(class_precision))
        ]
    }
    
    return metrics


def save_results(metrics: Dict, file_predictions: List[Dict], output_dir: str):
    """結果をJSON/CSVで保存"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # メトリクスをJSON保存
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {output_path / 'metrics.json'}")
    
    # ファイル別の予測結果をCSV保存
    with open(output_path / 'predictions.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'true_label', 'predicted_label', 'confidence', 'correct'])
        writer.writeheader()
        writer.writerows(file_predictions)
    
    print(f"Predictions saved to {output_path / 'predictions.csv'}")
    
    # サマリーを出力
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nClass-wise metrics:")
    for cm in metrics['class_metrics']:
        print(f"  {cm['class_name']}: F1={cm['f1_score']:.4f}, Support={cm['support']}")


def main():
    """メイン処理"""
    global CLASS_NAMES  # グローバル変数の宣言を関数の最初に移動
    
    try:
        print("=" * 60)
        print("Starting model evaluation...")
        print("=" * 60)
        print(f"Input data dir: {INPUT_DATA_DIR}")
        print(f"Input model dir: {INPUT_MODEL_DIR}")
        print(f"Output dir: {OUTPUT_DIR}")
        print(f"Class names: {CLASS_NAMES}")
        print(f"Target field: {TARGET_FIELD}")
        print(f"Auxiliary fields: {AUXILIARY_FIELDS}")
        print(f"Input size: {INPUT_HEIGHT}x{INPUT_WIDTH}")
        
        # ディレクトリの存在確認
        print("\nChecking directories...")
        print(f"Data dir exists: {Path(INPUT_DATA_DIR).exists()}")
        print(f"Model dir exists: {Path(INPUT_MODEL_DIR).exists()}")
        print(f"Output dir exists: {Path(OUTPUT_DIR).exists()}")
        
        # モデルをロード
        print("\n" + "=" * 60)
        print("Loading model...")
        print("=" * 60)
        model = load_tfjs_model(INPUT_MODEL_DIR)
        print(f"\nModel loaded successfully!")
        print(f"Model type: {type(model)}")
        try:
            model.summary()
        except:
            print("Model summary not available")
        
        # label_encoder.jsonからクラス名と問題タイプを読み込み（訓練時の設定を使用）
        label_encoder_path = Path(INPUT_MODEL_DIR) / 'label_encoder.json'
        problem_type = 'classification'  # デフォルト
        tolerance = 0.0  # デフォルト
        
        if label_encoder_path.exists():
            print(f"\nLoading label encoder from: {label_encoder_path}")
            with open(label_encoder_path, 'r') as f:
                label_encoder = json.load(f)
                # 訓練時のクラス名で上書き（重要：これにより正しいクラス数とクラス名を使用）
                CLASS_NAMES = label_encoder.get('classes', CLASS_NAMES)
                problem_type = label_encoder.get('problem_type', 'classification')
                tolerance = float(label_encoder.get('tolerance', 0.0))
                print(f"Loaded {len(CLASS_NAMES)} classes from label encoder: {CLASS_NAMES}")
                print(f"Problem type: {problem_type}")
                print(f"Tolerance: {tolerance}")
        else:
            print(f"\nWarning: label_encoder.json not found at {label_encoder_path}")
            print(f"Using CLASS_NAMES from environment: {CLASS_NAMES}")
            print(f"Using default problem_type: {problem_type}")
        
        # データセットを準備
        print("\nPreparing dataset...")
        X, y_true, file_list, label_list = prepare_dataset(INPUT_DATA_DIR)
        
        # 予測（問題タイプに応じて処理）
        print("\nRunning inference...")
        predictions = model.predict(X, batch_size=32, verbose=1)
        
        # ファイル別の予測結果
        file_predictions = []
        
        if problem_type == 'regression':
            # 回帰問題の処理
            y_pred_values = predictions.flatten()  # 連続値
            
            for i, (filename, true_label) in enumerate(zip(file_list, label_list)):
                pred_value = float(y_pred_values[i])
                true_value = float(true_label)
                error = abs(pred_value - true_value)
                correct = error <= tolerance  # 許容範囲内なら正解
                
                file_predictions.append({
                    'filename': filename,
                    'true_label': f'{true_value:.4f}',
                    'predicted_label': f'{pred_value:.4f}',
                    'confidence': f'{1.0 - min(error / (tolerance + 1e-8), 1.0):.4f}',  # 信頼度（誤差が小さいほど高い）
                    'correct': correct
                })
            
            # 回帰の評価指標
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error([float(l) for l in label_list], y_pred_values)
            mse = mean_squared_error([float(l) for l in label_list], y_pred_values)
            rmse = np.sqrt(mse)
            r2 = r2_score([float(l) for l in label_list], y_pred_values)
            within_tolerance = np.mean([fp['correct'] for fp in file_predictions])
            
            metrics = {
                'accuracy': float(within_tolerance),  # 許容範囲内の割合
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'tolerance': float(tolerance),
                'problem_type': 'regression',
                'class_metrics': []  # 回帰では不要
            }
        else:
            # 分類問題の処理
            y_pred = np.argmax(predictions, axis=1)
            
            for i, (filename, true_label, pred_idx) in enumerate(zip(file_list, label_list, y_pred)):
                pred_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f'Class_{pred_idx}'
                confidence = float(np.max(predictions[i]))
                
                # 許容範囲を考慮した正解判定（数値クラスの場合）
                if tolerance > 0:
                    try:
                        true_value = float(true_label)
                        pred_value = float(CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else pred_idx)
                        correct = abs(true_value - pred_value) <= tolerance
                    except (ValueError, IndexError):
                        correct = (y_true[i] == pred_idx)  # 数値でない場合は完全一致
                else:
                    correct = (y_true[i] == pred_idx)
                
                file_predictions.append({
                    'filename': filename,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': f'{confidence:.4f}',
                    'correct': correct
                })
            
            # 分類の評価指標を計算
            print("\nCalculating metrics...")
            metrics = calculate_metrics(y_true, y_pred, CLASS_NAMES)
            metrics['tolerance'] = float(tolerance)
            metrics['problem_type'] = 'classification'
            
            # 許容範囲を考慮した精度
            if tolerance > 0:
                accuracy_with_tolerance = np.mean([fp['correct'] for fp in file_predictions])
                metrics['accuracy_with_tolerance'] = float(accuracy_with_tolerance)
        
        # 結果を保存
        print("\nSaving results...")
        save_results(metrics, file_predictions, OUTPUT_DIR)
        
        print("\n" + "=" * 60)
        print("Evaluation completed successfully!")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: Evaluation failed!")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        print("=" * 60)
        raise  # エラーを再発生させてSageMakerに失敗を通知


if __name__ == '__main__':
    main()

