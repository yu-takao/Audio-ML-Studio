#!/usr/bin/env python3
"""
SageMaker Training Script for Audio Classification (Script Mode)
2D-CNN model training with optional auxiliary inputs

使用方法:
- このスクリプトをtar.gzにして S3 にアップロード
- SageMaker Training Job で AWS 提供の TensorFlow イメージと共に使用
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# SageMakerの環境変数
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
SM_OUTPUT_DATA_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser()
    
    # ハイパーパラメータ
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--test_split', type=float, default=0.15)
    parser.add_argument('--input_height', type=int, default=128)
    parser.add_argument('--input_width', type=int, default=128)
    parser.add_argument('--target_field', type=str, default='0')
    parser.add_argument('--auxiliary_fields', type=str, default='[]')
    parser.add_argument('--class_names', type=str, default='[]')
    
    # SageMaker 環境変数
    parser.add_argument('--model_dir', type=str, default=SM_MODEL_DIR)
    parser.add_argument('--train', type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument('--output_data_dir', type=str, default=SM_OUTPUT_DATA_DIR)
    
    return parser.parse_args()


def load_audio_file(file_path: str, target_sr: int = 22050) -> np.ndarray:
    """音声ファイルを読み込み（librosaを使用）"""
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio
    except ImportError:
        # librosaがない場合はscipy.io.wavfileを使用
        from scipy.io import wavfile
        sr, audio = wavfile.read(file_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        # リサンプリング（簡易版）
        if sr != target_sr:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def generate_mel_spectrogram(
    audio: np.ndarray, 
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """Melスペクトログラムを生成"""
    try:
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    except ImportError:
        # librosaがない場合は簡易実装
        from scipy import signal
        f, t, Sxx = signal.spectrogram(audio, sr, nperseg=n_fft, noverlap=n_fft-hop_length)
        # Melスケールに変換（簡易版）
        mel_spec_db = 10 * np.log10(Sxx + 1e-10)
        mel_spec_db = mel_spec_db[:n_mels, :]
    
    mel_spec_normalized = (mel_spec_db + 80) / 80
    mel_spec_normalized = np.clip(mel_spec_normalized, 0, 1)
    return mel_spec_normalized


def resize_spectrogram(spec: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """スペクトログラムをリサイズ"""
    spec_tensor = tf.convert_to_tensor(spec[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
    resized = tf.image.resize(spec_tensor, [target_height, target_width])
    return resized.numpy()[0, :, :, 0]


def parse_filename(filename: str, separator: str = '_') -> list:
    """ファイル名からメタデータを抽出"""
    name = os.path.splitext(filename)[0]
    parts = name.split(separator)
    return parts


def extract_field_value(filename: str, field_index: int, separator: str = '_'):
    """ファイル名から特定のフィールドを抽出"""
    parts = parse_filename(filename, separator)
    if 0 <= field_index < len(parts):
        try:
            return float(parts[field_index])
        except ValueError:
            return parts[field_index]
    return None


def build_model(
    input_height: int,
    input_width: int,
    num_classes: int,
    num_auxiliary: int = 0
) -> keras.Model:
    """2D-CNNモデルを構築"""
    
    spec_input = keras.Input(shape=(input_height, input_width, 1), name='spectrogram_input')
    
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(spec_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    if num_auxiliary > 0:
        aux_input = keras.Input(shape=(num_auxiliary,), name='auxiliary_input')
        aux_dense = keras.layers.Dense(32, activation='relu')(aux_input)
        aux_dense = keras.layers.BatchNormalization()(aux_dense)
        combined = keras.layers.concatenate([x, aux_dense])
        x = keras.layers.Dense(128, activation='relu')(combined)
        x = keras.layers.Dropout(0.5)(x)
        output = keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
        model = keras.Model(inputs=[spec_input, aux_input], outputs=output)
    else:
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        output = keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
        model = keras.Model(inputs=spec_input, outputs=output)
    
    return model


def load_dataset(args, class_names: list, target_field: int, auxiliary_fields: list):
    """データセットを読み込み"""
    data_dir = args.train
    input_height = args.input_height
    input_width = args.input_width
    
    print(f"Loading dataset from {data_dir}")
    
    spectrograms = []
    auxiliary_data = []
    labels = []
    
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if not filename.lower().endswith('.wav'):
                continue
            
            file_path = os.path.join(root, filename)
            audio = load_audio_file(file_path)
            if audio is None:
                continue
            
            spec = generate_mel_spectrogram(audio)
            spec = resize_spectrogram(spec, input_height, input_width)
            spectrograms.append(spec)
            
            target_value = extract_field_value(filename, target_field)
            if target_value is None:
                spectrograms.pop()
                continue
            
            label_str = str(target_value)
            if label_str in class_to_idx:
                labels.append(class_to_idx[label_str])
            else:
                print(f"Warning: Unknown class {label_str} in {filename}")
                spectrograms.pop()
                continue
            
            if auxiliary_fields:
                aux_values = []
                for field_idx in auxiliary_fields:
                    value = extract_field_value(filename, field_idx)
                    if isinstance(value, (int, float)):
                        aux_values.append(float(value))
                    else:
                        aux_values.append(0.0)
                auxiliary_data.append(aux_values)
    
    print(f"Loaded {len(spectrograms)} samples")
    
    X = np.array(spectrograms)[..., np.newaxis]
    y = keras.utils.to_categorical(labels, num_classes=len(class_names))
    
    X_aux = None
    if auxiliary_fields:
        X_aux = np.array(auxiliary_data)
        X_aux = (X_aux - X_aux.mean(axis=0)) / (X_aux.std(axis=0) + 1e-8)
    
    return X, X_aux, y


def main():
    args = parse_args()
    
    print("=" * 50)
    print("Audio ML Training Script (SageMaker Script Mode)")
    print("=" * 50)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Input Size: {args.input_height}x{args.input_width}")
    print(f"Training Data: {args.train}")
    print(f"Model Output: {args.model_dir}")
    print("=" * 50)
    
    # 追加パッケージのインストール
    print("Installing additional packages...")
    os.system('pip install librosa soundfile --quiet')
    
    auxiliary_fields = json.loads(args.auxiliary_fields)
    class_names = json.loads(args.class_names)
    target_field = int(args.target_field)
    
    print(f"Class names: {class_names}")
    print(f"Target field: {target_field}")
    print(f"Auxiliary fields: {auxiliary_fields}")
    
    X, X_aux, y = load_dataset(args, class_names, target_field, auxiliary_fields)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    if X_aux is not None:
        print(f"X_aux shape: {X_aux.shape}")
    
    # データ分割
    num_samples = len(X)
    test_size = int(num_samples * args.test_split)
    train_size = num_samples - test_size
    
    indices = np.random.permutation(num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    if X_aux is not None:
        X_aux_train, X_aux_test = X_aux[train_indices], X_aux[test_indices]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # モデル構築
    num_auxiliary = len(auxiliary_fields)
    model = build_model(
        input_height=args.input_height,
        input_width=args.input_width,
        num_classes=len(class_names),
        num_auxiliary=num_auxiliary
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # コールバック
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        ),
    ]
    
    # 訓練
    if X_aux is not None:
        train_data = [X_train, X_aux_train]
        test_data = [X_test, X_aux_test]
    else:
        train_data = X_train
        test_data = X_test
    
    history = model.fit(
        train_data, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        callbacks=callbacks,
        verbose=2
    )
    
    # 評価
    test_loss, test_acc = model.evaluate(test_data, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # モデル保存
    os.makedirs(args.model_dir, exist_ok=True)
    
    # SavedModel形式で保存（SageMaker推奨）
    model.save(os.path.join(args.model_dir, '1'))
    print(f"Model saved to {args.model_dir}/1")
    
    # TensorFlow.js形式に変換（可能な場合）
    try:
        import subprocess
        subprocess.run(['pip', 'install', 'tensorflowjs', '--quiet'], check=True)
        import tensorflowjs as tfjs
        tfjs_output = os.path.join(args.model_dir, 'tfjs_model')
        os.makedirs(tfjs_output, exist_ok=True)
        tfjs.converters.save_keras_model(model, tfjs_output)
        print(f"TensorFlow.js model saved to {tfjs_output}")
    except Exception as e:
        print(f"TensorFlow.js conversion skipped: {e}")
    
    # メタデータ保存
    metadata = {
        'class_names': class_names,
        'target_field': target_field,
        'auxiliary_fields': auxiliary_fields,
        'input_height': args.input_height,
        'input_width': args.input_width,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'epochs_trained': len(history.history['loss']),
    }
    
    with open(os.path.join(args.model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 訓練履歴保存
    history_data = {
        'loss': [float(v) for v in history.history['loss']],
        'accuracy': [float(v) for v in history.history['accuracy']],
        'val_loss': [float(v) for v in history.history.get('val_loss', [])],
        'val_accuracy': [float(v) for v in history.history.get('val_accuracy', [])],
    }
    
    with open(os.path.join(args.model_dir, 'history.json'), 'w') as f:
        json.dump(history_data, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    main()
