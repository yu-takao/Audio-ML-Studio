#!/usr/bin/env python3
"""
SageMaker Training Script for Audio Classification
オーディオファイルからスペクトログラムを生成し、CNNで分類するモデルを訓練

ハイパーパラメータ:
- epochs: 訓練エポック数
- batch_size: バッチサイズ
- learning_rate: 学習率
- validation_split: 検証データの割合
- test_split: テストデータの割合
- input_height: スペクトログラムの高さ
- input_width: スペクトログラムの幅
- target_field: ターゲットフィールドのインデックス（ファイル名の_区切り）
- auxiliary_fields: 補助フィールドのインデックス（JSON配列）
- class_names: クラス名（JSON配列）
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, callbacks
import logging
import boto3

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """コマンドライン引数を解析（基本型のみ）
    
    複雑な型（リスト、辞書）はSM_HPS環境変数から取得する。
    SageMakerがハイパーパラメータをコマンドライン引数として渡す際に、
    スペースを含むJSON配列が正しくパースされない問題を回避するため。
    """
    parser = argparse.ArgumentParser()
    
    # 基本型のハイパーパラメータのみ
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--test_split', type=float, default=0.15)
    parser.add_argument('--input_height', type=int, default=128)
    parser.add_argument('--input_width', type=int, default=128)
    parser.add_argument('--target_field', type=str, default='0')
    parser.add_argument('--problem_type', type=str, default='classification')
    parser.add_argument('--tolerance', type=float, default=0.0)
    
    # SageMaker環境変数
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    
    # 未知の引数は無視（SageMakerが追加する引数対応）
    args, _ = parser.parse_known_args()
    return args


def get_hyperparameters() -> dict:
    """SM_HPS環境変数から全ハイパーパラメータを取得
    
    SageMakerはSM_HPS環境変数にJSON形式で全ハイパーパラメータを格納する。
    複雑な型（リスト、辞書）はこちらから取得することで、
    コマンドライン引数のパース問題を回避できる。
    """
    sm_hps = os.environ.get('SM_HPS', '{}')
    try:
        hps = json.loads(sm_hps)
        logger.info(f"Loaded hyperparameters from SM_HPS: {list(hps.keys())}")
        return hps
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse SM_HPS: {e}")
        return {}


def extract_label_from_filename(filename: str, target_field_index: int) -> str:
    """ファイル名からラベルを抽出
    
    ファイル名の形式: field0_field1_field2_..._fieldN.wav
    target_field_indexで指定されたフィールドをラベルとして使用
    """
    # 拡張子を除去
    name_without_ext = os.path.splitext(filename)[0]
    
    # _で分割
    parts = name_without_ext.split('_')
    
    # target_field_indexが範囲内かチェック
    if target_field_index < len(parts):
        return parts[target_field_index]
    else:
        # 範囲外の場合は最初のフィールドを使用
        logger.warning(f"Target field index {target_field_index} out of range for {filename}, using first field")
        return parts[0] if parts else 'unknown'


def extract_auxiliary_features(filename: str, auxiliary_indices: list) -> dict:
    """ファイル名から補助特徴量を抽出"""
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    
    features = {}
    for idx in auxiliary_indices:
        if idx < len(parts):
            features[f'aux_{idx}'] = parts[idx]
    
    return features


def audio_to_spectrogram(audio_path: str, target_height: int, target_width: int) -> np.ndarray:
    """オーディオファイルをメルスペクトログラムに変換"""
    try:
        # オーディオを読み込み
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        # メルスペクトログラムを計算
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=target_height,
            n_fft=2048,
            hop_length=512
        )
        
        # デシベルスケールに変換
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 正規化 (-80dB to 0dB -> 0 to 1)
        mel_spec_normalized = (mel_spec_db + 80) / 80
        mel_spec_normalized = np.clip(mel_spec_normalized, 0, 1)
        
        # リサイズ（幅を調整）
        if mel_spec_normalized.shape[1] < target_width:
            # パディング
            pad_width = target_width - mel_spec_normalized.shape[1]
            mel_spec_normalized = np.pad(
                mel_spec_normalized, 
                ((0, 0), (0, pad_width)), 
                mode='constant'
            )
        elif mel_spec_normalized.shape[1] > target_width:
            # トリミング
            mel_spec_normalized = mel_spec_normalized[:, :target_width]
        
        return mel_spec_normalized
        
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return None


def load_dataset(data_dir: str, target_field_index: int, input_height: int, input_width: int):
    """データセットを読み込み"""
    logger.info(f"Loading dataset from {data_dir}")
    
    spectrograms = []
    labels = []
    filenames = []
    
    # 再帰的にWAVファイルを探索
    data_path = Path(data_dir)
    wav_files = list(data_path.rglob('*.wav')) + list(data_path.rglob('*.WAV'))
    
    logger.info(f"Found {len(wav_files)} WAV files")
    
    for wav_file in wav_files:
        # スペクトログラムを生成
        spectrogram = audio_to_spectrogram(str(wav_file), input_height, input_width)
        
        if spectrogram is not None:
            # ラベルを抽出
            label = extract_label_from_filename(wav_file.name, target_field_index)
            
            spectrograms.append(spectrogram)
            labels.append(label)
            filenames.append(wav_file.name)
    
    logger.info(f"Successfully processed {len(spectrograms)} files")
    
    return np.array(spectrograms), labels, filenames


def check_presplit_data(data_dir: str):
    """
    データが事前分割されているかチェック
    train/, validation/, test/ サブディレクトリがあれば事前分割とみなす
    """
    data_path = Path(data_dir)
    train_dir = data_path / 'train'
    val_dir = data_path / 'validation'
    test_dir = data_path / 'test'
    
    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        train_files = list(train_dir.rglob('*.wav')) + list(train_dir.rglob('*.WAV'))
        val_files = list(val_dir.rglob('*.wav')) + list(val_dir.rglob('*.WAV'))
        test_files = list(test_dir.rglob('*.wav')) + list(test_dir.rglob('*.WAV'))
        
        if len(train_files) > 0 and len(val_files) > 0 and len(test_files) > 0:
            logger.info("Detected pre-split data structure")
            logger.info(f"  - Train: {len(train_files)} files")
            logger.info(f"  - Validation: {len(val_files)} files")
            logger.info(f"  - Test: {len(test_files)} files")
            return True, str(train_dir), str(val_dir), str(test_dir)
    
    return False, None, None, None


def build_classification_model(input_shape: tuple, num_classes: int, learning_rate: float):
    """分類用CNNモデルを構築"""
    logger.info(f"Building classification model with input shape {input_shape} and {num_classes} classes")
    
    model = models.Sequential([
        # 入力層
        layers.Input(shape=input_shape),
        
        # 畳み込みブロック1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 畳み込みブロック2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 畳み込みブロック3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 畳み込みブロック4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 全結合層
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # 出力層（分類: softmax）
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # コンパイル
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_regression_model(input_shape: tuple, learning_rate: float):
    """回帰用CNNモデルを構築"""
    logger.info(f"Building regression model with input shape {input_shape}")
    
    model = models.Sequential([
        # 入力層
        layers.Input(shape=input_shape),
        
        # 畳み込みブロック1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 畳み込みブロック2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 畳み込みブロック3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 畳み込みブロック4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 全結合層
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # 出力層（回帰: linear）
        layers.Dense(1, activation='linear')
    ])
    
    # コンパイル（回帰用: MSE損失、MAE指標）
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model


def main():
    """メイン関数"""
    args = parse_args()
    hps = get_hyperparameters()
    
    # 複雑な型はSM_HPSから取得（コマンドライン引数のパース問題を回避）
    class_names = hps.get('class_names', [])
    auxiliary_fields = hps.get('auxiliary_fields', [])
    field_labels = hps.get('field_labels', [])

    # 単純な型も念のためSM_HPSから取得（優先的に使用）
    if 'problem_type' in hps:
        args.problem_type = hps['problem_type']
    if 'tolerance' in hps:
        args.tolerance = float(hps['tolerance'])
    if 'target_field' in hps:
        args.target_field = str(hps['target_field'])
    if 'epochs' in hps:
        args.epochs = int(hps['epochs'])
    if 'batch_size' in hps:
        args.batch_size = int(hps['batch_size'])
    if 'learning_rate' in hps:
        args.learning_rate = float(hps['learning_rate'])
    
    # S3アップロード用パラメータ（SM_HPSまたは環境変数から取得）
    bucket_name = hps.get('bucket_name') or os.environ.get('BUCKET_NAME')
    user_id = hps.get('user_id') or os.environ.get('USER_ID')
    job_name = hps.get('job_name') or os.environ.get('JOB_NAME')
    
    logger.info("=" * 50)
    logger.info("Audio ML Training Script")
    logger.info("=" * 50)
    logger.info(f"Parameters:")
    logger.info(f"  - problem_type: {args.problem_type}")
    logger.info(f"  - tolerance: {args.tolerance}")
    logger.info(f"  - epochs: {args.epochs}")
    logger.info(f"  - batch_size: {args.batch_size}")
    logger.info(f"  - learning_rate: {args.learning_rate}")
    logger.info(f"  - validation_split: {args.validation_split}")
    logger.info(f"  - test_split: {args.test_split}")
    logger.info(f"  - input_height: {args.input_height}")
    logger.info(f"  - input_width: {args.input_width}")
    logger.info(f"  - target_field: {args.target_field}")
    logger.info(f"  - auxiliary_fields: {auxiliary_fields}")
    logger.info(f"  - class_names: {class_names}")
    logger.info(f"  - field_labels: {field_labels}")
    logger.info(f"  - train directory: {args.train}")
    logger.info(f"  - model directory: {args.model_dir}")
    
    # パラメータを解析
    target_field_index = int(args.target_field)
    
    # auxiliary_fieldsをインデックスのリストに変換
    auxiliary_indices = auxiliary_fields if isinstance(auxiliary_fields, list) else []
    
    # 事前分割されたデータかチェック
    is_presplit, train_dir, val_dir, test_dir = check_presplit_data(args.train)
    
    if is_presplit:
        # 事前分割されたデータを使用（データリークを防ぐ正しい方法）
        logger.info("Using pre-split data (no data leakage)")
        
        X_train_raw, labels_train, _ = load_dataset(train_dir, target_field_index, args.input_height, args.input_width)
        X_val_raw, labels_val, _ = load_dataset(val_dir, target_field_index, args.input_height, args.input_width)
        X_test_raw, labels_test, _ = load_dataset(test_dir, target_field_index, args.input_height, args.input_width)
        
        if len(X_train_raw) == 0:
            raise ValueError("No valid audio files found in the training directory")
        
        # チャンネル次元を追加
        X_train = X_train_raw[..., np.newaxis]
        X_val = X_val_raw[..., np.newaxis]
        X_test = X_test_raw[..., np.newaxis]
        
        if args.problem_type == 'regression':
            # 回帰: ラベルを数値に変換
            logger.info("Processing labels for regression...")
            try:
                y_train = np.array([float(l) for l in labels_train], dtype=np.float32)
                y_val = np.array([float(l) for l in labels_val], dtype=np.float32)
                y_test = np.array([float(l) for l in labels_test], dtype=np.float32)
            except ValueError as e:
                raise ValueError(f"Failed to convert labels to float for regression: {e}")
            
            unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))
            label_encoder = None  # 回帰ではエンコーダー不要
            logger.info(f"Regression target range: {y_train.min():.2f} to {y_train.max():.2f}")
            logger.info(f"Unique values: {len(unique_labels)}")
        else:
            # 分類: 従来通りLabelEncoderを使用
            all_labels = labels_train + labels_val + labels_test
            label_encoder = LabelEncoder()
            label_encoder.fit(all_labels)
            
            y_train = label_encoder.transform(labels_train)
            y_val = label_encoder.transform(labels_val)
            y_test = label_encoder.transform(labels_test)
            
            unique_labels = label_encoder.classes_
            logger.info(f"Classes found: {unique_labels}")
            logger.info(f"Number of classes: {len(unique_labels)}")
        
        logger.info(f"Pre-split dataset:")
        logger.info(f"  - Training: {len(X_train)} samples (augmented)")
        logger.info(f"  - Validation: {len(X_val)} samples (original)")
        logger.info(f"  - Test: {len(X_test)} samples (original)")
        split_mode = "presplit"
        
    else:
        # 従来通りランダム分割（後方互換性のため）
        logger.info("Using random split (legacy mode)")
        
        # データセットを読み込み
        X, labels, filenames = load_dataset(
            args.train, 
            target_field_index, 
            args.input_height, 
            args.input_width
        )
        
        if len(X) == 0:
            raise ValueError("No valid audio files found in the training directory")
        
        # チャンネル次元を追加 (height, width) -> (height, width, 1)
        X = X[..., np.newaxis]
        
        if args.problem_type == 'regression':
            # 回帰: ラベルを数値に変換
            logger.info("Processing labels for regression...")
            try:
                y = np.array([float(l) for l in labels], dtype=np.float32)
            except ValueError as e:
                raise ValueError(f"Failed to convert labels to float for regression: {e}")
            
            unique_labels = np.unique(y)
            label_encoder = None
            logger.info(f"Regression target range: {y.min():.2f} to {y.max():.2f}")
            logger.info(f"Unique values: {len(unique_labels)}")
            
            # データを分割（回帰ではstratifyを使用しない）
            test_size = args.test_split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            val_size = args.validation_split / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42
            )
        else:
            # 分類: ラベルをエンコード
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(labels)
            
            # クラス情報をログ出力
            unique_labels = label_encoder.classes_
            logger.info(f"Classes found: {unique_labels}")
            logger.info(f"Number of classes: {len(unique_labels)}")
            logger.info(f"Samples per class:")
            for cls in unique_labels:
                count = sum(1 for l in labels if l == cls)
                logger.info(f"  - {cls}: {count} samples")
            
            # データを分割（分類ではstratifyを使用）
            test_size = args.test_split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            val_size = args.validation_split / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
            )
        
        logger.info(f"Dataset split:")
        logger.info(f"  - Training: {len(X_train)} samples")
        logger.info(f"  - Validation: {len(X_val)} samples")
        logger.info(f"  - Test: {len(X_test)} samples")
        split_mode = "random"
    
    # モデルを構築
    input_shape = (args.input_height, args.input_width, 1)
    if args.problem_type == 'regression':
        model = build_regression_model(input_shape, args.learning_rate)
    else:
        num_classes = len(unique_labels)
        model = build_classification_model(input_shape, num_classes, args.learning_rate)
    
    model.summary(print_fn=logger.info)
    
    # コールバック設定
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
    ]
    
    # 訓練
    logger.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=training_callbacks,
        verbose=1
    )
    
    # テストデータで評価
    logger.info("Evaluating on test data...")
    if args.problem_type == 'regression':
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Loss (MSE): {test_loss:.4f}")
        logger.info(f"Test MAE: {test_mae:.4f}")
        test_metric = test_mae  # 回帰ではMAEを主指標として使用
    else:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        test_metric = test_accuracy
    
    # モデルを保存
    model_path = os.path.join(args.model_dir, 'audio_classifier')
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # TensorFlow.js用にも保存（オプション）
    try:
        import tensorflowjs as tfjs
        tfjs_path = os.path.join(args.model_dir, 'tfjs_model')
        tfjs.converters.save_keras_model(model, tfjs_path)
        logger.info(f"TensorFlow.js model saved to {tfjs_path}")
    except ImportError:
        logger.info("TensorFlow.js not available, skipping TFJS export")
    
    # S3アップロード用の値は既にmain()の冒頭で取得済み

    # メタデータを保存
    def build_split_class_distribution(y_encoded: np.ndarray, encoder):
        dist = {}
        if y_encoded is None or encoder is None:
            return dist
        classes = encoder.classes_
        for idx, cls in enumerate(classes):
            dist[str(cls)] = int(np.sum(y_encoded == idx))
        return dist

    # 問題タイプに応じてメトリクス名を変更
    if args.problem_type == 'regression':
        metric_key = 'mae'
        history_metric_key = 'mae'
    else:
        metric_key = 'accuracy'
        history_metric_key = 'accuracy'

    metadata = {
        'problem_type': args.problem_type,
        'tolerance': args.tolerance,
        'classes': unique_labels.tolist() if hasattr(unique_labels, 'tolist') else list(unique_labels),
        'input_shape': list(input_shape),
        'target_field': args.target_field,
        'auxiliary_fields': auxiliary_indices,
        'dataset': {
            'split_mode': split_mode,
            'counts': {
                'train': int(len(X_train)),
                'validation': int(len(X_val)),
                'test': int(len(X_test)),
                'total': int(len(X_train) + len(X_val) + len(X_test)),
            },
            'class_distribution': build_split_class_distribution(y_train, label_encoder) if label_encoder else {}
        },
        'training_params': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'validation_split': args.validation_split,
            'test_split': args.test_split,
        },
        'metrics': {
            'test_loss': float(test_loss),
            f'test_{metric_key}': float(test_metric),
            'final_train_loss': float(history.history['loss'][-1]),
            f'final_train_{metric_key}': float(history.history[history_metric_key][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            f'final_val_{metric_key}': float(history.history[f'val_{history_metric_key}'][-1]),
        },
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            metric_key: [float(x) for x in history.history[history_metric_key]],
            'val_loss': [float(x) for x in history.history['val_loss']],
            f'val_{metric_key}': [float(x) for x in history.history[f'val_{history_metric_key}']],
        }
    }
    
    metadata_path = os.path.join(args.model_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")
    
    # ラベルエンコーダーも保存（problem_typeとtoleranceを含める）
    label_encoder_data = {
        'classes': unique_labels.tolist() if hasattr(unique_labels, 'tolist') else list(unique_labels),
        'problem_type': args.problem_type,
        'tolerance': args.tolerance,
    }
    label_encoder_path = os.path.join(args.model_dir, 'label_encoder.json')
    with open(label_encoder_path, 'w') as f:
        json.dump(label_encoder_data, f)
    logger.info(f"Label encoder saved to {label_encoder_path}")

    # 追加: S3へメタデータとラベルエンコーダーを直接アップロード（クライアントで別オブジェクトとして取得できるようにする）
    if bucket_name and user_id and job_name:
        s3_client = boto3.client('s3')
        base_key = f"models/{user_id}/{job_name}/output"
        try:
            s3_client.upload_file(metadata_path, bucket_name, f"{base_key}/model_metadata.json")
            logger.info(f"Uploaded metadata to s3://{bucket_name}/{base_key}/model_metadata.json")
        except Exception as e:
            logger.warning(f"Failed to upload metadata to S3: {e}")
        try:
            s3_client.upload_file(label_encoder_path, bucket_name, f"{base_key}/label_encoder.json")
            logger.info(f"Uploaded label encoder to s3://{bucket_name}/{base_key}/label_encoder.json")
        except Exception as e:
            logger.warning(f"Failed to upload label encoder to S3: {e}")
    else:
        logger.warning(
            "Skip uploading metadata to S3 because bucket/user/job is missing. "
            f"bucket_name={bucket_name}, user_id={user_id}, job_name={job_name}"
        )
    
    logger.info("=" * 50)
    logger.info("Training completed successfully!")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()

