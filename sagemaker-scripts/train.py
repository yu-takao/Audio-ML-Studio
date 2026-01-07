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
    """コマンドライン引数を解析"""
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
    # nargs='*'でリスト引数を受け取る（SageMakerがスペースで分割するため）
    parser.add_argument('--auxiliary_fields', nargs='*', default=[])
    parser.add_argument('--class_names', nargs='*', default=[])
    # S3アップロード用情報（オプション）
    parser.add_argument('--bucket_name', type=str, default=None)
    parser.add_argument('--user_id', type=str, default=None)
    parser.add_argument('--job_name', type=str, default=None)
    
    # SageMaker環境変数
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    
    # 未知の引数も許容（SageMakerが追加する引数対応）
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Unknown arguments ignored: {unknown}")
    return args


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


def build_model(input_shape: tuple, num_classes: int, learning_rate: float):
    """CNNモデルを構築"""
    logger.info(f"Building model with input shape {input_shape} and {num_classes} classes")
    
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
        
        # 出力層
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


def main():
    """メイン関数"""
    args = parse_args()
    
    logger.info("=" * 50)
    logger.info("Audio Classification Training Script")
    logger.info("=" * 50)
    logger.info(f"Parameters:")
    logger.info(f"  - epochs: {args.epochs}")
    logger.info(f"  - batch_size: {args.batch_size}")
    logger.info(f"  - learning_rate: {args.learning_rate}")
    logger.info(f"  - validation_split: {args.validation_split}")
    logger.info(f"  - test_split: {args.test_split}")
    logger.info(f"  - input_height: {args.input_height}")
    logger.info(f"  - input_width: {args.input_width}")
    logger.info(f"  - target_field: {args.target_field}")
    logger.info(f"  - auxiliary_fields: {args.auxiliary_fields}")
    logger.info(f"  - class_names: {args.class_names}")
    logger.info(f"  - train directory: {args.train}")
    logger.info(f"  - model directory: {args.model_dir}")
    
    # パラメータを解析
    target_field_index = int(args.target_field)
    
    # auxiliary_fieldsとclass_namesの処理（リストまたはJSON文字列）
    def parse_list_arg(arg):
        """リスト引数を解析（SageMakerが渡す形式に対応）"""
        if isinstance(arg, list):
            # nargs='*'で受け取った場合、最初の要素が'['で始まる場合はJSON
            if len(arg) > 0 and isinstance(arg[0], str) and arg[0].startswith('['):
                # スペースで分割された引数を結合してJSONとして解析
                joined = ' '.join(arg)
                try:
                    return json.loads(joined)
                except json.JSONDecodeError:
                    # それでもダメなら個々の要素をクリーンアップ
                    cleaned = []
                    for item in arg:
                        item = item.strip("[]',\" ")
                        if item:
                            cleaned.append(item)
                    return cleaned
            else:
                return arg
        elif isinstance(arg, str):
            if arg.startswith('['):
                return json.loads(arg)
            return [arg] if arg else []
        return []
    
    auxiliary_indices = parse_list_arg(args.auxiliary_fields)
    class_names = parse_list_arg(args.class_names)
    
    logger.info(f"Parsed auxiliary_indices: {auxiliary_indices}")
    logger.info(f"Parsed class_names: {class_names}")
    
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
        
        # 全ラベルを集めてエンコーダーを学習
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
        
        # ラベルをエンコード
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
        
        # データを分割
        # まずテストデータを分離
        test_size = args.test_split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 残りを訓練と検証に分割
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
    num_classes = len(unique_labels)
    model = build_model(input_shape, num_classes, args.learning_rate)
    
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
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    
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
    
    # S3アップロード用の値を環境変数から補完（SageMakerのEnvironmentで渡すため）
    bucket_name = args.bucket_name or os.environ.get('BUCKET_NAME')
    user_id = args.user_id or os.environ.get('USER_ID')
    job_name = args.job_name or os.environ.get('JOB_NAME')

    # メタデータを保存
    def build_split_class_distribution(y_encoded: np.ndarray, encoder: LabelEncoder):
        dist = {}
        if y_encoded is None:
            return dist
        classes = encoder.classes_
        for idx, cls in enumerate(classes):
            dist[str(cls)] = int(np.sum(y_encoded == idx))
        return dist

    metadata = {
        'classes': unique_labels.tolist(),
        'input_shape': list(input_shape),
        'target_field': args.target_field,
        'auxiliary_fields': auxiliary_indices,
        'dataset': {
            'split_mode': split_mode,
            'counts': {
                'train': int(len(X_train)),
                'validation': int(len(X_val)),
                'test': int(len(X_test)),
            },
            'class_distribution': {
                'train': build_split_class_distribution(y_train, label_encoder),
                'validation': build_split_class_distribution(y_val, label_encoder),
                'test': build_split_class_distribution(y_test, label_encoder),
            }
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
            'test_accuracy': float(test_accuracy),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        },
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        }
    }
    
    metadata_path = os.path.join(args.model_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")
    
    # ラベルエンコーダーも保存
    label_encoder_path = os.path.join(args.model_dir, 'label_encoder.json')
    with open(label_encoder_path, 'w') as f:
        json.dump({'classes': unique_labels.tolist()}, f)
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

