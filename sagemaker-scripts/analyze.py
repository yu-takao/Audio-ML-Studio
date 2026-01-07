#!/usr/bin/env python3
"""
SageMaker Processing Script for Model Analysis / Visualization
モデルの判定根拠を可視化するスクリプト

出力:
- Grad-CAMヒートマップ（サンプルごと＋クラス別平均）
- クラス別平均スペクトログラム
- 周波数帯別寄与度グラフ
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
import librosa
import matplotlib
matplotlib.use('Agg')  # GUIなし環境用
import matplotlib.pyplot as plt
import logging
from collections import defaultdict
import tarfile
import shutil

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser()
    
    # 入力パス
    parser.add_argument('--model-path', type=str, required=True,
                        help='S3 path to model.tar.gz or extracted model directory')
    parser.add_argument('--data-path', type=str, required=True,
                        help='S3 path to test data directory')
    
    # 出力パス
    parser.add_argument('--output-path', type=str, required=True,
                        help='S3 path to save analysis results')
    
    # 解析パラメータ
    parser.add_argument('--target-field', type=int, default=1,
                        help='Field index for label extraction from filename')
    parser.add_argument('--input-height', type=int, default=128)
    parser.add_argument('--input-width', type=int, default=128)
    parser.add_argument('--max-samples-per-class', type=int, default=10,
                        help='Maximum samples to analyze per class')
    
    # SageMaker Processing環境変数
    parser.add_argument('--local-model-dir', type=str, 
                        default='/opt/ml/processing/model')
    parser.add_argument('--local-data-dir', type=str,
                        default='/opt/ml/processing/data')
    parser.add_argument('--local-output-dir', type=str,
                        default='/opt/ml/processing/output')
    
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Unknown arguments ignored: {unknown}")
    return args


def extract_label_from_filename(filename: str, target_field_index: int) -> str:
    """ファイル名からラベルを抽出"""
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    if target_field_index < len(parts):
        return parts[target_field_index]
    return parts[0] if parts else 'unknown'


def audio_to_spectrogram(audio_path: str, target_height: int, target_width: int) -> np.ndarray:
    """オーディオファイルをメルスペクトログラムに変換"""
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=target_height, n_fft=2048, hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_normalized = (mel_spec_db + 80) / 80
        mel_spec_normalized = np.clip(mel_spec_normalized, 0, 1)
        
        # リサイズ
        if mel_spec_normalized.shape[1] < target_width:
            pad_width = target_width - mel_spec_normalized.shape[1]
            mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant')
        elif mel_spec_normalized.shape[1] > target_width:
            mel_spec_normalized = mel_spec_normalized[:, :target_width]
        
        return mel_spec_normalized
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return None


def load_keras_model(model_dir: str):
    """Kerasモデルを読み込む"""
    # model.tar.gzを展開する場合
    tar_path = os.path.join(model_dir, 'model.tar.gz')
    if os.path.exists(tar_path):
        logger.info(f"Extracting {tar_path}")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(model_dir)
    
    # audio_classifier ディレクトリを探す
    model_path = os.path.join(model_dir, 'audio_classifier')
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        return load_model(model_path)
    
    # SavedModel形式を探す
    for root, dirs, files in os.walk(model_dir):
        if 'saved_model.pb' in files:
            logger.info(f"Loading model from {root}")
            return load_model(root)
    
    raise FileNotFoundError(f"No Keras model found in {model_dir}")


def get_last_conv_layer(model):
    """最後のConv2D層を取得"""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in model")


def compute_gradcam(model, input_tensor, class_index, last_conv_layer_name):
    """Grad-CAMを計算"""
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(input_tensor)
        class_output = predictions[:, class_index]
    
    grads = tape.gradient(class_output, conv_output)
    
    # Global Average Pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv output by the pooled gradients
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    
    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()


def resize_heatmap(heatmap, target_shape):
    """ヒートマップをリサイズ"""
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = tf.image.resize(heatmap, target_shape).numpy()
    return heatmap[:, :, 0]


def save_gradcam_overlay(spectrogram, heatmap, output_path, title="Grad-CAM"):
    """スペクトログラムとGrad-CAMのオーバーレイ画像を保存"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 元のスペクトログラム
    axes[0].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Spectrogram')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Frequency (Mel)')
    
    # Grad-CAMヒートマップ
    axes[1].imshow(heatmap, aspect='auto', origin='lower', cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Frequency (Mel)')
    
    # オーバーレイ
    axes[2].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[2].imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Frequency (Mel)')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def save_class_avg_spectrogram(avg_spec, class_name, output_path):
    """クラス別平均スペクトログラムを保存"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(avg_spec, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f'Average Spectrogram - Class {class_name}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency (Mel)')
    plt.colorbar(im, ax=ax, label='Normalized Amplitude')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def save_frequency_importance(freq_importance, class_names, output_path):
    """周波数帯別寄与度グラフを保存"""
    n_classes = len(class_names)
    n_freq = len(freq_importance[class_names[0]])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_freq)
    width = 0.8 / n_classes
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        offset = (i - n_classes / 2 + 0.5) * width
        ax.bar(x + offset, freq_importance[cls], width, label=cls, color=color, alpha=0.8)
    
    ax.set_xlabel('Frequency Bin (Mel)')
    ax.set_ylabel('Importance Score (Avg Grad-CAM)')
    ax.set_title('Frequency Band Importance by Class')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    ax.set_xticks(x[::8])
    ax.set_xticklabels([f'{i}' for i in x[::8]])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    logger.info("=== Model Analysis Started ===")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output path: {args.output_path}")
    
    # ディレクトリ作成
    os.makedirs(args.local_output_dir, exist_ok=True)
    
    # モデル読み込み
    logger.info("Loading model...")
    model = load_keras_model(args.local_model_dir)
    model.summary(print_fn=logger.info)
    
    # 最後のConv層を取得
    last_conv_layer = get_last_conv_layer(model)
    last_conv_layer_name = last_conv_layer.name
    logger.info(f"Last Conv layer: {last_conv_layer_name}")
    
    # データ読み込み
    logger.info("Loading data...")
    data_dir = Path(args.local_data_dir)
    
    # test/ サブディレクトリがあればそこから読む
    test_dir = data_dir / 'test'
    if test_dir.exists():
        wav_files = list(test_dir.rglob('*.wav')) + list(test_dir.rglob('*.WAV'))
    else:
        wav_files = list(data_dir.rglob('*.wav')) + list(data_dir.rglob('*.WAV'))
    
    logger.info(f"Found {len(wav_files)} WAV files")
    
    # クラス別にグループ化
    files_by_class = defaultdict(list)
    for f in wav_files:
        label = extract_label_from_filename(f.name, args.target_field)
        files_by_class[label].append(f)
    
    class_names = sorted(files_by_class.keys())
    logger.info(f"Classes: {class_names}")
    
    # メタデータ読み込み（クラス→インデックスのマッピング）
    metadata_path = os.path.join(args.local_model_dir, 'model_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        model_classes = metadata.get('classes', class_names)
    else:
        model_classes = class_names
    
    class_to_idx = {c: i for i, c in enumerate(model_classes)}
    logger.info(f"Class to index: {class_to_idx}")
    
    # 解析結果の格納
    class_spectrograms = defaultdict(list)
    class_gradcams = defaultdict(list)
    sample_results = []
    
    # サンプルごとの解析
    logger.info("Analyzing samples...")
    for cls in class_names:
        files = files_by_class[cls][:args.max_samples_per_class]
        logger.info(f"  Class {cls}: {len(files)} samples")
        
        for i, wav_file in enumerate(files):
            spec = audio_to_spectrogram(str(wav_file), args.input_height, args.input_width)
            if spec is None:
                continue
            
            class_spectrograms[cls].append(spec)
            
            # モデル入力形式に変換
            input_tensor = spec[..., np.newaxis]
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # 予測
            predictions = model.predict(input_tensor, verbose=0)
            pred_class_idx = np.argmax(predictions[0])
            pred_class = model_classes[pred_class_idx] if pred_class_idx < len(model_classes) else str(pred_class_idx)
            confidence = float(predictions[0][pred_class_idx])
            
            # Grad-CAM計算（正解クラスに対して）
            if cls in class_to_idx:
                target_idx = class_to_idx[cls]
                heatmap = compute_gradcam(model, input_tensor, target_idx, last_conv_layer_name)
                heatmap_resized = resize_heatmap(heatmap, (args.input_height, args.input_width))
                class_gradcams[cls].append(heatmap_resized)
                
                # サンプル画像を保存（最初の数個のみ）
                if i < 3:
                    sample_output = os.path.join(
                        args.local_output_dir, 
                        f"sample_{cls}_{i}_gradcam.png"
                    )
                    save_gradcam_overlay(
                        spec, heatmap_resized, sample_output,
                        title=f"Class: {cls} | Pred: {pred_class} ({confidence:.2%})"
                    )
                    sample_results.append({
                        'filename': wav_file.name,
                        'true_class': cls,
                        'pred_class': pred_class,
                        'confidence': confidence,
                        'image': f"sample_{cls}_{i}_gradcam.png"
                    })
    
    # クラス別平均スペクトログラム
    logger.info("Computing class-wise average spectrograms...")
    for cls in class_names:
        if len(class_spectrograms[cls]) > 0:
            avg_spec = np.mean(class_spectrograms[cls], axis=0)
            output_path = os.path.join(args.local_output_dir, f"class_{cls}_avg_spectrogram.png")
            save_class_avg_spectrogram(avg_spec, cls, output_path)
    
    # クラス別平均Grad-CAM
    logger.info("Computing class-wise average Grad-CAMs...")
    class_avg_gradcams = {}
    for cls in class_names:
        if len(class_gradcams[cls]) > 0:
            avg_gradcam = np.mean(class_gradcams[cls], axis=0)
            class_avg_gradcams[cls] = avg_gradcam
            
            # 画像保存
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(avg_gradcam, aspect='auto', origin='lower', cmap='jet')
            ax.set_title(f'Average Grad-CAM - Class {cls}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency (Mel)')
            plt.colorbar(im, ax=ax, label='Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(args.local_output_dir, f"class_{cls}_avg_gradcam.png"), 
                        dpi=100, bbox_inches='tight')
            plt.close()
    
    # 周波数帯別寄与度（Grad-CAMを周波数軸で集計）
    logger.info("Computing frequency band importance...")
    freq_importance = {}
    for cls in class_names:
        if cls in class_avg_gradcams:
            # 時間方向に平均して周波数ごとの重要度を取得
            freq_importance[cls] = np.mean(class_avg_gradcams[cls], axis=1).tolist()
    
    if freq_importance:
        save_frequency_importance(
            freq_importance, 
            [c for c in class_names if c in freq_importance],
            os.path.join(args.local_output_dir, "frequency_importance.png")
        )
    
    # サマリーJSON
    summary = {
        'classes': class_names,
        'samples_per_class': {cls: len(files_by_class[cls]) for cls in class_names},
        'analyzed_per_class': {cls: len(class_spectrograms[cls]) for cls in class_names},
        'frequency_importance': freq_importance,
        'sample_results': sample_results,
        'output_files': {
            'frequency_importance': 'frequency_importance.png',
            'class_avg_spectrograms': [f"class_{cls}_avg_spectrogram.png" for cls in class_names],
            'class_avg_gradcams': [f"class_{cls}_avg_gradcam.png" for cls in class_names if cls in class_avg_gradcams],
            'sample_gradcams': [s['image'] for s in sample_results]
        }
    }
    
    with open(os.path.join(args.local_output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=== Analysis Complete ===")
    logger.info(f"Results saved to {args.local_output_dir}")


if __name__ == '__main__':
    main()

