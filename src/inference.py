"""
Inference script for making predictions on individual audio files.
"""

import os
import sys
import yaml
import torch
import torchaudio
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ConformerMultimodalModel
from src.data.preprocessing import AudioPreprocessor, TextPreprocessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inference for emotion recognition')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--text', type=str, default="",
                        help='Transcription text (optional)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, config: dict, device: str):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model
    model = ConformerMultimodalModel(
        audio_encoder_name=config['model']['audio_encoder'],
        text_encoder_name=config['model']['text_encoder'],
        num_classes=config['model']['num_classes'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        fusion_type=config['model']['fusion_type']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    return model


def load_and_preprocess_audio(
    audio_path: str,
    sample_rate: int,
    max_length: float,
    audio_preprocessor
) -> torch.Tensor:
    """Load and preprocess audio file."""
    print(f"Loading audio from {audio_path}...")
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Truncate or pad to max_length
    max_samples = int(max_length * sample_rate)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    elif waveform.shape[1] < max_samples:
        padding = max_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    # Apply preprocessing
    audio_features = audio_preprocessor(waveform.squeeze(0))
    
    return audio_features.unsqueeze(0)  # Add batch dimension


def preprocess_text(text: str, text_preprocessor) -> tuple:
    """Preprocess text."""
    if not text or text.strip() == "":
        text = "[EMPTY]"
    
    text_features = text_preprocessor(text)
    
    return (text_features['input_ids'].unsqueeze(0),
            text_features['attention_mask'].unsqueeze(0))


def predict(model, audio, text_input_ids, text_attention_mask, device, emotion_labels):
    """Make prediction."""
    # Move to device
    audio = audio.to(device)
    text_input_ids = text_input_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(audio, text_input_ids, text_attention_mask)
    
    # Get prediction
    logits = outputs['logits']
    probabilities = torch.softmax(logits, dim=-1)
    prediction = torch.argmax(logits, dim=-1).item()
    confidence = probabilities[0, prediction].item()
    
    # Get top 3 predictions
    top_probs, top_indices = torch.topk(probabilities[0], k=min(3, len(emotion_labels)))
    top_emotions = [(emotion_labels[idx.item()], prob.item()) 
                   for idx, prob in zip(top_indices, top_probs)]
    
    return prediction, confidence, top_emotions, probabilities[0]


def main():
    """Main inference function."""
    args = parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Initialize preprocessors
    audio_preprocessor = AudioPreprocessor(
        sample_rate=config['dataset']['sample_rate'],
        use_mel_spectrogram=False,
        apply_spec_augment=False
    )
    audio_preprocessor.eval()
    
    text_preprocessor = TextPreprocessor(
        model_name=config['text']['model_name'],
        max_length=config['text']['max_length']
    )
    
    # Preprocess audio
    audio = load_and_preprocess_audio(
        args.audio,
        config['dataset']['sample_rate'],
        config['audio']['max_length'],
        audio_preprocessor
    )
    
    # Preprocess text
    text_input_ids, text_attention_mask = preprocess_text(
        args.text,
        text_preprocessor
    )
    
    # Make prediction
    emotion_labels = config['dataset']['emotions']
    prediction, confidence, top_emotions, all_probs = predict(
        model, audio, text_input_ids, text_attention_mask,
        device, emotion_labels
    )
    
    # Print results
    print("\n" + "="*60)
    print("EMOTION RECOGNITION RESULT")
    print("="*60)
    print(f"Audio file: {args.audio}")
    if args.text:
        print(f"Transcription: {args.text}")
    print("\n" + "-"*60)
    print(f"Predicted Emotion: {emotion_labels[prediction]}")
    print(f"Confidence: {confidence:.2%}")
    print("\n" + "-"*60)
    print("Top 3 Predictions:")
    for i, (emotion, prob) in enumerate(top_emotions, 1):
        print(f"  {i}. {emotion:12s}: {prob:.2%}")
    print("\n" + "-"*60)
    print("All Probabilities:")
    for emotion, prob in zip(emotion_labels, all_probs):
        bar = "█" * int(prob * 40)
        print(f"  {emotion:12s}: {bar:40s} {prob:.2%}")
    print("="*60)


if __name__ == '__main__':
    main()

