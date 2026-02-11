import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import io

class YAMNetLoader:
    _instance = None
    _model = None

    @classmethod
    def get_model(cls):
        """Singleton to load model only once."""
        if cls._model is None:
            # Load YAMNet from TensorFlow Hub
            cls._model = hub.load('https://tfhub.dev/google/yamnet/1')
        return cls._model

def compute_embedding(audio_bytes: bytes, filename: str) -> list[float]:
    """
    Processes audio bytes and returns a 1024-d embedding vector.
    """
    # 1. Decode and Resample
    # YAMNet strictly requires 16kHz mono audio.
    # librosa handles decoding (mp3, wav) and resampling automatically.
    try:
        wav, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    except Exception as e:
        raise ValueError(f"Failed to process audio file {filename}: {e}")

    # 2. Inference
    model = YAMNetLoader.get_model()
    
    # The model returns: scores (classes), embeddings, spectrogram
    _, embeddings, _ = model(wav)

    # 3. Aggregation (CRITICAL STEP)
    # YAMNet produces 1 embedding per 0.48s chunk.
    # A 5-second file yields ~10 vectors. We need ONE vector to represent the file.
    # We use "Global Average Pooling" (mean across the 0-axis).
    global_embedding = np.mean(embeddings, axis=0)

    return global_embedding.tolist()
