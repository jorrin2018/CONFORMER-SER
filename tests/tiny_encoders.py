"""Encoders diminutos y ALEATORIOS para probar la ruta de codigo sin HuggingFace.

Solo para tests y para `check_deck_variants.py --tiny`. Nunca para entrenar: la
regla del repo es que un encoder que no carga para la corrida, y esto existe
precisamente para no tener que relajarla en los tests.

Las formas internas (bloques, capas, indices) son las de los modelos reales para que
las politicas de congelado se ejerciten igual: wav2vec2 con 12 bloques, DistilBERT
con 6, ViT con 12 sobre imagenes 224x224 con parches de 16.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TOKENIZER = str(ROOT / "src" / "data" / "tokenizers" / "distilbert-base-uncased")


def tiny_audio():
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=12, num_attention_heads=2,
                         intermediate_size=64, conv_dim=(16,) * 7,
                         num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4)
    return Wav2Vec2Model(cfg)


def tiny_text(vocab_size: int = 30522):
    from transformers import DistilBertConfig, DistilBertModel
    cfg = DistilBertConfig(vocab_size=vocab_size, dim=32, n_layers=6, n_heads=2,
                           hidden_dim=64, max_position_embeddings=512)
    return DistilBertModel(cfg)


def tiny_vit():
    from transformers import ViTConfig, ViTModel
    cfg = ViTConfig(hidden_size=32, num_hidden_layers=12, num_attention_heads=2,
                    intermediate_size=64, image_size=224, patch_size=16)
    return ViTModel(cfg)


@contextlib.contextmanager
def tiny_encoders():
    """Sustituye la carga de encoders preentrenados por los diminutos."""
    import src.models.conformer as conformer
    import src.ablation_studies.ablation_11_model as m11
    import src.ablation_studies.unimodal_baseline as uni

    orig_load = conformer._cargar_encoder_preentrenado
    orig_uni = uni._cargar_encoder_preentrenado
    orig_vit = m11.ViTModel

    def fake_load(nombre, que):
        return tiny_audio() if que == "audio" else tiny_text()

    class _FakeViT:
        @staticmethod
        def from_pretrained(*a, **k):
            return tiny_vit()

    conformer._cargar_encoder_preentrenado = fake_load
    uni._cargar_encoder_preentrenado = fake_load
    m11.ViTModel = _FakeViT
    try:
        yield
    finally:
        conformer._cargar_encoder_preentrenado = orig_load
        uni._cargar_encoder_preentrenado = orig_uni
        m11.ViTModel = orig_vit


def offline_config(cfg: dict) -> dict:
    """Tokenizer local en vez de descargarlo."""
    cfg.setdefault("text", {})["model_name"] = LOCAL_TOKENIZER
    return cfg
