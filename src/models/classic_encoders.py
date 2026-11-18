"""Encoders clasicos: descriptores acusticos (eGeMAPS / ComParE / COVAREP) y GloVe.

POR QUE EXISTEN (estudios 111-119). El aporte del articulo es que la ganancia de la
fusion crece con lo que informa el texto (CREMA-D < IEMOCAP < MELD). Un revisor puede
preguntar si eso es una propiedad de los DATOS o de wav2vec2 + DistilBERT. Repetir el
eje con una familia de encoders completamente distinta -- descriptores acusticos hechos
a mano y vectores de palabra estaticos, la familia de MulT / MCTN / MISA -- separa las
dos cosas. De paso, con ramas mas debiles y mas distintas entre si, la compuerta de
conflicto tiene otra oportunidad (H3).

QUE CAMBIA Y QUE NO. Solo el encoder de cada rama. La fusion, la cabeza, la perdida,
el optimizador de la fusion, los folds y la seleccion de modelo son los del 98. Para
no tocar el resto del codigo, los dos encoders imitan la interfaz de HuggingFace que
ya usan Ablation11Model y UnimodalBaselineModel:

    encoder.config.hidden_size
    audio_encoder(feats, lengths).last_hidden_state   -> (B, 1, H)
    text_encoder(input_ids=, attention_mask=).last_hidden_state -> (B, 1, H)

La secuencia de salida tiene longitud 1 y contiene YA el vector agrupado (media y
maximo enmascarados del BiLSTM). Asi el `mean(dim=1)` del audio y el `[:, 0, :]` del
texto ([CLS]) del modelo existente devuelven exactamente ese vector, sin que el relleno
entre en el promedio.

No hay pesos preentrenados que cargar: el BiLSTM se entrena desde cero y los vectores
GloVe se quedan congelados (como en la literatura clasica). Solo aprenden las filas de
los simbolos especiales ([PAD] no, [UNK], [CLS], [SPEECH], [PROSODY]).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

PREFIJO = "clasico:"

#: Simbolos especiales, en este orden, en las primeras filas del vocabulario.
ESPECIALES = ("[PAD]", "[UNK]", "[CLS]", "[SPEECH]", "[PROSODY]")
PAD, UNK, CLS = 0, 1, 2

# Contracciones al estilo PTB, como estan en GloVe: don't -> do n't, it's -> it 's.
# Partirlas importa: "don't" -> "do" a secas perderia la negacion.
_TOKEN_RE = re.compile(
    r"\[(?:SPEECH|PROSODY)\]|[A-Za-z]+(?=n't\b)|n't\b|'[A-Za-z]+|[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")


def es_clasico(nombre: Optional[str]) -> bool:
    return isinstance(nombre, str) and nombre.startswith(PREFIJO)


def tokenizar(texto: str) -> List[str]:
    """Tokenizacion sencilla, la misma al construir el vocabulario y al entrenar."""
    return _TOKEN_RE.findall(texto or "")


def variantes(token: str) -> List[str]:
    """Formas que se buscan en GloVe, de la mas a la menos especifica."""
    out = [token, token.lower(), token.capitalize()]
    vistos, res = set(), []
    for v in out:
        if v and v not in vistos:
            vistos.add(v)
            res.append(v)
    return res


def cargar_glove(ruta: str) -> Dict:
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe el vocabulario GloVe {ruta}. Generalo con:\n"
            f"  python scripts\\extraer_clasicas.py --study <N> --glove <glove.840B.300d.zip>")
    d = torch.load(ruta, map_location="cpu", weights_only=False)
    if tuple(d["itos"][:len(ESPECIALES)]) != ESPECIALES:
        raise ValueError(f"{ruta}: las primeras filas no son los simbolos especiales")
    return d


class GloveTokenizer:
    """Tokenizer con la interfaz minima que usa TextPreprocessor (llamada y len)."""

    def __init__(self, ruta_glove: str):
        d = cargar_glove(ruta_glove)
        self.itos = list(d["itos"])
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.meta = d.get("meta", {})
        self.n_unk = 0
        self.n_tokens = 0

    def __len__(self) -> int:
        return len(self.itos)

    def ids(self, texto: str) -> List[int]:
        out = [CLS]
        for tok in tokenizar(texto):
            self.n_tokens += 1
            for v in variantes(tok):
                i = self.stoi.get(v)
                if i is not None:
                    out.append(i)
                    break
            else:
                self.n_unk += 1
                out.append(UNK)
        return out

    def __call__(self, texto: str, padding="max_length", truncation=True,
                 max_length: int = 128, return_tensors="pt", **_):
        ids = self.ids(texto)
        if truncation:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length" and len(ids) < max_length:
            mask += [0] * (max_length - len(ids))
            ids += [PAD] * (max_length - len(ids))
        return {"input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([mask], dtype=torch.long)}


def _media_max(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, T, D), (B, T) -> (B, 2D): media y maximo solo sobre posiciones reales."""
    m = mask.unsqueeze(-1).to(seq.dtype)
    media = (seq * m).sum(1) / m.sum(1).clamp(min=1.0)
    maximo = seq.masked_fill(m == 0, torch.finfo(seq.dtype).min).max(1).values
    return torch.cat([media, maximo], dim=-1)


class _BiLSTMPool(nn.Module):
    def __init__(self, in_dim: int, d_model: int, lstm_hidden: int, lstm_layers: int,
                 dropout: float):
        super().__init__()
        self.entrada = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model),
                                     nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.salida = nn.Dropout(dropout)
        self.hidden_size = 4 * lstm_hidden  # media + max de un BiLSTM

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        lengths = lengths.clamp(min=1, max=T)
        h = self.entrada(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.to(x.device).unsqueeze(1)
        return self.salida(_media_max(out, mask))


class ClassicAudioEncoder(nn.Module):
    """Descriptores por frame (B, T, F) -> BiLSTM -> media y maximo enmascarados."""

    es_clasico = True

    def __init__(self, in_dim: int, d_model: int = 256, lstm_hidden: int = 256,
                 lstm_layers: int = 2, dropout: float = 0.3, fuente: str = "?"):
        super().__init__()
        self.red = _BiLSTMPool(in_dim, d_model, lstm_hidden, lstm_layers, dropout)
        self.config = SimpleNamespace(hidden_size=self.red.hidden_size, fuente=fuente,
                                      in_dim=in_dim)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> BaseModelOutput:
        if feats.dim() != 3:
            raise RuntimeError(
                "ClassicAudioEncoder recibe descriptores (B, T, F), no la forma de onda. "
                "El batch tiene que traer 'audio_feats' (ver src/data/classic_features.py).")
        return BaseModelOutput(last_hidden_state=self.red(feats, lengths).unsqueeze(1))


class ClassicTextEncoder(nn.Module):
    """GloVe congelado (+ simbolos especiales entrenables) -> BiLSTM -> media y maximo."""

    es_clasico = True

    def __init__(self, vectores: torch.Tensor, d_model: int = 256, lstm_hidden: int = 256,
                 lstm_layers: int = 2, dropout: float = 0.3,
                 finetune_embeddings: bool = False):
        super().__init__()
        V, E = vectores.shape
        self.n_especiales = len(ESPECIALES)
        self.palabras = nn.Embedding.from_pretrained(vectores.float(), freeze=not finetune_embeddings,
                                                     padding_idx=PAD)
        self.especiales = nn.Embedding(self.n_especiales, E)
        nn.init.normal_(self.especiales.weight, 0.0, 0.1)
        self.red = _BiLSTMPool(E, d_model, lstm_hidden, lstm_layers, dropout)
        self.config = SimpleNamespace(hidden_size=self.red.hidden_size, vocab_size=V,
                                      embedding_dim=E)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                **_) -> BaseModelOutput:
        es_esp = (input_ids < self.n_especiales).unsqueeze(-1)
        emb = torch.where(es_esp,
                          self.especiales(input_ids.clamp(max=self.n_especiales - 1)),
                          self.palabras(input_ids))
        lengths = attention_mask.sum(1).to(torch.long)
        return BaseModelOutput(last_hidden_state=self.red(emb, lengths).unsqueeze(1))


def _cfg_clasico(config: Dict) -> Dict:
    c = ((config or {}).get("model", {}) or {}).get("clasico")
    if not c:
        raise ValueError("Encoder 'clasico:*' sin bloque model.clasico en el config")
    return c


def meta_audio(ruta_cache: str) -> Dict:
    ruta = Path(ruta_cache).with_suffix(".json")
    if not ruta.is_file():
        raise FileNotFoundError(
            f"Falta {ruta} (metadatos de {ruta_cache}). Genera el cache con:\n"
            f"  python scripts\\extraer_clasicas.py --study <N>")
    return json.loads(ruta.read_text(encoding="utf-8"))


def build_classic_encoder(nombre: str, que: str, config: Dict) -> nn.Module:
    """'clasico:lld' (audio) o 'clasico:glove' (texto), con las dimensiones de model.clasico."""
    c = _cfg_clasico(config)
    e = c.get("encoder", {}) or {}
    kw = dict(d_model=int(e.get("d_model", 256)), lstm_hidden=int(e.get("lstm_hidden", 256)),
              lstm_layers=int(e.get("lstm_layers", 2)), dropout=float(e.get("dropout", 0.3)))
    tipo = nombre[len(PREFIJO):]
    if que == "audio":
        if tipo != "lld":
            raise ValueError(f"audio_encoder {nombre!r}: el unico encoder clasico de audio es 'clasico:lld'")
        meta = meta_audio(c["audio_cache"])
        enc = ClassicAudioEncoder(in_dim=int(meta["dim"]), fuente=meta.get("fuente", "?"), **kw)
        print(f"[clasico] audio: {meta.get('fuente')} ({meta['dim']} descriptores/frame, "
              f"{meta.get('fps')} fps) -> BiLSTM {kw['lstm_layers']}x{kw['lstm_hidden']} "
              f"-> {enc.config.hidden_size}")
        return enc
    if tipo != "glove":
        raise ValueError(f"text_encoder {nombre!r}: el unico encoder clasico de texto es 'clasico:glove'")
    d = cargar_glove(c["glove_cache"])
    enc = ClassicTextEncoder(d["vectors"], finetune_embeddings=bool(e.get("finetune_embeddings", False)),
                             **kw)
    m = d.get("meta", {})
    print(f"[clasico] texto: GloVe {m.get('fuente', '?')} ({enc.config.vocab_size} palabras, "
          f"{enc.config.embedding_dim}-d, cobertura {m.get('cobertura_tokens', '?')}) "
          f"-> BiLSTM -> {enc.config.hidden_size}")
    return enc
