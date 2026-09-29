"""
Registro de corpus. Un unico sitio donde se decide que clase lee que corpus.

POR QUE. Hasta el 22/09/2026 `IEMOCAPDataset` se instanciaba directamente en nueve
sitios (src/train.py x4, src/evaluate.py, src/evaluate_ensemble.py,
scripts/run_ablation.py, scripts/check_random_protocol.py, scripts/fit_speechcue_stats.py,
scripts/smoke_dataset.py). Anadir un segundo corpus significaba tocar los nueve y
esperar no olvidarse de ninguno. Ahora todos piden `build_dataset(corpus=...)` y el
corpus se elige en el YAML con `dataset.corpus`.

El valor por defecto es 'iemocap' a proposito: todos los configs que ya existen no
declaran `dataset.corpus` y tienen que seguir cargando exactamente lo mismo que antes.

COMO ANADIR UN CORPUS:

  1. Escribir la clase en src/data/, heredando de BaseSERDataset, implementando
     _collect_records() y -- si hay cifras de referencia publicadas -- _assert_integrity().
  2. Registrarla en CORPORA, abajo.
  3. Poner `corpus: <nombre>` en el config del estudio.

La clase nueva tiene que aceptar la MISMA firma de constructor y devolver los mismos
dicts, porque train.py le pasa los mismos argumentos a los tres splits.
"""
from typing import Dict, Type

from src.data.base_dataset import BaseSERDataset
from src.data.cremad_dataset import CremaDDataset
from src.data.dataset import IEMOCAPDataset

#: nombre en el YAML -> clase. Las claves se comparan en minusculas.
CORPORA: Dict[str, Type[BaseSERDataset]] = {
    'iemocap': IEMOCAPDataset,
    'cremad': CremaDDataset,
    'crema-d': CremaDDataset,
}

DEFAULT_CORPUS = 'iemocap'


def get_dataset_class(corpus: str = DEFAULT_CORPUS) -> Type[BaseSERDataset]:
    """Devuelve la clase registrada para `corpus`, o falla nombrando las que hay."""
    clave = (corpus or DEFAULT_CORPUS).strip().lower()
    if clave not in CORPORA:
        raise KeyError(
            f"corpus '{corpus}' no esta registrado. Disponibles: "
            f"{sorted(CORPORA)}. Se registran en src/data/registry.py."
        )
    return CORPORA[clave]


def build_dataset(corpus: str = DEFAULT_CORPUS, **kwargs) -> BaseSERDataset:
    """Construye el dataset del corpus pedido con los argumentos de siempre."""
    return get_dataset_class(corpus)(**kwargs)


def corpus_from_config(config: Dict) -> str:
    """Lee `dataset.corpus` del config. Ausente -> 'iemocap', que es lo historico."""
    return (config.get('dataset', {}) or {}).get('corpus', DEFAULT_CORPUS)
