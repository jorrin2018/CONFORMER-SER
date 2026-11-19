"""Per-machine dataset paths.

The YAML configs carry the dataset path of the machine where they were written
(``D:\\DOCTORADO\\DATASETS\\...``). Rather than editing every tracked YAML on each
machine, an optional, gitignored ``config/local_paths.yaml`` maps corpus -> path:

    datasets:
      iemocap: "F:\\DOCTORADO\\DATASETS\\EMOTION\\IEMOCAP\\archive\\IEMOCAP_full_release"

``apply_local_paths(config)`` replaces ``config['dataset']['data_path']`` when the
config's corpus has an entry. It runs at config load time, so the ``config.json``
written into each run directory records the path that was actually read.

Without the file nothing changes. An entry pointing to a directory that does not
exist raises instead of silently falling back to the YAML path.
"""

from pathlib import Path
from typing import Dict, Optional

import yaml

LOCAL_PATHS_FILE = Path(__file__).resolve().parents[2] / 'config' / 'local_paths.yaml'
DEFAULT_CORPUS = 'iemocap'

_cache: Optional[Dict[str, str]] = None
_announced = set()


def _load_local_paths() -> Dict[str, str]:
    global _cache
    if _cache is None:
        _cache = {}
        if LOCAL_PATHS_FILE.is_file():
            with open(LOCAL_PATHS_FILE, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            for corpus, path in (data.get('datasets') or {}).items():
                if path:
                    _cache[str(corpus).strip().lower()] = str(path)
    return _cache


def apply_local_paths(config: Dict) -> Dict:
    """Override dataset.data_path from config/local_paths.yaml, in place. Returns config."""
    dataset = config.get('dataset')
    if not isinstance(dataset, dict):
        return config
    corpus = (dataset.get('corpus') or DEFAULT_CORPUS).strip().lower()
    local = _load_local_paths().get(corpus)
    if local is None:
        return config
    if not Path(local).is_dir():
        raise FileNotFoundError(
            f"{LOCAL_PATHS_FILE.name}: la ruta de '{corpus}' no existe: {local}"
        )
    previous = dataset.get('data_path')
    dataset['data_path'] = local
    if previous != local and corpus not in _announced:
        _announced.add(corpus)
        print(f"[local_paths] {corpus}: data_path -> {local} (config: {previous})")
    return config
