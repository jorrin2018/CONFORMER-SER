#!/usr/bin/env python
"""
check_dataset_refactor.py -- Que el refactor del cargador no cambio NADA en IEMOCAP.

El 22/09/2026 `IEMOCAPDataset` se partio en dos: `BaseSERDataset` (audio, texto, pesos
de clase, filtro de hablante, subconjunto por indices) y la subclase con lo propio de
IEMOCAP (donde estan los wav, como se parsean etiquetas y transcripciones, el contrato
de integridad). El objetivo es poder anadir un segundo corpus para la revista.

Un refactor asi no puede "parecer" correcto. Este repo ya vivio un caso en el que un
troceado de cadena mal hecho tiraba 2,943 de 5,531 utterances SIN AVISAR, dejando solo
las frases guionadas con solapamiento lexico entre train y test. Nada se quejo durante
meses. Por eso aqui no se comprueba que el codigo importe: se compara, registro a
registro, contra una COPIA CONGELADA del archivo anterior al refactor
(tests/fixtures/dataset_pre_refactor.py.frozen), que se carga por ruta como un modulo
aparte y convive en memoria con el nuevo.

Se comparan tres cosas:
  1. La lista de registros completa, en orden, para cada sesion y para las cinco juntas.
  2. Los caminos de filtro de hablante y de subconjunto por indices, que se movieron
     de sitio y son los que mas facil se rompen.
  3. Los tensores que devuelve __getitem__ para una muestra de indices, con los
     preprocesadores reales.

Uso:
    python scripts/check_dataset_refactor.py
    python scripts/check_dataset_refactor.py --items 40
"""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation_studies import load_base_config          # noqa: E402
from src.data import build_dataset                          # noqa: E402
from src.data.preprocessing import AudioPreprocessor, TextPreprocessor  # noqa: E402

CONGELADO = ROOT / 'tests' / 'fixtures' / 'dataset_pre_refactor.py.frozen'
CLAVES = ('audio_path', 'transcription', 'emotion', 'emotion_idx',
          'utterance_id', 'session')


def cargar_congelado():
    """Importa la copia previa al refactor como modulo independiente."""
    if not CONGELADO.is_file():
        raise SystemExit(
            f'No encuentro la copia congelada en {CONGELADO}.\n'
            'Sin ella este test no prueba nada. Recuperala del commit anterior al '
            'refactor: git show <sha>:src/data/dataset.py > '
            'tests/fixtures/dataset_pre_refactor.py.frozen')
    # La copia lleva sufijo .frozen a proposito, para que nadie la importe por
    # accidente ni la recoja un linter. El precio es que spec_from_file_location no
    # reconoce la extension y devuelve None, asi que hay que darle el loader a mano.
    loader = importlib.machinery.SourceFileLoader('dataset_pre_refactor', str(CONGELADO))
    spec = importlib.util.spec_from_file_location(
        'dataset_pre_refactor', str(CONGELADO), loader=loader)
    if spec is None:
        raise SystemExit(f'No pude construir el spec del modulo para {CONGELADO}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['dataset_pre_refactor'] = mod
    spec.loader.exec_module(mod)
    return mod


def comparar_registros(viejo, nuevo, etiqueta):
    """Compara dos listas de registros y devuelve (ok, motivo)."""
    if len(viejo) != len(nuevo):
        return False, f'longitudes distintas: {len(viejo)} vs {len(nuevo)}'
    for i, (a, b) in enumerate(zip(viejo, nuevo)):
        for k in CLAVES:
            if a.get(k) != b.get(k):
                return False, (f'registro {i}, clave {k!r}: '
                               f'{a.get(k)!r} vs {b.get(k)!r}')
    return True, f'{len(nuevo)} registros identicos'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', type=int, default=25,
                    help='cuantos indices comparar con __getitem__ (0 = ninguno)')
    ap.add_argument('--data-path', default=None,
                    help='ruta del corpus. Por defecto la de load_base_config(), que '
                         'sale del checkpoint 20251029_190659 y NO es la que usan los '
                         'YAML de ablacion.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    previo = cargar_congelado()
    cfg = load_base_config()
    ds_cfg, au_cfg = cfg['dataset'], cfg['audio']
    if args.data_path:
        ds_cfg['data_path'] = args.data_path

    # Aviso: hay DOS copias del corpus en disco. load_base_config() lee el config.json
    # del checkpoint 20251029_190659, que apunta a ...\archive (1)\IEMOCAP_full_release,
    # mientras que los YAML de ablacion apuntan a ...\IEMOCAP_full_release. Las dos
    # cargan las 5,531 utterances con los mismos 2,943 improvisados, asi que hoy son
    # equivalentes -- pero este verificador estaria leyendo una copia distinta de la
    # que usan los experimentos. Si alguna vez divergen, esto daria verde sobre el
    # corpus equivocado. Con --data-path se fuerza la de los experimentos.
    yaml_ablacion = ROOT / 'config' / 'ablation'
    rutas_yaml = set()
    for f in sorted(yaml_ablacion.glob('ablation_*.yaml')):
        for linea in f.read_text(encoding='utf-8').splitlines():
            if linea.strip().startswith('data_path:'):
                rutas_yaml.add(linea.split(':', 1)[1].strip().strip('"').replace('\\\\', '\\'))
    if rutas_yaml and ds_cfg['data_path'] not in rutas_yaml:
        print('\n  AVISO: esta verificacion usa una copia del corpus DISTINTA de la que')
        print(f'  usan los YAML de ablacion.')
        print(f'      aqui        : {ds_cfg["data_path"]}')
        for r in sorted(rutas_yaml):
            print(f'      ablaciones  : {r}')
        print('  Hoy las dos cargan lo mismo. Para verificar sobre la de los')
        print('  experimentos: --data-path "<ruta de las ablaciones>"')

    comun = dict(
        data_path=ds_cfg['data_path'],
        emotions=ds_cfg['emotions'],
        sample_rate=ds_cfg['sample_rate'],
        audio_preprocessor=None,
        text_preprocessor=None,
        max_audio_length=au_cfg['max_length'],
        training=False,
    )

    casos = [(f'sesion {s}', dict(comun, sessions=[s])) for s in (1, 2, 3, 4, 5)]
    casos.append(('las cinco sesiones', dict(comun, sessions=[1, 2, 3, 4, 5])))

    # Filtro de hablante y subconjunto por indices: los dos caminos que se movieron.
    sonda = previo.IEMOCAPDataset(**dict(comun, sessions=[1]))
    hablantes = sorted({previo.speaker_id(d['utterance_id']) for d in sonda.data})
    if hablantes:
        casos.append((f'excluye hablante {hablantes[0]}',
                      dict(comun, sessions=[1], exclude_speakers=[hablantes[0]])))
        casos.append((f'solo hablante {hablantes[0]}',
                      dict(comun, sessions=[1], only_speakers=[hablantes[0]])))
    n = len(sonda.data)
    casos.append(('subconjunto por indices',
                  dict(comun, sessions=[1],
                       subset_indices=list(range(0, n, 7)))))

    print(f'\n  copia congelada: {CONGELADO.relative_to(ROOT)}')
    print(f'  corpus: {ds_cfg["data_path"]}\n')
    print(f'  {"caso":<34}{"resultado":<12}detalle')
    print('  ' + '-' * 86)

    ok_total = True
    for etiqueta, kwargs in casos:
        try:
            a = previo.IEMOCAPDataset(**kwargs).data
            b = build_dataset(corpus='iemocap', **kwargs).data
            ok, motivo = comparar_registros(a, b, etiqueta)
        except Exception as e:                                  # noqa: BLE001
            ok, motivo = False, f'excepcion: {type(e).__name__}: {e}'
        ok_total &= ok
        print(f'  {etiqueta:<34}{"IGUAL" if ok else "DISTINTO":<12}{motivo}')

    # __getitem__ con los preprocesadores reales.
    if args.items > 0:
        print()
        try:
            def preps():
                # Mismos argumentos que usa create_dataloaders para el split de
                # test: modo evaluacion, sin aumentacion. Si esto se desviara de
                # train.py el test compararia otra cosa.
                a = AudioPreprocessor(
                    sample_rate=ds_cfg['sample_rate'],
                    use_mel_spectrogram=False,
                    apply_spec_augment=False,
                    apply_waveform_augment=False,
                )
                a.eval()
                t = TextPreprocessor(
                    model_name=cfg['text']['model_name'],
                    max_length=cfg['text']['max_length'],
                    apply_text_augment=False,
                )
                t.eval()
                return a, t
            ap1, tp1 = preps()
            ap2, tp2 = preps()
            kw = dict(comun, sessions=[1])
            a = previo.IEMOCAPDataset(**dict(kw, audio_preprocessor=ap1,
                                             text_preprocessor=tp1))
            b = build_dataset(corpus='iemocap',
                              **dict(kw, audio_preprocessor=ap2,
                                     text_preprocessor=tp2))
            paso = max(1, len(a) // args.items)
            idxs = list(range(0, len(a), paso))[:args.items]
            malos = []
            for i in idxs:
                sa, sb = a[i], b[i]
                if set(sa) != set(sb):
                    malos.append((i, f'claves distintas: {sorted(set(sa) ^ set(sb))}'))
                    continue
                for k in sa:
                    va, vb = sa[k], sb[k]
                    igual = (torch.equal(va, vb) if torch.is_tensor(va) else va == vb)
                    if not igual:
                        malos.append((i, f'clave {k!r} difiere'))
                        break
            if malos:
                ok_total = False
                print(f'  __getitem__: {len(malos)} de {len(idxs)} indices DIFIEREN')
                for i, m in malos[:5]:
                    print(f'      idx {i}: {m}')
            else:
                print(f'  __getitem__: {len(idxs)} indices con tensores identicos')
        except Exception as e:                                   # noqa: BLE001
            ok_total = False
            print(f'  __getitem__: excepcion: {type(e).__name__}: {e}')

    print()
    if ok_total:
        print('  El refactor no cambia lo que se carga. Sigue faltando la prueba')
        print('  definitiva: reproducir un fold del estudio 74 y obtener su mismo')
        print('  numero. Esto descarta que se haya roto la lectura; no que se haya')
        print('  roto algo aguas abajo.\n')
        return 0
    print('  HAY DIFERENCIAS. No seguir hasta entender cual y por que.\n')
    return 1


if __name__ == '__main__':
    sys.exit(main())
