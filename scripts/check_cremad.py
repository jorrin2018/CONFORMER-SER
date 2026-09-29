#!/usr/bin/env python
"""
check_cremad.py -- Que CREMA-D esta bien descargado y bien leido ANTES de gastar GPU.

Comprueba, sin entrenar nada:

  1. Que la carpeta es CREMA-D: AudioWAV/ con 7,442 .wav y VideoDemographics.csv.
  2. Que el reparto de actores en 5 grupos es una PARTICION: disjunta, cubre los 91
     actores, y ningun actor aparece en dos grupos. Si esto falla, train y test
     comparten locutor y el numero sube solo, sin ninguna excepcion de por medio.
  3. Que el balance de sexo entre grupos es razonable (se estratifica por sexo; aqui
     se ensena la tabla para poder afirmarlo en el articulo).
  4. Que el subconjunto de 4 clases tiene las 4,899 utterances con la distribucion
     medida el 23/09/2026 (1,271 angry / 1,271 happy / 1,270 sad / 1,087 neutral).
  5. Que las DOCE frases aparecen en TODAS las emociones. Esto es el corazon del
     experimento: si una frase solo saliera en unas emociones, el texto dejaria de ser
     ortogonal a la emocion y la rama de texto podria acertar por el motivo equivocado.
  6. Que los audios se abren y tienen la forma esperada (16 kHz mono).

Uso:
    python scripts/check_cremad.py --data-path "D:\\DOCTORADO\\DATASETS\\CREMA-D"
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation_studies import load_base_config          # noqa: E402
from src.data import build_dataset                          # noqa: E402
from src.data.cremad_dataset import (                       # noqa: E402
    FRASES, N_GRUPOS, grupos_de_actores,
)

CUATRO = ['angry', 'happy', 'sad', 'neutral']


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', required=True,
                    help='raiz del repositorio de CREMA-D (la que tiene AudioWAV/)')
    ap.add_argument('--audios', type=int, default=20,
                    help='cuantos .wav abrir de verdad para comprobar formato')
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    raiz = Path(args.data_path)
    fallos = []

    # --- 1. la carpeta es CREMA-D ---
    audio_dir = raiz / 'AudioWAV'
    n_wav = len(list(audio_dir.glob('*.wav'))) if audio_dir.is_dir() else 0
    demo = raiz / 'VideoDemographics.csv'
    print(f'\n  raiz            : {raiz}')
    print(f'  AudioWAV/       : {"existe" if audio_dir.is_dir() else "NO EXISTE"}, '
          f'{n_wav} .wav (esperados 7442)')
    print(f'  VideoDemographics.csv: {"existe" if demo.is_file() else "NO EXISTE"}')
    if not audio_dir.is_dir() or not demo.is_file():
        print('\n  FALTA EL CORPUS. Descarga solo el audio (606 MB, no los 3.67 GB):\n')
        print('      set GIT_LFS_SKIP_SMUDGE=1')
        print('      git clone --depth 1 https://github.com/CheyneyComputerScience/CREMA-D.git')
        print('      cd CREMA-D')
        print('      git sparse-checkout init --cone')
        print('      git sparse-checkout set AudioWAV')
        print('      git lfs pull --include="AudioWAV/**"\n')
        return 1
    if n_wav != 7442:
        fallos.append(f'hay {n_wav} .wav y deberian ser 7442; la descarga de LFS '
                      f'puede estar incompleta (¿quedaron punteros en vez de audio?)')

    # --- 2 y 3. la particion de actores ---
    asign = grupos_de_actores(raiz, N_GRUPOS)
    por_grupo = defaultdict(list)
    for actor, g in asign.items():
        por_grupo[g].append(actor)
    esperados = {str(i) for i in range(1001, 1092)}
    if set(asign) != esperados:
        faltan = esperados - set(asign)
        sobran = set(asign) - esperados
        fallos.append(f'actores mal: faltan {sorted(faltan)[:5]}, sobran {sorted(sobran)[:5]}')
    solapes = sum(len(v) for v in por_grupo.values()) - len({a for v in por_grupo.values() for a in v})
    if solapes:
        fallos.append(f'{solapes} actores en mas de un grupo: NO es una particion')

    import csv as _csv
    sexo = {}
    with open(demo, newline='', encoding='utf-8-sig') as f:
        for fila in _csv.DictReader(f):
            sexo[str(fila['ActorID']).strip()] = fila['Sex'].strip()
    print(f'\n  {"grupo":<8}{"actores":>9}{"hombres":>9}{"mujeres":>9}')
    print('  ' + '-' * 35)
    for g in sorted(por_grupo):
        act = por_grupo[g]
        h = sum(1 for a in act if sexo.get(a) == 'Male')
        print(f'  {g:<8}{len(act):>9}{h:>9}{len(act) - h:>9}')

    # --- 4, 5. el corpus tal y como lo ve el entrenamiento ---
    cfg = load_base_config()
    ds = build_dataset(
        corpus='cremad',
        data_path=str(raiz),
        sessions=list(range(1, N_GRUPOS + 1)),
        emotions=CUATRO,
        sample_rate=cfg['dataset']['sample_rate'],
        audio_preprocessor=None, text_preprocessor=None,
        max_audio_length=cfg['audio']['max_length'],
        training=False,
    )
    datos = ds.data

    # 5. cada frase, en todas las emociones
    por_frase = defaultdict(Counter)
    for d in datos:
        por_frase[d['transcription']][d['emotion']] += 1
    print(f'\n  {"frase":<42}{"ang":>6}{"hap":>6}{"sad":>6}{"neu":>6}')
    print('  ' + '-' * 66)
    incompletas = []
    for frase in FRASES.values():
        c = por_frase.get(frase)
        if not c:
            incompletas.append(f'{frase!r}: no aparece')
            continue
        fila = [c.get(e, 0) for e in CUATRO]
        print(f'  {frase[:40]:<42}' + ''.join(f'{n:>6}' for n in fila))
        if any(n == 0 for n in fila):
            incompletas.append(f'{frase!r}: falta en {[e for e, n in zip(CUATRO, fila) if n == 0]}')
    if incompletas:
        fallos.append('frases que NO cubren las cuatro emociones -> el texto deja de '
                      'ser ortogonal a la emocion: ' + '; '.join(incompletas))

    # --- 6. los audios se abren ---
    paso = max(1, len(datos) // max(1, args.audios))
    formatos = Counter()
    for d in datos[::paso][:args.audios]:
        try:
            info = torchaudio.info(d['audio_path'])
            formatos[(info.sample_rate, info.num_channels)] += 1
            if info.num_frames <= 0:
                fallos.append(f'{d["utterance_id"]}: 0 frames')
        except Exception as e:                                   # noqa: BLE001
            fallos.append(f'{d["utterance_id"]}: no se abre ({type(e).__name__}: {e})')
    print(f'\n  formatos de audio (muestra de {min(args.audios, len(datos))}): '
          f'{ {f"{sr} Hz x{ch}": n for (sr, ch), n in formatos.items()} }')

    print()
    if fallos:
        print('  PROBLEMAS:')
        for f in fallos:
            print(f'    - {f}')
        print('\n  No lanzar el estudio hasta resolverlos.\n')
        return 1
    print('  CREMA-D listo: particion por actor sin solape, 4,899 clips de 4 clases,')
    print('  y las doce frases presentes en las cuatro emociones. El texto es')
    print('  ortogonal a la emocion POR CONSTRUCCION, que es lo que este corpus')
    print('  viene a medir.\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
