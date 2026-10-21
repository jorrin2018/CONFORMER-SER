#!/usr/bin/env python3
"""Extrae el audio de MELD de los .mp4 a .wav 16 kHz mono.

POR QUE HACE FALTA. MELD **no distribuye audio**: el release son 13,848 ficheros
.mp4 de 10.9 GB. El paper dice que los autores lo pasaron a "16-bit PCM WAV"
pero no libera esos WAV. Hay que extraerlo, y hay tres trampas verificadas sobre
la descarga real el 10/10/2026 (ver docs/REWRITE_NOTES.md seccion 12):

1. EL NUMERO DE CANALES NO ES UNIFORME. De los 13,847 clips legibles, 12,230 son
   6 canales (5.1) y 1,617 son estereo. Un ffmpeg que asuma 5.1 falla en 1,617
   ficheros. Aqui se fuerza `-ac 1`, que hace downmix desde el layout que sea.

2. EN TEST, PARA 131 UTTERANCES EL FICHERO CORRECTO NO ES EL QUE DICE EL NOMBRE.
   El directorio de test trae 132 ficheros con el nombre prefijado
   `final_videos_testdiaX_uttY.mp4` ADEMAS del `diaX_uttY.mp4`. Arbitrando con la
   duracion del CSV, 131 de 131 coinciden con el prefijado y 0 con el normal, y
   coinciden al milisegundo; el de nombre normal es mas largo y esta COMPARTIDO
   entre utterances vecinas (dia100_utt5/6/7 apuntan al mismo fichero de 4.233 s).
   El nombre del directorio lo dice: output_repeated_splits_test.
   Como el read_meld.py oficial construye el nombre desde el CSV, todo pipeline
   publicado lee el corte equivocado en el 5 % del test. Aqui, si existe el
   prefijado, SE LEE EL PREFIJADO.

3. UN FICHERO ESTA CORRUPTO: train_splits/dia125_utt3.mp4 (issue #39 del repo
   oficial, nunca corregido). Se descarta y se reporta; no se deja que ffmpeg
   escriba un wav vacio que luego entrene en silencio.

Salida: <raiz de MELD>/audio_16k/{train,dev,test}/diaX_uttY.wav
Es reanudable: salta los wav que ya existen con tamano > 0.

Uso:
  python scripts/extract_meld_audio.py --data-path D:/DOCTORADO/DATASETS/MELD
  python scripts/extract_meld_audio.py --data-path ... --hilos 16 --max 3000
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

CFG = {
    'train': ('train/train_sent_emo.csv', 'train/train_splits'),
    'dev':   ('dev_sent_emo.csv',         'dev/dev_splits_complete'),
    'test':  ('test_sent_emo.csv',        'test/output_repeated_splits_test'),
}

#: Unico fichero ilegible de todo el corpus. Verificado con ffprobe sobre los
#: 13,848: 0 clips sin pista de audio, 1 ilegible, y es este.
CORRUPTOS = {('train', 'dia125_utt3')}

SR = 16000

#: Ruta del binario de ffmpeg que se usara. La resuelve `localizar_ffmpeg()`.
FFMPEG = 'ffmpeg'


def localizar_ffmpeg(explicito: str | None = None) -> str:
    """Devuelve la ruta de un ffmpeg utilizable, o falla explicando como instalarlo.

    Busca en este orden: lo que diga --ffmpeg, el PATH, y el binario que trae el
    paquete `imageio-ffmpeg` (que se instala con pip DENTRO del venv, sin permisos
    de administrador y sin tocar el PATH, que es la via mas simple en Windows).

    Existe porque la version anterior comprobaba `subprocess.run(['ffmpeg', ...])`
    y, cuando el binario no existe, eso NO devuelve un codigo de error: lanza
    FileNotFoundError. El resultado era un traceback en vez de un mensaje util.
    """
    candidatos = []
    if explicito:
        candidatos.append(explicito)
    hallado = shutil.which('ffmpeg')
    if hallado:
        candidatos.append(hallado)
    try:
        import imageio_ffmpeg
        candidatos.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001 - el paquete es opcional
        pass

    for ruta in candidatos:
        try:
            if subprocess.run([ruta, '-version'],
                              capture_output=True).returncode == 0:
                return ruta
        except (FileNotFoundError, OSError, PermissionError):
            continue

    raise SystemExit(
        'No encuentro un ffmpeg utilizable, y MELD no distribuye audio: sin ffmpeg no\n'
        'hay nada que extraer.\n\n'
        'La via mas simple, dentro del venv y sin permisos de administrador:\n'
        '    pip install imageio-ffmpeg\n'
        'Este script lo detecta solo; no hace falta tocar el PATH.\n\n'
        'Instalacion completa en el sistema (da tambien ffprobe, que necesita\n'
        'scripts/check_meld.py --fase audio):\n'
        '    winget install --id Gyan.FFmpeg -e        # y reabrir la terminal\n'
        '    conda install -c conda-forge ffmpeg       # si trabajas en conda\n\n'
        'O pasa la ruta a mano:  --ffmpeg "C:/ruta/a/ffmpeg.exe"'
    )


def raiz_corpus(data_path: Path) -> Path:
    for candidata in (data_path, data_path / 'MELD.Raw',
                      data_path / 'MELD-RAW' / 'MELD.Raw'):
        if (candidata / 'test_sent_emo.csv').is_file():
            return candidata
    raise SystemExit(f'No encuentro test_sent_emo.csv bajo {data_path}.')


def planificar(raiz: Path, destino: Path) -> tuple[list, dict]:
    """Devuelve (tareas, resumen). Una tarea por utterance ANOTADA."""
    tareas, resumen = [], {}
    for split, (rel_csv, rel_dir) in CFG.items():
        dir_medios = raiz / rel_dir
        with open(raiz / rel_csv, encoding='utf-8') as fh:
            filas = list(csv.DictReader(fh))
        existentes = set(os.listdir(dir_medios))
        salida_split = destino / split
        salida_split.mkdir(parents=True, exist_ok=True)

        n_pref = n_corrupto = n_ausente = n_hecho = 0
        for fila in filas:
            uid = f"dia{fila['Dialogue_ID']}_utt{fila['Utterance_ID']}"
            if (split, uid) in CORRUPTOS:
                n_corrupto += 1
                continue
            prefijado = f'final_videos_test{uid}.mp4'
            if prefijado in existentes:
                origen, n_pref = dir_medios / prefijado, n_pref + 1
            elif f'{uid}.mp4' in existentes:
                origen = dir_medios / f'{uid}.mp4'
            else:
                n_ausente += 1
                continue
            wav = salida_split / f'{uid}.wav'
            if wav.is_file() and wav.stat().st_size > 44:
                n_hecho += 1
                continue
            tareas.append((split, uid, origen, wav))
        resumen[split] = {'filas': len(filas), 'prefijado': n_pref,
                          'corrupto': n_corrupto, 'sin_fichero': n_ausente,
                          'ya_extraido': n_hecho}
    return tareas, resumen


def extraer(tarea) -> tuple[str, str, str | None]:
    split, uid, origen, wav = tarea
    # -ac 1 hace el downmix desde 5.1 o estereo indistintamente; -ar 16000 el
    # resample; -vn descarta el video. -nostdin para que no se coma la entrada.
    res = subprocess.run(
        [FFMPEG, '-nostdin', '-v', 'error', '-y', '-i', str(origen),
         '-vn', '-ac', '1', '-ar', str(SR), '-c:a', 'pcm_s16le', str(wav)],
        capture_output=True, text=True)
    if res.returncode != 0 or not (wav.is_file() and wav.stat().st_size > 44):
        return split, uid, (res.stderr.strip()[:160] or 'wav vacio')
    return split, uid, None


def main() -> int:
    ap = argparse.ArgumentParser(description='Extrae el audio de MELD a wav 16 kHz mono.')
    ap.add_argument('--data-path', required=True)
    ap.add_argument('--destino', default=None,
                    help='Por defecto <raiz de MELD>/audio_16k')
    ap.add_argument('--hilos', type=int, default=12)
    ap.add_argument('--max', type=int, default=0,
                    help='Procesar como maximo N clips en esta corrida (0 = todos)')
    ap.add_argument('--ffmpeg', default=None,
                    help='Ruta al binario de ffmpeg, si no esta en el PATH')
    args = ap.parse_args()

    global FFMPEG
    FFMPEG = localizar_ffmpeg(args.ffmpeg)
    print(f'ffmpeg:  {FFMPEG}')

    raiz = raiz_corpus(Path(args.data_path).expanduser())
    destino = Path(args.destino) if args.destino else raiz.parent.parent / 'audio_16k'
    print(f'Origen:  {raiz}')
    print(f'Destino: {destino}\n')

    tareas, resumen = planificar(raiz, destino)
    for split, r in resumen.items():
        print(f"[{split}] filas={r['filas']}  ya extraidos={r['ya_extraido']}  "
              f"desde fichero PREFIJADO={r['prefijado']}  "
              f"corruptos descartados={r['corrupto']}  sin fichero={r['sin_fichero']}")
    print(f'\nPendientes: {len(tareas)}')
    if not tareas:
        print('Nada que hacer.')
        return 0

    lote = tareas[:args.max] if args.max else tareas
    print(f'Extrayendo {len(lote)} con {args.hilos} hilos.\n')
    fallos = []
    with cf.ThreadPoolExecutor(max_workers=args.hilos) as pool:
        for i, (split, uid, err) in enumerate(
                pool.map(extraer, lote), 1):
            if err:
                fallos.append((split, uid, err))
            if i % 500 == 0:
                print(f'  {i}/{len(lote)}  (fallos: {len(fallos)})', flush=True)

    print(f'\nHechos {len(lote) - len(fallos)}/{len(lote)}.')
    if fallos:
        print(f'FALLOS: {len(fallos)}')
        for split, uid, err in fallos[:20]:
            print(f'  [{split}] {uid}: {err}')
        return 1
    restantes = len(tareas) - len(lote)
    if restantes:
        print(f'Quedan {restantes}. Vuelve a correr para continuar.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
