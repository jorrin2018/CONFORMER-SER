#!/usr/bin/env python3
"""Auditoria a nivel de archivo del corpus MELD (Poria et al., ACL 2019).

Motivo: MELD no distribuye audio (solo .mp4) y su release tiene defectos
documentados en el issue tracker oficial que NUNCA fueron corregidos, mas
defectos que no estan publicados en ninguna parte. Antes de escribir el loader
hay que saber exactamente que hay en disco.

Fases:
  estructura  CSVs, conteo de clips, diff CSV<->medios, duplicados, etiquetas,
              hablantes, solapes de timestamps. No usa ffprobe: es rapida.
  audio       ffprobe sobre cada .mp4: existencia de pista de audio, codec,
              sample rate, canales y duracion real vs la del CSV.
              Reanudable: guarda checkpoint tras cada bloque.

Uso:
  python scripts/check_meld.py --data-path D:/DOCTORADO/DATASETS/MELD --fase estructura
  python scripts/check_meld.py --data-path D:/DOCTORADO/DATASETS/MELD --fase audio --bloque 2000

El script NO escribe nada dentro del corpus. El checkpoint va a --salida.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- referencias

# Tabla de distribucion de etiquetas publicada (paper + README oficial).
# Si el conteo real no cuadra, la descarga esta incompleta o es otra version.
REF_EMOCION = {
    'neutral':  {'train': 4710, 'dev': 470, 'test': 1256},
    'joy':      {'train': 1743, 'dev': 163, 'test':  402},
    'surprise': {'train': 1205, 'dev': 150, 'test':  281},
    'anger':    {'train': 1109, 'dev': 153, 'test':  345},
    'sadness':  {'train':  683, 'dev': 111, 'test':  208},
    'disgust':  {'train':  271, 'dev':  22, 'test':   68},
    'fear':     {'train':  268, 'dev':  40, 'test':   50},
}
REF_FILAS = {'train': 9989, 'dev': 1109, 'test': 2610}

# Contrato de 4 clases, para alinear con IEMOCAP y CREMA-D.
# OJO: un resultado de 4 clases NO es comparable con la literatura de MELD,
# que reporta weighted F1 sobre las 7 clases.
MAPA_4CLASES = {'anger': 'angry', 'joy': 'happy', 'sadness': 'sad', 'neutral': 'neutral'}

# Rutas relativas a la raiz de MELD.Raw. Los nombres de los directorios de
# medios NO estan documentados en el README (dice literalmente "XXX.tar.gz");
# estos tres estan verificados contra la descarga real.
CFG = {
    'train': ('train/train_sent_emo.csv', 'train/train_splits'),
    'dev':   ('dev_sent_emo.csv',         'dev/dev_splits_complete'),
    'test':  ('test_sent_emo.csv',        'test/output_repeated_splits_test'),
}

PAT_CLIP = re.compile(r'^dia(\d+)_utt(\d+)\.mp4$')
# Prefijo espurio que aparece en 132 archivos del split de test.
PAT_PREFIJADO = re.compile(r'^final_videos_test(dia(\d+)_utt(\d+)\.mp4)$')

TOL_DURACION = 0.06  # s. El redondeo del contenedor mp4 introduce ~22-42 ms.


# ------------------------------------------------------------------ utilidades

def a_segundos(marca: str) -> float:
    """'00:02:42,287' -> 162.287. El separador de ms es coma, estilo SRT."""
    horas, minutos, resto = marca.split(':')
    segundos, milis = resto.split(',')
    return int(horas) * 3600 + int(minutos) * 60 + int(segundos) + int(milis) / 1000


def raiz_corpus(data_path: Path) -> Path:
    """Acepta .../MELD, .../MELD/MELD-RAW o .../MELD-RAW/MELD.Raw."""
    for candidata in (data_path, data_path / 'MELD.Raw',
                      data_path / 'MELD-RAW' / 'MELD.Raw'):
        if (candidata / 'test_sent_emo.csv').is_file():
            return candidata
    raise SystemExit(
        f"No encuentro test_sent_emo.csv bajo {data_path}.\n"
        "Pasa la carpeta que contiene MELD-RAW/MELD.Raw con --data-path."
    )


def leer_csv(ruta: Path) -> list[dict]:
    with open(ruta, encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def clips_en_disco(directorio: Path) -> tuple[dict, dict, list]:
    """Devuelve (clips_normales, clips_prefijados, nombres_no_reconocidos).

    Las claves son (Dialogue_ID, Utterance_ID). Se ignoran los archivos
    AppleDouble '._*' que genera macOS al comprimir, que no son medios.
    """
    normales, prefijados, raros, appledouble = {}, {}, [], 0
    for nombre in sorted(os.listdir(directorio)):
        if nombre.startswith('._'):
            appledouble += 1
            continue
        if not nombre.endswith('.mp4'):
            raros.append(nombre)
            continue
        if (m := PAT_CLIP.match(nombre)):
            normales[(int(m.group(1)), int(m.group(2)))] = nombre
        elif (m := PAT_PREFIJADO.match(nombre)):
            prefijados[(int(m.group(2)), int(m.group(3)))] = nombre
        else:
            raros.append(nombre)
    return normales, prefijados, raros, appledouble


def duracion_mp4(ruta: Path) -> float | None:
    res = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', str(ruta)],
        capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except ValueError:
        return None


def sondear(ruta: Path) -> dict:
    """ffprobe completo de un clip. 'ok' False => ffprobe no pudo leerlo."""
    res = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries',
         'format=duration:stream=codec_type,codec_name,sample_rate,channels',
         '-of', 'json', str(ruta)],
        capture_output=True, text=True)
    if res.returncode != 0:
        return {'ok': False, 'error': res.stderr.strip()[:200]}
    datos = json.loads(res.stdout or '{}')
    pistas = [s for s in datos.get('streams', []) if s.get('codec_type') == 'audio']
    salida = {
        'ok': True,
        'duracion': float(datos.get('format', {}).get('duration') or 0.0),
        'tiene_audio': bool(pistas),
    }
    if pistas:
        salida.update(codec=pistas[0].get('codec_name'),
                      sample_rate=int(pistas[0].get('sample_rate') or 0),
                      canales=int(pistas[0].get('channels') or 0))
    return salida


# --------------------------------------------------------------- fase estructura

def fase_estructura(raiz: Path) -> dict:
    print('=' * 78)
    print('FASE 1 - ESTRUCTURA')
    print('=' * 78)
    informe, todas_filas = {}, {}
    fallos = []

    for split, (rel_csv, rel_dir) in CFG.items():
        ruta_csv, ruta_dir = raiz / rel_csv, raiz / rel_dir
        if not ruta_csv.is_file():
            fallos.append(f'[{split}] falta el CSV: {ruta_csv}')
            continue
        if not ruta_dir.is_dir():
            fallos.append(f'[{split}] falta el directorio de medios: {ruta_dir}')
            continue

        filas = leer_csv(ruta_csv)
        todas_filas[split] = filas
        normales, prefijados, raros, appledouble = clips_en_disco(ruta_dir)
        claves_csv = {(int(f['Dialogue_ID']), int(f['Utterance_ID'])) for f in filas}

        print(f'\n--- {split} ---')
        print(f'  filas CSV           {len(filas)}   (referencia publicada: {REF_FILAS[split]})')
        print(f'  claves unicas CSV   {len(claves_csv)}')
        print(f'  clips diaX_uttY     {len(normales)}')
        print(f'  clips prefijados    {len(prefijados)}   (nombre "final_videos_test...")')
        print(f'  AppleDouble ._*     {appledouble}   (ignorados; basura de macOS)')
        if raros:
            print(f'  NOMBRES NO RECONOCIDOS ({len(raros)}): {raros[:10]}')

        if len(filas) != REF_FILAS[split]:
            fallos.append(f'[{split}] filas CSV {len(filas)} != referencia {REF_FILAS[split]}')
        if len(claves_csv) != len(filas):
            fallos.append(f'[{split}] (Dialogue_ID, Utterance_ID) no es clave unica')

        sin_medio = sorted(claves_csv - set(normales) - set(prefijados))
        sin_fila = sorted(set(normales) - claves_csv)
        print(f'  anotacion SIN clip  {len(sin_medio)}'
              + (f'  -> {["dia%d_utt%d" % k for k in sin_medio[:10]]}' if sin_medio else ''))
        print(f'  clip SIN anotacion  {len(sin_fila)}'
              + (f'  -> {["dia%d_utt%d" % k for k in sin_fila[:10]]}' if sin_fila else ''))

        informe[split] = {
            'filas': len(filas), 'clips': len(normales), 'prefijados': len(prefijados),
            'appledouble': appledouble,
            'sin_medio': ['dia%d_utt%d' % k for k in sin_medio],
            'sin_fila': ['dia%d_utt%d' % k for k in sin_fila],
            'nombres_raros': raros,
        }

        # --- duplicados: dos archivos para la misma utterance -----------------
        dur_csv = {(int(f['Dialogue_ID']), int(f['Utterance_ID'])):
                   a_segundos(f['EndTime']) - a_segundos(f['StartTime']) for f in filas}
        con_dos = sorted(set(normales) & set(prefijados))
        # Una utterance duplicada que no esta en el CSV no se puede arbitrar:
        # no hay anotacion contra la que comparar la duracion.
        duplicadas = [k for k in con_dos if k in dur_csv]
        sin_arbitro = [k for k in con_dos if k not in dur_csv]
        if sin_arbitro:
            print(f'\n  {len(sin_arbitro)} utterances con dos archivos pero SIN fila en el'
                  f' CSV (no arbitrables): {["dia%d_utt%d" % k for k in sin_arbitro]}')
            informe_sin_arbitro = ['dia%d_utt%d' % k for k in sin_arbitro]
        else:
            informe_sin_arbitro = []
        if duplicadas:
            print(f'\n  {len(duplicadas)} utterances anotadas tienen DOS archivos. Arbitro:'
                  f' la duracion del CSV (EndTime - StartTime).')
            gana_pref = gana_norm = ninguno = 0
            detalle = []
            for clave in duplicadas:
                objetivo = dur_csv[clave]
                d_norm = duracion_mp4(ruta_dir / normales[clave])
                d_pref = duracion_mp4(ruta_dir / prefijados[clave])
                if d_norm is None or d_pref is None:
                    ninguno += 1
                    continue
                if abs(d_pref - objetivo) < abs(d_norm - objetivo):
                    gana_pref += 1
                else:
                    gana_norm += 1
                detalle.append({'clip': 'dia%d_utt%d' % clave, 'csv': round(objetivo, 3),
                                'normal': round(d_norm, 3), 'prefijado': round(d_pref, 3)})
            print(f'    coincide con el CSV -> prefijado: {gana_pref}   '
                  f'nombre normal: {gana_norm}   indeterminado: {ninguno}')
            if gana_pref and not gana_norm:
                print('    VEREDICTO: para estas utterances el archivo diaX_uttY.mp4 NO es')
                print('    el corte que describe la anotacion. Hay que leer el prefijado.')
            informe[split]['duplicadas'] = detalle
            informe[split]['duplicadas_sin_arbitro'] = informe_sin_arbitro
            informe[split]['duplicadas_veredicto'] = {
                'prefijado': gana_pref, 'normal': gana_norm, 'indeterminado': ninguno}

    if not todas_filas:
        raise SystemExit('\n'.join(fallos) or 'No se pudo leer ningun split.')

    # --- etiquetas -----------------------------------------------------------
    print('\n--- distribucion de emociones vs tabla publicada ---')
    print(f'{"emocion":10} ' + ' '.join(f'{s:>18}' for s in CFG))
    for emocion in sorted(REF_EMOCION, key=lambda e: -sum(REF_EMOCION[e].values())):
        celdas = []
        for split in CFG:
            esperado = REF_EMOCION[emocion][split]
            real = sum(1 for f in todas_filas.get(split, []) if f['Emotion'] == emocion)
            marca = 'ok' if real == esperado else f'!= {esperado}'
            if real != esperado:
                fallos.append(f'[{split}] {emocion}: {real} != {esperado}')
            celdas.append(f'{real:>7} ({marca:>8})')
        print(f'{emocion:10} ' + ' '.join(f'{c:>18}' for c in celdas))

    etiquetas = {f['Emotion'] for filas in todas_filas.values() for f in filas}
    inesperadas = etiquetas - set(REF_EMOCION)
    if inesperadas:
        fallos.append(f'etiquetas de emocion inesperadas: {sorted(inesperadas)}')

    print('\n--- contrato de 4 clases (anger/joy/sadness/neutral) ---')
    total4 = 0
    for split, filas in todas_filas.items():
        cuenta = collections.Counter(MAPA_4CLASES[f['Emotion']] for f in filas
                                     if f['Emotion'] in MAPA_4CLASES)
        n = sum(cuenta.values())
        total4 += n
        print(f'  {split:6} {n:6}  ' + '  '.join(f'{k}={v}' for k, v in sorted(cuenta.items())))
    print(f'  TOTAL  {total4:6}   descartadas (disgust/fear/surprise): '
          f'{sum(len(f) for f in todas_filas.values()) - total4}')
    informe['total_4clases'] = total4

    # --- hablantes -----------------------------------------------------------
    print('\n--- hablantes: el split oficial NO es independiente de hablante ---')
    por_split = {s: collections.Counter(f['Speaker'] for f in filas)
                 for s, filas in todas_filas.items()}
    lideres = ['Joey', 'Ross', 'Rachel', 'Phoebe', 'Monica', 'Chandler']
    print(f'{"hablante":12} ' + ' '.join(f'{s:>8}' for s in CFG))
    for nombre in lideres:
        print(f'{nombre:12} ' + ' '.join(f'{por_split.get(s, {}).get(nombre, 0):>8}' for s in CFG))
    print(f'{"% 6 lideres":12} ' + ' '.join(
        f'{100 * sum(por_split.get(s, {}).get(n, 0) for n in lideres) / max(1, len(todas_filas.get(s, []))):>7.1f}%'
        for s in CFG))
    if 'train' in por_split and 'test' in por_split:
        comun = set(por_split['train']) & set(por_split['test'])
        print(f'  hablantes compartidos train/test: {len(comun)} de {len(por_split["test"])} en test')
        informe['hablantes_compartidos_train_test'] = len(comun)

    grupales = [n for n in set().union(*por_split.values())
                if ' and ' in n or n in {'All', 'Both', 'Woman', 'Man', 'Guy', 'Girl',
                                         'Director', 'The Director'}]
    n_grupales = sum(c.get(n, 0) for c in por_split.values() for n in grupales)
    print(f'  filas con hablante no-individual (All, "X and Y", Woman...): {n_grupales}')
    print(f'    etiquetas: {sorted(grupales)[:12]}')
    informe['filas_hablante_grupal'] = n_grupales

    # --- duraciones y solapes ------------------------------------------------
    print('\n--- duraciones segun el CSV y solapes dentro del dialogo ---')
    for split, filas in todas_filas.items():
        dur = [a_segundos(f['EndTime']) - a_segundos(f['StartTime']) for f in filas]
        largos = sorted(((d, f"dia{f['Dialogue_ID']}_utt{f['Utterance_ID']}")
                         for d, f in zip(dur, filas) if d > 30), reverse=True)
        print(f'  {split:6} media={statistics.mean(dur):.3f}s  mediana={statistics.median(dur):.3f}s  '
              f'min={min(dur):.3f}s  max={max(dur):.3f}s')
        print(f'         <0.5s: {sum(d < .5 for d in dur)}   <1.0s: {sum(d < 1 for d in dur)}'
              f'   <1.5s: {sum(d < 1.5 for d in dur)}   >30s: {len(largos)} {[n for _, n in largos]}')
        if any(d <= 0 for d in dur):
            cero = [f"dia{f['Dialogue_ID']}_utt{f['Utterance_ID']}"
                    for d, f in zip(dur, filas) if d <= 0]
            print(f'         DURACION CERO O NEGATIVA: {cero}')
            informe.setdefault('duracion_cero', {})[split] = cero

        por_dialogo = collections.defaultdict(list)
        for f in filas:
            por_dialogo[int(f['Dialogue_ID'])].append(
                (int(f['Utterance_ID']), a_segundos(f['StartTime']), a_segundos(f['EndTime'])))
        solapes = 0
        for secuencia in por_dialogo.values():
            secuencia.sort()
            for anterior, actual in zip(secuencia, secuencia[1:]):
                if actual[1] < anterior[2]:
                    solapes += 1
        print(f'         utterances que empiezan antes de que acabe la anterior: {solapes}'
              f' ({100 * solapes / len(filas):.1f}%)')
        informe.setdefault('solapes', {})[split] = solapes
        informe.setdefault('duraciones', {})[split] = {
            'media': round(statistics.mean(dur), 3),
            'mediana': round(statistics.median(dur), 3),
            'menor_1s': sum(d < 1 for d in dur),
            'menor_1_5s': sum(d < 1.5 for d in dur),
            'mayor_30s': [n for _, n in largos],
        }

    informe['fallos'] = fallos
    print('\n' + '=' * 78)
    if fallos:
        print(f'FASE 1: {len(fallos)} DISCREPANCIAS')
        for f in fallos:
            print(f'  - {f}')
    else:
        print('FASE 1 OK: conteos y etiquetas cuadran con la tabla publicada.')
    print('=' * 78)
    return informe


# -------------------------------------------------------------------- fase audio

def fase_audio(raiz: Path, ruta_checkpoint: Path, tam_bloque: int,
               hilos: int) -> dict:
    print('=' * 78)
    print('FASE 2 - AUDIO (ffprobe por clip). Reanudable.')
    print('=' * 78)

    estado = {}
    if ruta_checkpoint.is_file():
        estado = json.loads(ruta_checkpoint.read_text(encoding='utf-8'))
        print(f'Checkpoint cargado: {sum(len(v) for v in estado.values())} clips ya sondeados.')

    pendientes = []
    dur_csv = {}
    for split, (rel_csv, rel_dir) in CFG.items():
        ruta_dir = raiz / rel_dir
        if not ruta_dir.is_dir():
            continue
        filas = leer_csv(raiz / rel_csv)
        for f in filas:
            dur_csv[(split, int(f['Dialogue_ID']), int(f['Utterance_ID']))] = \
                a_segundos(f['EndTime']) - a_segundos(f['StartTime'])
        hechos = estado.setdefault(split, {})
        for nombre in sorted(os.listdir(ruta_dir)):
            if nombre.startswith('._') or not nombre.endswith('.mp4'):
                continue
            if nombre not in hechos:
                pendientes.append((split, ruta_dir / nombre, nombre))

    lote = pendientes[:tam_bloque]
    print(f'Pendientes: {len(pendientes)}. Procesando {len(lote)} con {hilos} hilos.\n')
    # ffprobe es un subproceso: el cuello de botella es el spawn y la E/S del
    # montaje, asi que un pool de hilos escala bien aunque sea Python.
    with cf.ThreadPoolExecutor(max_workers=hilos) as pool:
        futuros = {pool.submit(sondear, ruta): (split, nombre)
                   for split, ruta, nombre in lote}
        for i, fut in enumerate(cf.as_completed(futuros), 1):
            split, nombre = futuros[fut]
            try:
                estado[split][nombre] = fut.result()
            except Exception as exc:  # noqa: BLE001 - se registra, no se aborta
                estado[split][nombre] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
            if i % 1000 == 0:
                print(f'  {i}/{len(lote)}...', flush=True)
                ruta_checkpoint.write_text(json.dumps(estado), encoding='utf-8')
    ruta_checkpoint.write_text(json.dumps(estado), encoding='utf-8')

    restantes = len(pendientes) - min(tam_bloque, len(pendientes))
    if restantes:
        print(f'\nQuedan {restantes} clips. Vuelve a correr la fase audio para continuar.')
        return {'incompleto': True, 'restantes': restantes}

    print('\n--- resumen ---')
    informe = {}
    for split, clips in estado.items():
        ilegibles = [n for n, v in clips.items() if not v.get('ok')]
        sin_audio = [n for n, v in clips.items() if v.get('ok') and not v.get('tiene_audio')]
        formatos = collections.Counter(
            f"{v.get('codec')}@{v.get('sample_rate')}Hz/{v.get('canales')}ch"
            for v in clips.values() if v.get('tiene_audio'))
        print(f'\n[{split}] sondeados={len(clips)}')
        print(f'  ILEGIBLES por ffprobe: {len(ilegibles)} {ilegibles[:10]}')
        print(f'  SIN pista de audio:    {len(sin_audio)} {sin_audio[:10]}')
        print('  formatos de audio:')
        for formato, n in formatos.most_common():
            print(f'    {formato:28} {n}')

        desalineados = []
        for nombre, v in clips.items():
            if not v.get('ok'):
                continue
            m = PAT_CLIP.match(nombre)
            if not m:
                continue
            clave = (split, int(m.group(1)), int(m.group(2)))
            if clave in dur_csv and abs(v['duracion'] - dur_csv[clave]) > TOL_DURACION:
                desalineados.append((round(v['duracion'] - dur_csv[clave], 3), nombre))
        desalineados.sort(key=lambda t: -abs(t[0]))
        print(f'  duracion del archivo discrepa del CSV en >{TOL_DURACION*1000:.0f} ms: '
              f'{len(desalineados)}')
        for dif, nombre in desalineados[:10]:
            print(f'    {nombre:22} {dif:+.3f}s')
        informe[split] = {
            'sondeados': len(clips), 'ilegibles': ilegibles, 'sin_audio': sin_audio,
            'formatos': dict(formatos),
            'desalineados': [{'clip': n, 'dif': d} for d, n in desalineados],
        }
    print('\n' + '=' * 78)
    return informe


# --------------------------------------------------------------------- entrada

def main() -> int:
    ap = argparse.ArgumentParser(description='Auditoria del corpus MELD.')
    ap.add_argument('--data-path', required=True,
                    help='Carpeta MELD (acepta .../MELD o .../MELD-RAW/MELD.Raw)')
    ap.add_argument('--fase', choices=['estructura', 'audio', 'todo'], default='estructura')
    ap.add_argument('--bloque', type=int, default=2000,
                    help='Clips a sondear por corrida en la fase audio')
    ap.add_argument('--hilos', type=int, default=12,
                    help='Sondeos ffprobe en paralelo en la fase audio')
    ap.add_argument('--salida', default='logs/meld_audit.json',
                    help='Donde escribir el informe y el checkpoint')
    args = ap.parse_args()

    raiz = raiz_corpus(Path(args.data_path).expanduser())
    print(f'Corpus: {raiz}\n')

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    # Fusionar con lo que ya hubiera: correr una sola fase no debe borrar el
    # informe de la otra.
    informe = {'raiz': str(raiz)}
    if salida.is_file():
        try:
            informe.update(json.loads(salida.read_text(encoding='utf-8')))
            informe['raiz'] = str(raiz)
        except json.JSONDecodeError:
            pass

    if args.fase in ('estructura', 'todo'):
        informe['estructura'] = fase_estructura(raiz)
    if args.fase in ('audio', 'todo'):
        # Mismo cuidado que en extract_meld_audio.py: si el binario no existe,
        # subprocess lanza FileNotFoundError en vez de devolver codigo de error.
        try:
            disponible = subprocess.run(
                ['ffprobe', '-version'], capture_output=True).returncode == 0
        except (FileNotFoundError, OSError, PermissionError):
            disponible = False
        if not disponible:
            print('ffprobe no esta en el PATH, asi que la fase audio no puede correr.\n'
                  'Hace falta una instalacion completa de ffmpeg (el paquete pip\n'
                  'imageio-ffmpeg trae ffmpeg pero NO ffprobe):\n'
                  '    winget install --id Gyan.FFmpeg -e      # y reabrir la terminal\n'
                  '    conda install -c conda-forge ffmpeg\n'
                  'La fase "estructura" no necesita ffprobe mas que para arbitrar los\n'
                  '131 duplicados de test; el resto funciona sin el.')
            return 2
        informe['audio'] = fase_audio(
            raiz, salida.with_name(salida.stem + '_ffprobe.json'),
            args.bloque, args.hilos)

    salida.write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nInforme escrito en {salida}')
    fallos = informe.get('estructura', {}).get('fallos', [])
    return 1 if fallos else 0


if __name__ == '__main__':
    sys.exit(main())
