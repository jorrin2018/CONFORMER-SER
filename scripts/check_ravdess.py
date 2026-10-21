#!/usr/bin/env python
"""
check_ravdess.py -- Verifica la copia de RAVDESS en disco contra el diseno publicado.

POR QUE EXISTE. El contrato de RAVDESS en src/data/ravdess_dataset.py esta DERIVADO del
diseno publicado (Livingstone & Russo, PLoS ONE 13(5): e0196391, 2018), no medido sobre
una descarga. Este script es lo que convierte "derivado" en "verificado". En un repo
donde un bug silencioso tiro 2,943 de 5,531 utterances durante meses, un corpus nuevo no
entra sin un verificador que lo mire fichero a fichero ANTES de gastar GPU.

Lo que comprueba, en este orden:
  1. Que la carpeta existe y tiene las 24 carpetas Actor_NN.
  2. Que TODOS los nombres casan el patron de siete campos, y nombra los que no.
  3. El diseno factorial completo por actor: 60 ensayos de habla, con neutral a 4
     (una sola intensidad) y las otras siete a 8. Nombra los huecos.
  4. Los totales: 1,440 de habla y 672 en el subconjunto de 4 clases del eje.
  5. Que neutral NO tiene intensidad strong (RAVDESS no la grabo).
  6. Los parametros reales del audio con --fase audio: sample rate, canales, bits y
     duracion. El diseno dice 48 kHz; si la copia viene remuestreada, hay que saberlo.
  7. Que el reparto en 5 grupos de actores es una particion y esta estratificado por sexo.

Uso:
    python scripts/check_ravdess.py --data-path "D:/DOCTORADO/DATASETS/RAVDESS"
    python scripts/check_ravdess.py --data-path "..." --fase audio
    python scripts/check_ravdess.py --data-path "..." --fase todo

Salida: codigo 0 si todo cuadra, 1 si hay algo que revisar. El detalle va a stdout y,
con --json <ruta>, tambien a un informe.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ravdess_dataset import (  # noqa: E402
    FRASES,
    MAPA_EMOCION,
    N_ACTORES,
    N_GRUPOS,
    PATRON,
    RavdessDataset,
    grupos_de_actores,
    sexo_de_actor,
)

CANAL_HABLA = "01"
INTENSIDAD_STRONG = "02"
#: Ensayos de habla por actor y emocion, del diseno publicado.
#: neutral: 1 intensidad x 2 frases x 2 repeticiones = 4. Las otras: 2 x 2 x 2 = 8.
ESPERADO_POR_ACTOR_EMOCION = {cod: (4 if cod == "01" else 8) for cod in MAPA_EMOCION}


def localizar(raiz: Path) -> Path:
    return RavdessDataset._localizar_raiz_audio(raiz)


def fase_estructura(base: Path, problemas: list, informe: dict) -> None:
    print("\n" + "=" * 72)
    print("FASE 1: estructura y diseno factorial")
    print("=" * 72)

    actores_dir = sorted(p.name for p in base.glob("Actor_*") if p.is_dir())
    print(f"  carpetas Actor_*: {len(actores_dir)}")
    informe["carpetas_actor"] = len(actores_dir)
    if len(actores_dir) != N_ACTORES:
        problemas.append(
            f"hay {len(actores_dir)} carpetas Actor_* y deberian ser {N_ACTORES}")

    raros: list = []
    canto = 0
    # (actor, emocion) -> n   y   (actor, emocion, intensidad) -> n
    cuenta = collections.Counter()
    neutral_strong: list = []
    por_emocion = collections.Counter()
    frases_vistas = set()
    total_habla = 0

    for wav in sorted(base.rglob("*.wav")):
        m = PATRON.match(wav.stem)
        if not m:
            raros.append(wav.name)
            continue
        c = m.groupdict()
        if c["canal"] != CANAL_HABLA:
            canto += 1
            continue
        total_habla += 1
        cuenta[(c["actor"], c["emocion"])] += 1
        frases_vistas.add(c["frase"])
        emo = MAPA_EMOCION.get(c["emocion"])
        if emo:
            por_emocion[emo] += 1
        if c["emocion"] == "01" and c["intensidad"] == INTENSIDAD_STRONG:
            neutral_strong.append(wav.name)

    print(f"  .wav de habla       : {total_habla}  (esperado {RavdessDataset.REF_TOTAL_HABLA})")
    print(f"  .wav de canto       : {canto}  (se ignoran: decision 1)")
    print(f"  nombres fuera de patron: {len(raros)}")
    print(f"  frases distintas    : {len(frases_vistas)}  (esperado {len(FRASES)})")
    informe.update(total_habla=total_habla, canto=canto, fuera_de_patron=len(raros),
                   frases=len(frases_vistas))

    if raros:
        problemas.append(f"{len(raros)} ficheros no casan el patron: {raros[:10]}")
    if total_habla != RavdessDataset.REF_TOTAL_HABLA:
        problemas.append(
            f"{total_habla} .wav de habla, esperados {RavdessDataset.REF_TOTAL_HABLA}")
    if len(frases_vistas) != len(FRASES):
        problemas.append(
            f"{len(frases_vistas)} frases distintas, esperadas {len(FRASES)}")
    if neutral_strong:
        problemas.append(
            f"{len(neutral_strong)} clips de neutral con intensidad strong, que RAVDESS "
            f"no tiene por diseno: {neutral_strong[:10]}")

    # Diseno factorial, actor a actor. Nombrar los huecos, no solo contarlos.
    huecos = []
    for i in range(1, N_ACTORES + 1):
        actor = f"{i:02d}"
        for cod, esperado in sorted(ESPERADO_POR_ACTOR_EMOCION.items()):
            visto = cuenta[(actor, cod)]
            if visto != esperado:
                huecos.append({
                    "actor": actor, "emocion": MAPA_EMOCION[cod],
                    "visto": visto, "esperado": esperado,
                })
    informe["huecos_diseno"] = huecos
    if huecos:
        print(f"  HUECOS en el diseno factorial: {len(huecos)}")
        for h in huecos[:15]:
            print(f"      Actor_{h['actor']}  {h['emocion']:<10} "
                  f"{h['visto']} de {h['esperado']}")
        if len(huecos) > 15:
            print(f"      ... y {len(huecos) - 15} mas")
        problemas.append(f"{len(huecos)} celdas del diseno factorial no cuadran")
    else:
        print("  diseno factorial    : completo, 60 ensayos de habla por actor")

    print("\n  por emocion (habla, todas las clases):")
    for emo in sorted(por_emocion):
        print(f"      {emo:<12}{por_emocion[emo]:>6}")
    informe["por_emocion"] = dict(por_emocion)

    # Subconjunto de 4 clases del eje.
    cuatro = {k: por_emocion.get(k, 0) for k in RavdessDataset.REF_4CLASS_POR_EMOCION}
    total4 = sum(cuatro.values())
    print(f"\n  subconjunto de 4 clases: {total4}  "
          f"(esperado {RavdessDataset.REF_4CLASS_TOTAL})")
    print(f"      {cuatro}")
    print(f"      esperado: {RavdessDataset.REF_4CLASS_POR_EMOCION}")
    informe["cuatro_clases"] = {"total": total4, "por_emocion": cuatro}
    if total4 != RavdessDataset.REF_4CLASS_TOTAL:
        problemas.append(
            f"subconjunto de 4 clases: {total4}, esperado "
            f"{RavdessDataset.REF_4CLASS_TOTAL}")
    if cuatro != RavdessDataset.REF_4CLASS_POR_EMOCION:
        problemas.append(
            f"distribucion de 4 clases {cuatro}, esperada "
            f"{RavdessDataset.REF_4CLASS_POR_EMOCION}")


def fase_folds(problemas: list, informe: dict) -> None:
    print("\n" + "=" * 72)
    print("FASE 2: reparto en grupos de actores")
    print("=" * 72)
    asignacion = grupos_de_actores(N_GRUPOS)
    por_grupo = collections.defaultdict(list)
    for actor, g in asignacion.items():
        por_grupo[g].append(actor)

    if len(asignacion) != N_ACTORES:
        problemas.append(f"el reparto cubre {len(asignacion)} actores de {N_ACTORES}")
    vistos = sorted(a for lista in por_grupo.values() for a in lista)
    if len(vistos) != len(set(vistos)):
        problemas.append("el reparto NO es una particion: hay actores repetidos")

    for g in sorted(por_grupo):
        lista = sorted(por_grupo[g])
        m = sum(1 for a in lista if sexo_de_actor(a) == "M")
        f = len(lista) - m
        print(f"  grupo {g}: {len(lista)} actores (M={m}, F={f})  {lista}")
        if abs(m - f) > 1:
            problemas.append(
                f"grupo {g} desequilibrado por sexo: M={m}, F={f}")
    informe["grupos"] = {str(g): sorted(v) for g, v in por_grupo.items()}
    print("  NOTA: 24 no es divisible por 5, asi que los grupos salen 6/6/4/4/4. "
          "Los folds no pesan lo mismo; declararlo.")


def fase_audio(base: Path, problemas: list, informe: dict, limite: int) -> None:
    print("\n" + "=" * 72)
    print("FASE 3: parametros reales del audio")
    print("=" * 72)
    try:
        import torchaudio
    except ImportError:
        print("  torchaudio no disponible: se omite.")
        return

    tasas = collections.Counter()
    canales = collections.Counter()
    duraciones = []
    vacios = []
    #: Ficheros con mas de un canal. RAVDESS trae cinco dual-mono entre 1,440 (hallazgo
    #: propio, 2026-10-20). Se nombran y se comprueba que los canales sigan siendo
    #: identicos: si algun dia difieren, el downmix estaria mezclando dos senales
    #: distintas y eso si habria que tratarlo.
    multicanal = []
    canales_distintos = []
    leidos = 0
    for wav in sorted(base.rglob("*.wav")):
        m = PATRON.match(wav.stem)
        if not m or m.groupdict()["canal"] != CANAL_HABLA:
            continue
        if limite and leidos >= limite:
            break
        try:
            info = torchaudio.info(str(wav))
        except Exception as e:  # noqa: BLE001
            problemas.append(f"no se puede leer {wav.name}: {e}")
            continue
        leidos += 1
        tasas[info.sample_rate] += 1
        canales[info.num_channels] += 1
        if info.num_channels > 1:
            multicanal.append(f"{wav.parent.name}/{wav.name}")
            try:
                import torch
                w, _ = torchaudio.load(str(wav))
                if not bool(torch.equal(w[0], w[1])):
                    canales_distintos.append(
                        f"{wav.parent.name}/{wav.name} "
                        f"(max|L-R|={float((w[0] - w[1]).abs().max()):.6g})")
            except Exception as e:  # noqa: BLE001
                problemas.append(f"no se pueden comparar los canales de {wav.name}: {e}")
        dur = info.num_frames / info.sample_rate if info.sample_rate else 0.0
        duraciones.append(dur)
        if dur < 0.5:
            vacios.append((wav.name, round(dur, 3)))

    print(f"  ficheros leidos: {leidos}")
    print(f"  sample rate    : {dict(tasas)}")
    print(f"  canales        : {dict(canales)}")
    if duraciones:
        duraciones.sort()
        n = len(duraciones)
        print(f"  duracion (s)   : min {duraciones[0]:.2f}  "
              f"mediana {duraciones[n // 2]:.2f}  "
              f"p90 {duraciones[int(n * 0.9)]:.2f}  max {duraciones[-1]:.2f}")
        informe["duracion"] = {
            "min": round(duraciones[0], 3),
            "mediana": round(duraciones[n // 2], 3),
            "p90": round(duraciones[int(n * 0.9)], 3),
            "max": round(duraciones[-1], 3),
        }
    informe["sample_rate"] = {str(k): v for k, v in tasas.items()}
    informe["canales"] = {str(k): v for k, v in canales.items()}

    if multicanal:
        print(f"  ficheros con >1 canal: {len(multicanal)}  (RAVDESS trae 5 dual-mono "
              f"entre 1,440; hallazgo propio del 2026-10-20)")
        for f in multicanal:
            print(f"      {f}")
        informe["multicanal"] = multicanal
        if len(multicanal) != 5:
            print("  AVISO: se esperaban 5. Si son otros, la copia difiere de la de "
                  "Zenodo y hay que mirar cuales.")
        if canales_distintos:
            problemas.append(
                f"{len(canales_distintos)} ficheros estereo cuyos canales NO son "
                f"identicos: el downmix a mono estaria mezclando dos senales distintas, "
                f"y eso si hay que tratarlo. {canales_distintos}")
        else:
            print("      los canales son bit a bit identicos (dual-mono): el downmix a "
                  "mono no pierde nada.")
        informe["canales_distintos"] = canales_distintos

    if 48000 not in tasas and tasas:
        print("  AVISO: el diseno publicado es 48 kHz y aqui no aparece. La copia puede "
              "venir remuestreada; no es un fallo, pero hay que declararlo.")
    if len(tasas) > 1:
        problemas.append(f"sample rate no uniforme: {dict(tasas)}")
    if vacios:
        problemas.append(f"{len(vacios)} ficheros de menos de 0.5 s: {vacios[:10]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True,
                    help="Raiz de RAVDESS (la que contiene Actor_01/ o "
                         "Audio_Speech_Actors_01-24/)")
    ap.add_argument("--fase", choices=["estructura", "audio", "todo"],
                    default="estructura")
    ap.add_argument("--limite-audio", type=int, default=0,
                    help="Leer solo los primeros N ficheros en la fase de audio "
                         "(0 = todos). Util para una pasada rapida.")
    ap.add_argument("--json", help="Escribir el informe aqui")
    args = ap.parse_args()

    raiz = Path(args.data_path)
    try:
        base = localizar(raiz)
    except FileNotFoundError as e:
        print(f"\n*** {e}\n")
        return 1
    print(f"\n  raiz de audio: {base}")

    problemas: list = []
    informe: dict = {"raiz": str(base), "fase": args.fase}

    if args.fase in ("estructura", "todo"):
        fase_estructura(base, problemas, informe)
        fase_folds(problemas, informe)
    if args.fase in ("audio", "todo"):
        fase_audio(base, problemas, informe, args.limite_audio)

    print("\n" + "=" * 72)
    if problemas:
        print(f"HAY {len(problemas)} COSAS QUE REVISAR. NO entrenes todavia.")
        for p in problemas:
            print(f"  - {p}")
    else:
        print("TODO CUADRA con el contrato de src/data/ravdess_dataset.py,")
        print("verificado contra la descarga de Zenodo el 2026-10-20.")
    print("=" * 72)
    informe["problemas"] = problemas

    if args.json:
        Path(args.json).write_text(json.dumps(informe, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"\ninforme en {args.json}")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
