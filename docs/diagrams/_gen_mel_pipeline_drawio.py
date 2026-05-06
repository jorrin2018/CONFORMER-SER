"""Generate mel_vit_pipeline_LNCS.drawio with programmatic heatmap blocks.

Re-run after editing preprocessing text or layout constants.
"""

from __future__ import annotations


def heat_grid(
    x0: float,
    y0: float,
    cols: int,
    rows: int,
    cw: float,
    ch: float,
    prefix: str,
    palette,
) -> str:
    parts: list[str] = []
    pid = 0
    for r in range(rows):
        for c in range(cols):
            v = (r / max(rows - 1, 1)) * 0.55 + (c / max(cols - 1, 1)) * 0.45
            col = palette(v)
            cx = x0 + c * cw
            cy = y0 + r * ch
            parts.append(
                f'        <mxCell id="{prefix}{pid}" value="" '
                f'style="rounded=0;fillColor={col};strokeColor=#bbbbbb;strokeWidth=0.25;" '
                f'vertex="1" parent="1">\n'
                f'          <mxGeometry x="{cx:.2f}" y="{cy:.2f}" width="{cw:.2f}" '
                f'height="{ch:.2f}" as="geometry" />\n'
                f"        </mxCell>"
            )
            pid += 1
    return "\n".join(parts)


def pal_mel(v: float) -> str:
    m = ["#e8f2fb", "#b8d4ef", "#6ea9dc", "#2E5AAC", "#1a4580"]
    return m[min(int(v * 4.99), 4)]


def pal_norm(v: float) -> str:
    m = ["#061f47", "#2E5AAC", "#8ec5e8", "#d4af37", "#fceea8"]
    return m[min(int(v * 4.99), 4)]


def main() -> None:
    # Step 2: inside frame 294,104 / 168x62
    h2 = heat_grid(296.0, 112.0, 16, 8, 10.25, 7.125, "hm2_", pal_mel)
    # Step 3: same layout inside 544,104
    h3 = heat_grid(546.0, 112.0, 16, 8, 10.25, 7.125, "hm3_", pal_norm)
    # Step 4: 14x14 in square 834,108 / 128x128
    h4 = heat_grid(835.0, 109.0, 14, 14, 9.0, 9.0, "hm4_", pal_norm)

    parts_vit: list[str] = []
    gx0, gy0, gside = 1368.0, 114.0, 160.0
    psz = gside / 14.0
    for r in range(14):
        for c in range(14):
            cx = gx0 + c * psz + 0.5
            cy = gy0 + r * psz + 0.5
            col = "#d4ead8" if (r, c) != (5, 8) else "#f5c97a"
            sw = "1.2" if (r, c) == (5, 8) else "0.35"
            parts_vit.append(
                f'        <mxCell id="pg_{r}_{c}" value="" '
                f'style="rounded=0;fillColor={col};strokeColor=#555555;strokeWidth={sw};" '
                f'vertex="1" parent="1">\n'
                f'          <mxGeometry x="{cx:.2f}" y="{cy:.2f}" width="{psz - 1:.2f}" '
                f'height="{psz - 1:.2f}" as="geometry" />\n'
                f"        </mxCell>"
            )

    full = f"""<?xml version="1.0" encoding="UTF-8"?>
<!--
  Mel-spectrogram → ViT pipeline (7 steps + embedding).
  Springer LNCS-style, 16:5 canvas, vector primitives. Generated; edit in draw.io.
-->
<mxfile host="app.diagrams.net" agent="CONFORMER-SER" version="22.1.0">
  <diagram id="mel-vit-pipeline" name="Mel ViT Pipeline">
    <mxGraphModel dx="2600" dy="820" grid="0" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2600" pageHeight="812" math="0" shadow="0" background="#ffffff">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <mxCell id="ttl" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:12px&quot;&gt;&lt;b&gt;Visual stream preprocessing pipeline&lt;/b&gt;&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;fontColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="40" y="12" width="520" height="22" as="geometry" />
        </mxCell>

        <mxCell id="s1_box" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:11px&quot;&gt;Raw Waveform&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1;fontColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="40" y="48" width="180" height="38" as="geometry" />
        </mxCell>
        <mxCell id="s1_bg" value="" style="rounded=0;fillColor=#f2f7fc;strokeColor=#1f5188;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="44" y="96" width="172" height="58" as="geometry" />
        </mxCell>
        <mxCell id="s1_w1" value="" style="endArrow=none;html=1;strokeWidth=1.6;strokeColor=#2E5AAC;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="48" y="128" as="sourcePoint" /><mxPoint x="64" y="114" as="targetPoint" /></mxGeometry></mxCell>
        <mxCell id="s1_w2" value="" style="endArrow=none;html=1;strokeWidth=1.6;strokeColor=#2E5AAC;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="64" y="114" as="sourcePoint" /><mxPoint x="82" y="138" as="targetPoint" /></mxGeometry></mxCell>
        <mxCell id="s1_w3" value="" style="endArrow=none;html=1;strokeWidth=1.6;strokeColor=#2E5AAC;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="82" y="138" as="sourcePoint" /><mxPoint x="98" y="110" as="targetPoint" /></mxGeometry></mxCell>
        <mxCell id="s1_w4" value="" style="endArrow=none;html=1;strokeWidth=1.6;strokeColor=#2E5AAC;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="98" y="110" as="sourcePoint" /><mxPoint x="118" y="144" as="targetPoint" /></mxGeometry></mxCell>
        <mxCell id="s1_w5" value="" style="endArrow=none;html=1;strokeWidth=1.6;strokeColor=#2E5AAC;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="118" y="144" as="sourcePoint" /><mxPoint x="136" y="106" as="targetPoint" /></mxGeometry></mxCell>
        <mxCell id="s1_w6" value="" style="endArrow=none;html=1;strokeWidth=1.6;strokeColor=#2E5AAC;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="136" y="106" as="sourcePoint" /><mxPoint x="156" y="142" as="targetPoint" /></mxGeometry></mxCell>
        <mxCell id="s1_w7" value="" style="endArrow=none;html=1;strokeWidth=1.6;strokeColor=#2E5AAC;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="156" y="142" as="sourcePoint" /><mxPoint x="176" y="120" as="targetPoint" /></mxGeometry></mxCell>
        <mxCell id="s1_w8" value="" style="endArrow=none;html=1;strokeWidth=1.6;strokeColor=#2E5AAC;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="176" y="120" as="sourcePoint" /><mxPoint x="196" y="146" as="targetPoint" /></mxGeometry></mxCell>
        <mxCell id="s1_cap" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:9px&quot;&gt;16 kHz, mono&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;fontColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="60" y="162" width="140" height="16" as="geometry" />
        </mxCell>

        <mxCell id="s2_box" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:11px&quot;&gt;MelSpectrogram()&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="288" y="48" width="180" height="38" as="geometry" />
        </mxCell>
        <mxCell id="s2_frame" value="" style="rounded=0;fillColor=none;strokeColor=#333333;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="294" y="104" width="168" height="62" as="geometry" />
        </mxCell>
        <mxCell id="s2_ylab" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:7px&quot;&gt;Mel freq.&lt;br&gt;(80 bins)&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;fontColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="242" y="106" width="52" height="56" as="geometry" />
        </mxCell>
        <mxCell id="s2_xlab" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:7px&quot;&gt;Time frames&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;fontColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="318" y="172" width="120" height="12" as="geometry" />
        </mxCell>
{h2}
        <mxCell id="s2_sub" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:8px&quot;&gt;n_fft=400, hop=160, n_mels=80&lt;/font&gt;" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#f6f6f6;strokeColor=#888888;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="278" y="192" width="200" height="24" as="geometry" />
        </mxCell>

        <mxCell id="s3_box" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:10px&quot;&gt;Log Compression +&lt;br&gt;Z-score Normalization&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="538" y="42" width="200" height="48" as="geometry" />
        </mxCell>
        <mxCell id="s3_frame" value="" style="rounded=0;fillColor=none;strokeColor=#333333;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="544" y="104" width="168" height="62" as="geometry" />
        </mxCell>
{h3}
        <mxCell id="s3_cb1" value="" style="rounded=0;fillColor=#061f47;strokeColor=none;" vertex="1" parent="1"><mxGeometry x="722" y="118" width="8" height="10" as="geometry" /></mxCell>
        <mxCell id="s3_cb2" value="" style="rounded=0;fillColor=#2E5AAC;strokeColor=none;" vertex="1" parent="1"><mxGeometry x="722" y="128" width="8" height="10" as="geometry" /></mxCell>
        <mxCell id="s3_cb3" value="" style="rounded=0;fillColor=#8ec5e8;strokeColor=none;" vertex="1" parent="1"><mxGeometry x="722" y="138" width="8" height="10" as="geometry" /></mxCell>
        <mxCell id="s3_cb4" value="" style="rounded=0;fillColor=#d4af37;strokeColor=none;" vertex="1" parent="1"><mxGeometry x="722" y="148" width="8" height="10" as="geometry" /></mxCell>
        <mxCell id="s3_cb5" value="" style="rounded=0;fillColor=#fceea8;strokeColor=none;" vertex="1" parent="1"><mxGeometry x="722" y="158" width="8" height="10" as="geometry" /></mxCell>
        <mxCell id="s3_cbl" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:6px&quot;&gt;low … high&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;fontColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="708" y="170" width="36" height="12" as="geometry" />
        </mxCell>
        <mxCell id="s3_cap" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:9px&quot;&gt;per-utterance normalization&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;fontColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="548" y="178" width="180" height="14" as="geometry" />
        </mxCell>

        <mxCell id="s4_box" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:11px&quot;&gt;Bilinear Resize&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="808" y="48" width="180" height="38" as="geometry" />
        </mxCell>
        <mxCell id="s4_sq" value="" style="rounded=0;fillColor=none;strokeColor=#333333;strokeWidth=1.2;" vertex="1" parent="1">
          <mxGeometry x="834" y="108" width="128" height="128" as="geometry" />
        </mxCell>
{h4}
        <mxCell id="s4_lbl" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:9px&quot;&gt;224 × 224&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;fontColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="850" y="244" width="100" height="14" as="geometry" />
        </mxCell>

        <mxCell id="s5_box" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:11px&quot;&gt;Channel Replication ×3&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="1078" y="48" width="200" height="38" as="geometry" />
        </mxCell>
        <mxCell id="s5_r" value="" style="rounded=0;fillColor=#b8d4ef;strokeColor=#333333;strokeWidth=1;" vertex="1" parent="1"><mxGeometry x="1108" y="108" width="60" height="56" as="geometry" /></mxCell>
        <mxCell id="s5_g" value="" style="rounded=0;fillColor=#a8d4a8;strokeColor=#333333;strokeWidth=1;" vertex="1" parent="1"><mxGeometry x="1172" y="108" width="60" height="56" as="geometry" /></mxCell>
        <mxCell id="s5_b" value="" style="rounded=0;fillColor=#dcd0c8;strokeColor=#333333;strokeWidth=1;" vertex="1" parent="1"><mxGeometry x="1236" y="108" width="60" height="56" as="geometry" /></mxCell>
        <mxCell id="s5_cap" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:9px&quot;&gt;224 × 224 × 3&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;fontColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="1120" y="176" width="100" height="14" as="geometry" />
        </mxCell>

        <mxCell id="s6_box" value="&lt;font face=&quot;Times New Roman&quot; color=&quot;#FFFFFF&quot; style=&quot;font-size:11px&quot;&gt;&lt;b&gt;ViT-Base/16&lt;/b&gt;&lt;/font&gt;&lt;div&gt;&lt;font face=&quot;Times New Roman&quot; color=&quot;#FFFFFF&quot; style=&quot;font-size:8px&quot;&gt;16×16 patches, [CLS]&lt;br&gt;(ImageNet-21k, frozen)&lt;/font&gt;&lt;/div&gt;" style="rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#3d8f4a;strokeColor=#2a6534;strokeWidth=1;fontColor=#ffffff;" vertex="1" parent="1">
          <mxGeometry x="1348" y="40" width="200" height="56" as="geometry" />
        </mxCell>
        <mxCell id="s6_grid_bg" value="" style="rounded=0;fillColor=#f0f7f1;strokeColor=#333333;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="1368" y="114" width="160" height="112" as="geometry" />
        </mxCell>
{chr(10).join(parts_vit)}
        <mxCell id="s6_cap" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:8px&quot;&gt;patch grid (one patch highlighted)&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;fontColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="1358" y="234" width="180" height="14" as="geometry" />
        </mxCell>

        <mxCell id="s7_box" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:11px&quot;&gt;Linear Projection&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="1628" y="52" width="170" height="38" as="geometry" />
        </mxCell>
        <mxCell id="s7_vt" value="" style="rounded=1;fillColor=#e8e8e8;strokeColor=#000000;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="1668" y="108" width="18" height="72" as="geometry" />
        </mxCell>
        <mxCell id="s7_ar" value="" style="endArrow=classic;html=1;strokeWidth=1.5;strokeColor=#000000;" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry"><mxPoint x="1694" y="144" as="sourcePoint" /><mxPoint x="1734" y="144" as="targetPoint" /></mxGeometry>
        </mxCell>
        <mxCell id="s7_vb" value="" style="rounded=1;fillColor=#d0d0d0;strokeColor=#000000;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="1742" y="124" width="24" height="40" as="geometry" />
        </mxCell>
        <mxCell id="s7_lbl" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:9px&quot;&gt;768 → 512 dim&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;fontColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="1640" y="192" width="146" height="14" as="geometry" />
        </mxCell>

        <mxCell id="out" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:11px&quot;&gt;&lt;b&gt;m&lt;/b&gt; ∈ ℝ&lt;sup&gt;512&lt;/sup&gt;&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#F5C97A;strokeColor=#C17F2A;strokeWidth=1.2;fontColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="1910" y="52" width="140" height="44" as="geometry" />
        </mxCell>
        <mxCell id="out_icon" value="" style="rounded=0;fillColor=#e89a24;strokeColor=#a86b18;strokeWidth=1;" vertex="1" parent="1"><mxGeometry x="1940" y="112" width="80" height="22" as="geometry" /></mxCell>
        <mxCell id="out_sub" value="&lt;font face=&quot;Times New Roman&quot; style=&quot;font-size:8px&quot;&gt;output embedding&lt;/font&gt;" style="text;html=1;strokeColor=none;fillColor=none;fontColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="1914" y="142" width="132" height="12" as="geometry" />
        </mxCell>

        <mxCell id="a12" style="endArrow=classic;html=1;strokeWidth=1.2;strokeColor=#000000;" edge="1" parent="1" source="s1_box" target="s2_box"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="a23" style="endArrow=classic;html=1;strokeWidth=1.2;strokeColor=#000000;" edge="1" parent="1" source="s2_box" target="s3_box"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="a34" style="endArrow=classic;html=1;strokeWidth=1.2;strokeColor=#000000;" edge="1" parent="1" source="s3_box" target="s4_box"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="a45" style="endArrow=classic;html=1;strokeWidth=1.2;strokeColor=#000000;" edge="1" parent="1" source="s4_box" target="s5_box"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="a56" style="endArrow=classic;html=1;strokeWidth=1.2;strokeColor=#000000;" edge="1" parent="1" source="s5_box" target="s6_box"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="a67" style="endArrow=classic;html=1;strokeWidth=1.2;strokeColor=#000000;" edge="1" parent="1" source="s6_box" target="s7_box"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="a7o" style="endArrow=classic;html=1;strokeWidth=1.2;strokeColor=#000000;" edge="1" parent="1" source="s7_box" target="out"><mxGeometry relative="1" as="geometry" /></mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

    out_path = "d:/DOCTORADO/PROYECTOS/CONFORMER-SER/docs/diagrams/mel_vit_pipeline_LNCS.drawio"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
