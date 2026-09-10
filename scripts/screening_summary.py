r"""Resumen de screening para las ablaciones de las series 50 y 60.

Lee los `final_results.json` bajo `checkpoints/` y tabula WA/UA por estudio.
Uso: python scripts\screening_summary.py
"""
import json
import glob
import os

# Anclado a la raiz del repositorio (padre de scripts/), no al directorio actual.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(REPO_ROOT, 'checkpoints')

studies = {}
for path in sorted(glob.glob(os.path.join(CKPT, 'ablation_5*/**/final_results.json')) + glob.glob(os.path.join(CKPT, 'ablation_6*/**/final_results.json'))):
    # Extract study number from path like checkpoints/ablation_58/.../final_results.json
    parts = path.replace('\\', '/').split('/')
    study_dir = [p for p in parts if p.startswith('ablation_')][0]
    study_num = study_dir.split('ablation_')[1]
    with open(path) as f:
        data = json.load(f)
    metrics = data.get('test_metrics', {})
    per_class = metrics.get('per_class', {})
    neutral_f1 = per_class.get('neutral', {}).get('f1', 0) if isinstance(per_class, dict) else 0
    studies[int(study_num)] = {
        'ua': metrics.get('unweighted_accuracy', 0),
        'wa': metrics.get('weighted_accuracy', 0),
        'f1_macro': metrics.get('f1_macro', 0),
        'neutral_f1': neutral_f1,
        'path': path
    }

print("=== SCREENING RESULTS SUMMARY ===")
print(f"{'Study':<8} {'UA':<8} {'WA':<8} {'F1 Macro':<10} {'Neutral F1':<12}")
print("-" * 50)
for s in sorted(studies):
    m = studies[s]
    print(f"{s:<8} {m['ua']:<8.4f} {m['wa']:<8.4f} {m['f1_macro']:<10.4f} {m['neutral_f1']:<12.4f}")

print()
best_ua = max(studies, key=lambda x: studies[x]['ua'])
best_wa = max(studies, key=lambda x: studies[x]['wa'])
best_f1 = max(studies, key=lambda x: studies[x]['f1_macro'])
best_neutral = max(studies, key=lambda x: studies[x]['neutral_f1'])

print(f"Best UA:        Study {best_ua} ({studies[best_ua]['ua']:.4f})")
print(f"Best WA:        Study {best_wa} ({studies[best_wa]['wa']:.4f})")
print(f"Best F1 Macro:  Study {best_f1} ({studies[best_f1]['f1_macro']:.4f})")
print(f"Best Neutral:   Study {best_neutral} ({studies[best_neutral]['neutral_f1']:.4f})")

# Also load previous best for comparison
print("\n=== COMPARISON WITH PREVIOUS BEST ===")
prev_best = {
    'Study 45': {'ua': 0.7923, 'wa': 0.7961, 'f1_macro': 0.7868, 'neutral_f1': 0.6476},
    'Study 55': {'ua': 0.7866, 'wa': 0.8000, 'f1_macro': 0.7839, 'neutral_f1': 0.5941},
    'Study 58': studies.get(58, {}),
    'Study 59': studies.get(59, {}),
    'Study 60': studies.get(60, {}),
    'Study 61': studies.get(61, {}),
    'Study 62': studies.get(62, {}),
}

print(f"{'Model':<12} {'UA':<8} {'WA':<8} {'F1 Macro':<10} {'Neutral F1':<12}")
print("-" * 52)
for name, m in prev_best.items():
    if m:
        print(f"{name:<12} {m.get('ua', 0):<8.4f} {m.get('wa', 0):<8.4f} {m.get('f1_macro', 0):<10.4f} {m.get('neutral_f1', 0):<12.4f}")
