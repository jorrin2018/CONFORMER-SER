import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def aggregate_kfold_results(summary_path):
    """
    Reads the K-Fold summary and individual fold results to create a consolidated confusion matrix.
    Since we don't have raw predictions saved, we'll try to aggregate metrics if possible,
    or just visualize the summary stats.
    
    Ideally, training script should save raw confusion matrices (npy or json).
    Since it saves images, we will rely on the summary json stats for text report,
    and we will attempt to load 'final_results.json' from each fold to see if we can reconstruct it.
    """
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"Loaded summary for {summary['num_folds']} folds.")
    
    # Metrics storage
    wa_scores = []
    ua_scores = []
    
    for fold in summary['folds']:
        wa = fold.get('test_weighted_accuracy', 0)
        ua = fold.get('test_unweighted_accuracy', 0)
        wa_scores.append(wa)
        ua_scores.append(ua)
        
        print(f"Fold Test Session {fold.get('test_session')}: WA={wa:.4f}, UA={ua:.4f}")
        
    mean_wa = np.mean(wa_scores)
    std_wa = np.std(wa_scores)
    mean_ua = np.mean(ua_scores)
    std_ua = np.std(ua_scores)
    
    print("\n" + "="*40)
    print(f"K-Fold Global Results (Ablation 29)")
    print("="*40)
    print(f"Weighted Accuracy:   {mean_wa*100:.2f}% ± {std_wa*100:.2f}%")
    print(f"Unweighted Accuracy: {mean_ua*100:.2f}% ± {std_ua*100:.2f}%")
    print("="*40)
    
    # Plotting Scores Distribution
    plt.figure(figsize=(8, 6))
    folds = range(1, len(wa_scores) + 1)
    plt.plot(folds, [s*100 for s in wa_scores], 'o-', label='Weighted Accuracy', color='#2E86C1', linewidth=2)
    plt.plot(folds, [s*100 for s in ua_scores], 's--', label='Unweighted Accuracy', color='#E74C3C', linewidth=2)
    
    plt.axhline(y=mean_wa*100, color='#2E86C1', linestyle=':', alpha=0.5)
    plt.axhline(y=mean_ua*100, color='#E74C3C', linestyle=':', alpha=0.5)
    
    plt.title(f'K-Fold Cross Validation Results (Ablation 29)\nMean WA: {mean_wa*100:.1f}% | Mean UA: {mean_ua*100:.1f}%', fontsize=14)
    plt.xlabel('Fold Number', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(60, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(folds)
    
    output_path = Path(summary_path).parent / 'kfold_performance_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved to: {output_path}")

if __name__ == "__main__":
    summary_file = "checkpoints/ablation_25_kfold/kfold_summary_ablation_29.json"
    
    if not os.path.exists(summary_file):
        print(f"Error: Summary file not found at {summary_file}")
    else:
        aggregate_kfold_results(summary_file)

