   
                                
                                                       
                   
   

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def generate_comparative_table(results: Dict, save_path: str = "results/comparative_table.png"):
       
                                                         
    
                      
                                                              
    
         
                                                                     
                                       
       
    
                                                                         
    reference_data = {
        'TGNNExplainer': {
            'Wikipedia_TGNN': {'fid+': 0.05, 'fid-': 0.90, 'sparsity': 0.33, 'char': 0.09},
            'Wikipedia_TGAT': {'fid+': 0.08, 'fid-': 0.87, 'sparsity': 0.25, 'char': 0.14},
            'UCI_TGNN': {'fid+': 0.10, 'fid-': 0.82, 'sparsity': 0.35, 'char': 0.17},
            'UCI_TGAT': {'fid+': 0.14, 'fid-': 0.76, 'sparsity': 0.36, 'char': 0.24},
        },
        'Baseline': {
            'Wikipedia_TGNN': {'fid+': 0.18, 'fid-': 0.91, 'sparsity': 0.06, 'char': 0.30},
            'Wikipedia_TGAT': {'fid+': 0.18, 'fid-': 0.83, 'sparsity': 0.07, 'char': 0.30},
            'UCI_TGNN': {'fid+': 0.20, 'fid-': 0.72, 'sparsity': 0.06, 'char': 0.31},
            'UCI_TGAT': {'fid+': 0.32, 'fid-': 0.66, 'sparsity': 0.04, 'char': 0.42},
        }
    }
    
                                       
    table_data = []
    
                   
    for config_name, metrics in reference_data['TGNNExplainer'].items():
        dataset, model = config_name.split('_')
        dataset_full = "Wikipedia" if dataset == "Wikipedia" else "UCI-Messages"
        table_data.append([
            'TGNNExplainer',
            dataset_full,
            model,
            f"{metrics['fid+']:.2f}",
            f"{metrics['fid-']:.2f}",
            f"{metrics['sparsity']:.2f}",
            f"{metrics['char']:.2f}"
        ])
    
          
    for config_name, metrics in reference_data['Baseline'].items():
        dataset, model = config_name.split('_')
        dataset_full = "Wikipedia" if dataset == "Wikipedia" else "UCI-Messages"
        table_data.append([
            'Baseline',
            dataset_full,
            model,
            f"{metrics['fid+']:.2f}",
            f"{metrics['fid-']:.2f}",
            f"{metrics['sparsity']:.2f}",
            f"{metrics['char']:.2f}"
        ])
    
                                     
    for exp_name, exp_results in results.items():
        model_name = exp_results['model_name']
        dataset_name = exp_results['dataset_name']
        final_metrics = exp_results['final_metrics']
        
        fid_plus = np.mean(final_metrics['fidelity_plus_scores'])
        fid_minus = np.mean(final_metrics['fidelity_minus_scores'])
        sparsity = np.mean(final_metrics['sparsity_scores'])
        char = np.mean(final_metrics['char_scores'])
        
        table_data.append([
            'CausalVGAE_Explainer',
            dataset_name,
            model_name,
            f"{fid_plus:.2f}",
            f"{fid_minus:.2f}",
            f"{sparsity:.2f}",
            f"{char:.2f}"
        ])
    
                        
    columns = ['MÃ©thode', 'Dataset', 'ModÃ¨le', 'Fid+', 'Fid-', 'Sparsity', 'Char']
    df = pd.DataFrame(table_data, columns=columns)
    
                        
    csv_path = save_path.replace('.png', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"âœ… Tableau CSV sauvegardÃ©: {csv_path}")
    
                            
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
                                 
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.20, 0.15, 0.10, 0.10, 0.10, 0.15, 0.10]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
                         
    for (i, j), cell in table.get_celld().items():
        if i == 0:
                     
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4CAF50')
        else:
                    
            method = df.iloc[i-1]['MÃ©thode']
            if method == 'TGNNExplainer':
                cell.set_facecolor('#FFE6E6')
            elif method == 'Baseline':
                cell.set_facecolor('#E6F3FF')
            else:                        
                cell.set_facecolor('#E8F5E9')
                cell.set_text_props(weight='bold')
    
    plt.title('Tableau Comparatif des MÃ©thodes d\'ExplicabilitÃ©\n' +
             'causalvgae-explainer vs TGNNExplainer vs Baseline',
             fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"âœ… Tableau visuel sauvegardÃ©: {save_path}")
    
                              
    print("\n" + "="*80)
    print("TABLEAU COMPARATIF DES RÃ‰SULTATS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df


def generate_performance_comparison_plot(results: Dict, save_path: str = "results/performance_comparison.png"):
       
                                                       
    
         
                                          
                                       
       
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
                                               
    methods = []
    fid_plus_values = []
    fid_minus_values = []
    sparsity_values = []
    char_values = []
    
    for exp_name, exp_results in results.items():
        model_name = exp_results['model_name']
        dataset_name = exp_results['dataset_name']
        final_metrics = exp_results['final_metrics']
        
        methods.append(f"{model_name}\n{dataset_name}")
        fid_plus_values.append(np.mean(final_metrics['fidelity_plus_scores']))
        fid_minus_values.append(np.mean(final_metrics['fidelity_minus_scores']))
        sparsity_values.append(np.mean(final_metrics['sparsity_scores']))
        char_values.append(np.mean(final_metrics['char_scores']))
    
                       
    axes[0, 0].bar(methods, fid_plus_values, color='#4CAF50', alpha=0.7)
    axes[0, 0].set_title('FidÃ©litÃ©+ (fid+)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Score', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
                       
    axes[0, 1].bar(methods, fid_minus_values, color='#FF9800', alpha=0.7)
    axes[0, 1].set_title('FidÃ©litÃ©- (fid-)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
                      
    axes[1, 0].bar(methods, sparsity_values, color='#2196F3', alpha=0.7)
    axes[1, 0].set_title('SparsitÃ©', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].axhline(y=0.45, color='red', linestyle='--', label='Cible: 0.45')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
                                    
    axes[1, 1].bar(methods, char_values, color='#9C27B0', alpha=0.7)
    axes[1, 1].set_title('Score de CaractÃ©risation (char)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"âœ… Graphiques de performance sauvegardÃ©s: {save_path}")


def export_results_to_latex(df: pd.DataFrame, save_path: str = "results/comparative_table.tex"):
       
                                      
    
         
                                
                                       
       
    latex_str = df.to_latex(index=False, float_format="%.2f")
    
    with open(save_path, 'w') as f:
        f.write(latex_str)
    
    print(f"âœ… Tableau LaTeX sauvegardÃ©: {save_path}")

