   
                                          
   

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict


def plot_training_metrics(history: Dict, save_path: str = "results/training_metrics.png"):
       
                                             
    
         
                                          
                                       
       
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
             
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
                  
    axes[0, 1].plot(epochs, history['fidelity_plus'], 'g-', linewidth=2)
    axes[0, 1].set_title('Fidélité+ (fid+)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Fid+')
    axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    
                  
    axes[0, 2].plot(epochs, history['fidelity_minus'], 'orange', linewidth=2)
    axes[0, 2].set_title('Fidélité- (fid-)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Fid-')
    axes[0, 2].axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    axes[0, 2].grid(True, alpha=0.3)
    
                 
    axes[1, 0].plot(epochs, history['sparsity'], 'purple', linewidth=2)
    axes[1, 0].set_title('Sparsité', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Sparsity')
    axes[1, 0].axhline(y=0.45, color='red', linestyle='--', alpha=0.5, label='Cible: 0.45')
    axes[1, 0].axhspan(0.4, 0.5, alpha=0.2, color='yellow')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
                               
    axes[1, 1].plot(epochs, history['char_scores'], 'brown', linewidth=2)
    axes[1, 1].set_title('Score de Caractérisation (char)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Char Score')
    axes[1, 1].grid(True, alpha=0.3)
    
                            
    axes[1, 2].plot(epochs, history['temporal_continuity'], 'pink', linewidth=2)
    axes[1, 2].set_title('Continuité Temporelle', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Continuity')
    axes[1, 2].grid(True, alpha=0.3)
    
               
    axes[2, 0].plot(epochs, history['aufsc_plus'], 'darkgreen', linewidth=2)
    axes[2, 0].set_title('AUFSC+ (Area Under Fidelity+-Sparsity)', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('AUFSC+')
    axes[2, 0].grid(True, alpha=0.3)
    
               
    axes[2, 1].plot(epochs, history['aufsc_minus'], 'darkorange', linewidth=2)
    axes[2, 1].set_title('AUFSC- (Area Under Fidelity--Sparsity)', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('AUFSC-')
    axes[2, 1].grid(True, alpha=0.3)
    
                                
    axes[2, 2].plot(epochs, history['causal_info_flow'], 'teal', linewidth=2)
    axes[2, 2].set_title('Flux d\'Information Causale I(Z_c → y)', fontsize=12, fontweight='bold')
    axes[2, 2].set_xlabel('Epoch')
    axes[2, 2].set_ylabel('Causal Flow')
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.suptitle('causalvgae-explainer - Métriques d\'Entraînement Complètes',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Métriques d'entraînement sauvegardées: {save_path}")


def plot_fidelity_sparsity_curves(history: Dict, save_path: str = "results/fidelity_sparsity_curves.png"):
       
                                       
    
         
                                          
                                       
       
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    fidelity_plus = np.array(history['fidelity_plus'])
    fidelity_minus = np.array(history['fidelity_minus'])
    sparsity = np.array(history['sparsity'])
    epochs = np.arange(len(fidelity_plus))
    
                             
    scatter1 = ax1.scatter(sparsity, fidelity_plus, c=epochs, cmap='viridis',
                          s=80, alpha=0.7, edgecolors='black', linewidth=1)
    ax1.set_xlabel('Sparsité', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Fidélité+ (fid+)', fontsize=14, fontweight='bold')
    ax1.set_title('Trade-off Fidélité+-Sparsité', fontsize=15, fontweight='bold')
    ax1.axvline(x=0.45, color='red', linestyle='--', alpha=0.5, label='Sparsité Cible')
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Fid+ Idéal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='Époque')
    
                             
    scatter2 = ax2.scatter(sparsity, fidelity_minus, c=epochs, cmap='plasma',
                          s=80, alpha=0.7, edgecolors='black', linewidth=1)
    ax2.set_xlabel('Sparsité', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Fidélité- (fid-)', fontsize=14, fontweight='bold')
    ax2.set_title('Trade-off Fidélité--Sparsité', fontsize=15, fontweight='bold')
    ax2.axvline(x=0.45, color='red', linestyle='--', alpha=0.5, label='Sparsité Cible')
    ax2.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Fid- Cible')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='Époque')
    
    plt.suptitle('causalvgae-explainer - Courbes Fidélité-Sparsité',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Courbes fidélité-sparsité sauvegardées: {save_path}")
