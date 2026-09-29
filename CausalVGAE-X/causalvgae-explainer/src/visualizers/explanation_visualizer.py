   
                                                               
                                                            
   

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Optional
from pathlib import Path
import seaborn as sns


class ExplanationVisualizer:
       
                                                                      
       
    
    def __init__(self, save_dir: str = "results/figures"):
           
             
                                                          
           
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
               
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_causal_subgraph(self,
                            snapshot: Dict,
                            causal_mask: torch.Tensor,
                            threshold: float = 0.5,
                            title: str = "Causal Subgraph",
                            save_name: str = None):
           
                                                                         
        
             
                                                   
                                                           
                                                    
                                     
                                                      
           
                              
        edge_index = snapshot['edge_index'].cpu().numpy()
        mask_values = causal_mask.squeeze().cpu().numpy()
        
                                  
        G = nx.DiGraph()
        
                            
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(src, dst)
        
                                      
        important_nodes = np.where(mask_values > threshold)[0]
        
        if len(important_nodes) == 0:
            print("⚠️  Aucun nœud important trouvé avec ce seuil")
            return
        
                                          
        subG = G.subgraph(important_nodes)
        
                
        pos = nx.spring_layout(subG, k=1, iterations=50)
        
                                          
        node_colors = [mask_values[node] for node in subG.nodes()]
        
                           
        node_sizes = [500 + 1000 * mask_values[node] for node in subG.nodes()]
        
                         
        fig, ax = plt.subplots(figsize=(14, 10))
        
                             
        nx.draw_networkx_edges(subG, pos, alpha=0.3, edge_color='gray',
                              arrows=True, arrowsize=10, ax=ax)
        
                            
        nodes = nx.draw_networkx_nodes(subG, pos, node_color=node_colors,
                                      node_size=node_sizes, cmap='YlOrRd',
                                      vmin=0, vmax=1, ax=ax)
        
                
        nx.draw_networkx_labels(subG, pos, font_size=8, ax=ax)
        
                  
        plt.colorbar(nodes, ax=ax, label='Causal Importance')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Graphe sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_fidelity_vs_sparsity(self,
                                  fidelity_plus: List[float],
                                  fidelity_minus: List[float],
                                  sparsity: List[float],
                                  save_name: str = "fidelity_vs_sparsity.png"):
           
                                                              
        
             
                                                     
                                                      
                                                  
                                                   
           
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
                               
        axes[0].scatter(sparsity, fidelity_plus, alpha=0.6, s=50, color='blue')
        axes[0].plot(np.sort(sparsity), np.poly1d(np.polyfit(sparsity, fidelity_plus, 2))(np.sort(sparsity)),
                    color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Sparsity', fontsize=12)
        axes[0].set_ylabel('Fidelity+', fontsize=12)
        axes[0].set_title('Fidelity+ vs Sparsity', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
                               
        axes[1].scatter(sparsity, fidelity_minus, alpha=0.6, s=50, color='green')
        axes[1].plot(np.sort(sparsity), np.poly1d(np.polyfit(sparsity, fidelity_minus, 2))(np.sort(sparsity)),
                    color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Sparsity', fontsize=12)
        axes[1].set_ylabel('Fidelity-', fontsize=12)
        axes[1].set_title('Fidelity- vs Sparsity', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Courbes sauvegardées: {save_path}")
        
        plt.show()
    
    def plot_temporal_continuity(self,
                                causal_masks: List[torch.Tensor],
                                timestamps: List[float],
                                save_name: str = "temporal_continuity.png"):
           
                                                              
        
             
                                                               
                                                 
                                     
           
                                                            
        similarities = []
        
        for i in range(len(causal_masks) - 1):
            mask_t = causal_masks[i].squeeze()
            mask_t1 = causal_masks[i + 1].squeeze()
            
                               
            similarity = torch.cosine_similarity(mask_t, mask_t1, dim=0)
            similarities.append(similarity.item())
        
                       
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(timestamps[1:], similarities, marker='o', linewidth=2, markersize=6, color='purple')
        ax.fill_between(timestamps[1:], similarities, alpha=0.3, color='purple')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('Temporal Continuity of Causal Explanations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
                            
        ax.axhline(y=np.mean(similarities), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(similarities):.3f}')
        ax.legend()
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Continuité temporelle sauvegardée: {save_path}")
        
        plt.show()
    
    def plot_training_history(self,
                             history: Dict,
                             save_name: str = "training_history.png"):
           
                                             
        
             
                                                       
                                     
           
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
                     
        axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', marker='o', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', marker='s', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
                                         
        train_losses = history['train_losses']
        for key in ['recon', 'kl', 'causal', 'fidelity']:
            values = [epoch_losses[key] for epoch_losses in train_losses]
            axes[0, 1].plot(epochs, values, label=key, marker='o', linewidth=2)
        
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Loss Component', fontsize=12)
        axes[0, 1].set_title('Training Loss Components', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
                                      
        for key in ['sparsity', 'char']:
            values = [epoch_losses[key] for epoch_losses in train_losses]
            axes[1, 0].plot(epochs, values, label=key, marker='o', linewidth=2)
        
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Loss Component', fontsize=12)
        axes[1, 0].set_title('Sparsity & Characterization', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
                       
        values = [epoch_losses['temporal'] for epoch_losses in train_losses]
        axes[1, 1].plot(epochs, values, label='Temporal', marker='o', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Temporal Loss', fontsize=12)
        axes[1, 1].set_title('Temporal Continuity Loss', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Historique d'entraînement sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self,
                               metrics_dict: Dict[str, Dict],
                               save_name: str = "metrics_comparison.png"):
           
                                                        
        
             
                                                             
                                     
           
        methods = list(metrics_dict.keys())
        metric_names = ['fidelity_plus', 'fidelity_minus', 'sparsity', 'characterization']
        
                              
        data = {metric: [metrics_dict[method].get(metric, 0) for method in methods]
                for metric in metric_names}
        
                             
        x = np.arange(len(methods))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
                                     
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        for i, (metric, color) in enumerate(zip(metric_names, colors)):
            offset = width * (i - 1.5)
            ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title(), color=color)
        
        ax.set_xlabel('Methods', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Explainability Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Comparaison des métriques sauvegardée: {save_path}")
        
        plt.show()
    
    def create_explanation_report(self,
                                 explainer: torch.nn.Module,
                                 snapshots: List[Dict],
                                 num_snapshots: int = 5,
                                 device: str = 'cuda'):
           
                                                       
        
             
                                                  
                                                      
                                                         
                          
           
        print("\n📊 Création du rapport d'explications...")
        
        explainer.eval()
        
                                            
        indices = np.linspace(0, len(snapshots)-1, num_snapshots, dtype=int)
        
        for i, idx in enumerate(indices):
            snapshot = snapshots[idx]
            
                                     
            snapshot = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in snapshot.items()}
            
                                   
            with torch.no_grad():
                explanation = explainer.explain(snapshot, threshold=0.5)
            
                        
            title = f"Snapshot {idx} - Time: {snapshot.get('timestamps', 0):.3f}"
            save_name = f"causal_subgraph_snapshot_{idx}.png"
            
            self.plot_causal_subgraph(
                snapshot,
                explanation['causal_mask'],
                threshold=0.5,
                title=title,
                save_name=save_name
            )
        
        print("✅ Rapport d'explications créé!")
