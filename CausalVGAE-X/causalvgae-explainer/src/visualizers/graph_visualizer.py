   
                                                
                                                    
   

import torch
import matplotlib.pyplot as plt
import numpy as np

try:
    import networkx as nx
    from matplotlib.patches import Patch
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️ NetworkX non disponible. Visualisations de graphes désactivées.")


def visualize_causal_explanations(trainer, snapshots, num_snapshots: int = 4,
                                  save_path: str = "results/causal_explanations.png"):
       
                                                               
    
         
                                        
                                     
                                                       
                                       
       
    if not NETWORKX_AVAILABLE:
        print("❌ NetworkX requis pour la visualisation")
        return
    
                                             
    indices = np.linspace(0, len(snapshots)-1, num_snapshots, dtype=int)
    selected_snapshots = [snapshots[i] for i in indices]
    
    fig, axes = plt.subplots(2, num_snapshots, figsize=(6*num_snapshots, 12))
    
    if num_snapshots == 1:
        axes = axes.reshape(2, 1)
    
    trainer.model.eval()
    prev_z_causal = None
    
    with torch.no_grad():
        for idx, snapshot in enumerate(selected_snapshots):
            output = trainer.model(snapshot, prev_z_causal)
            
            causal_mask = output['causal_mask'].cpu().numpy().flatten()
            node_features = snapshot['node_features'].cpu().numpy()
            edge_index = snapshot['edge_index'].cpu().numpy()
            
                                              
            ax1 = axes[0, idx]
            visualize_graph_snapshot(
                node_features, edge_index, causal_mask,
                ax1, f"Snapshot {idx+1} - Complet",
                filtered=False
            )
            
                                             
            ax2 = axes[1, idx]
            visualize_graph_snapshot(
                node_features, edge_index, causal_mask,
                ax2, f"Snapshot {idx+1} - Filtré (Causal)",
                filtered=True
            )
            
            prev_z_causal = output['z_causal']
    
                     
    legend_elements = [
        Patch(facecolor='#FF0000', label='🔴 Relations Importantes', alpha=0.9),
        Patch(facecolor='#00FF00', label='🟢 Relations Semi-Importantes', alpha=0.8),
        Patch(facecolor='#87CEEB', label='🔵 Relations Non-Importantes', alpha=0.7),
        Patch(facecolor='#D3D3D3', label='⚪ Relations Bruit', alpha=0.6)
    ]
    
    fig.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=12,
              frameon=True, fancybox=True, shadow=True)
    
    plt.suptitle('causalvgae-explainer - Explications Causales\nDataset Wikipedia',
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Explications causales sauvegardées: {save_path}")


def visualize_graph_snapshot(node_features, edge_index, causal_mask, ax, title, filtered=False):
       
                                   
    
         
                                         
                                      
                                  
                          
                              
                                                                        
       
    G = nx.Graph()
    
                       
    num_nodes = node_features.shape[0]
    G.add_nodes_from(range(num_nodes))
    
                        
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        G.add_edge(int(src), int(dst))
    
                        
    pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)
    
                                               
    node_importances = []
    for i in range(num_nodes):
        if i < len(causal_mask):
            importance = causal_mask[i]
        else:
            importance = 0.0
        node_importances.append(importance)
    
    node_importances = np.array(node_importances)
    
            
    high_threshold = 0.7
    medium_threshold = 0.3
    
    if filtered:
                                               
        important_nodes = np.where(node_importances > medium_threshold)[0]
        G_filtered = G.subgraph(important_nodes)
        
        if len(G_filtered.nodes()) > 0:
            pos_filtered = {n: pos[n] for n in G_filtered.nodes()}
            
                                
            node_colors = []
            for node in G_filtered.nodes():
                imp = node_importances[node]
                if imp > high_threshold:
                    node_colors.append('#FF0000')         
                else:
                    node_colors.append('#00FF00')        
            
                      
            nx.draw_networkx_nodes(G_filtered, pos_filtered, ax=ax,
                                  node_color=node_colors, node_size=400,
                                  alpha=0.95, edgecolors='black', linewidths=2)
            
                    
            edge_colors = []
            edge_widths = []
            for edge in G_filtered.edges():
                src_imp = node_importances[edge[0]]
                dst_imp = node_importances[edge[1]]
                avg_imp = (src_imp + dst_imp) / 2
                
                if avg_imp > high_threshold:
                    edge_colors.append('#FF0000')
                    edge_widths.append(4)
                else:
                    edge_colors.append('#00FF00')
                    edge_widths.append(3)
            
            nx.draw_networkx_edges(G_filtered, pos_filtered, ax=ax,
                                  edge_color=edge_colors, width=edge_widths,
                                  alpha=0.9)
            
            nx.draw_networkx_labels(G_filtered, pos_filtered, ax=ax,
                                   font_size=9, font_weight='bold')
        else:
            ax.text(0.5, 0.5, 'Aucune relation\ncausale significative',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, fontweight='bold')
    
    else:
                        
        node_colors = []
        for imp in node_importances:
            if imp > high_threshold:
                node_colors.append('#FF0000')                     
            elif imp > medium_threshold:
                node_colors.append('#00FF00')                         
            else:
                node_colors.append('#87CEEB')                             
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                              node_size=350, alpha=0.9,
                              edgecolors='black', linewidths=1.5)
        
                              
        for edge in G.edges():
            src_imp = node_importances[edge[0]] if edge[0] < len(node_importances) else 0
            dst_imp = node_importances[edge[1]] if edge[1] < len(node_importances) else 0
            avg_imp = (src_imp + dst_imp) / 2
            
            if avg_imp > high_threshold:
                color = '#FF0000'
                width = 3.5
            elif avg_imp > medium_threshold:
                color = '#00FF00'
                width = 2.5
            else:
                color = '#87CEEB'
                width = 1.5
            
            nx.draw_networkx_edges(G, pos, edgelist=[edge], ax=ax,
                                  edge_color=[color], width=width, alpha=0.8)
        
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')


def visualize_temporal_evolution(trainer, snapshots, selected_nodes=None,
                                 save_path: str = "results/temporal_evolution.png"):
       
                                                            
    
         
                                        
                                     
                                                               
                                       
       
    if selected_nodes is None:
        selected_nodes = [0, 1, 2, 3, 4]
    
    trainer.model.eval()
    
    evolution_data = {node: {'importance': [], 'timestamps': []} 
                     for node in selected_nodes}
    
    prev_z_causal = None
    
    with torch.no_grad():
        for i, snapshot in enumerate(snapshots[:min(50, len(snapshots))]):
            output = trainer.model(snapshot, prev_z_causal)
            
            z_causal = output['z_causal'].cpu().numpy()
            timestamp = snapshot['timestamps'].item()
            
            for node in selected_nodes:
                if node < len(z_causal):
                    importance = np.mean(np.abs(z_causal[node]))
                    evolution_data[node]['importance'].append(importance)
                    evolution_data[node]['timestamps'].append(timestamp)
            
            prev_z_causal = output['z_causal']
    
                   
    plt.figure(figsize=(15, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, node in enumerate(selected_nodes):
        if evolution_data[node]['importance']:
            plt.plot(evolution_data[node]['timestamps'],
                    evolution_data[node]['importance'],
                    label=f'User {node}',
                    color=colors[i % len(colors)],
                    linewidth=2.5, marker='o', markersize=4)
    
    plt.xlabel('Temps Normalisé', fontsize=14, fontweight='bold')
    plt.ylabel('Importance Causale', fontsize=14, fontweight='bold')
    plt.title('Évolution de l\'Importance Causale des Utilisateurs\n' +
             'causalvgae-explainer - Analyse Temporelle',
             fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Évolution temporelle sauvegardée: {save_path}")
