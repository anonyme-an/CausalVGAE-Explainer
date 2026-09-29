   
                                                
                                                            
                               
   

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch_geometric.utils import subgraph


class ExplainabilityMetrics:
       
                                                             
    
              
                                                                       
                                                                      
                                           
                                                            
       
    
    def __init__(self, target_model: nn.Module, device: str = 'cuda'):
           
             
                                                      
                                    
           
        self.target_model = target_model
        self.target_model.eval()
        self.device = device
        
    def fidelity_plus(self, 
                      snapshot: Dict,
                      causal_mask: torch.Tensor,
                      original_pred: torch.Tensor,
                      threshold: float = 0.5) -> float:
           
                                                           
        
                                                                              
        
                                                                     
        
             
                                        
                                                     
                                               
                                                   
            
                
                                        
           
        with torch.no_grad():
                                         
            important_nodes = (causal_mask.squeeze() > threshold).nonzero(as_tuple=True)[0]
            
            if len(important_nodes) == 0:
                return 0.0
            
                                     
            sub_edge_index, _ = subgraph(
                important_nodes,
                snapshot['edge_index'],
                relabel_nodes=True,
                num_nodes=snapshot['node_features'].size(0)
            )
            
                                                           
            causal_snapshot = {
                'node_features': snapshot['node_features'][important_nodes],
                'edge_index': sub_edge_index,
                'timestamps': snapshot.get('timestamps', torch.zeros(1, device=self.device))
            }
            
                                                    
            if 'source' in snapshot and 'target' in snapshot:
                                                              
                node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(important_nodes)}
                
                                                                               
                valid_pairs = []
                for src, tgt in zip(snapshot['source'], snapshot['target']):
                    if src.item() in node_map and tgt.item() in node_map:
                        valid_pairs.append((node_map[src.item()], node_map[tgt.item()]))
                
                if len(valid_pairs) > 0:
                    valid_pairs = torch.tensor(valid_pairs, device=self.device)
                    causal_snapshot['source'] = valid_pairs[:, 0]
                    causal_snapshot['target'] = valid_pairs[:, 1]
            
                                                  
            try:
                causal_pred = self.target_model(causal_snapshot)
                
                                    
                diff = torch.abs(original_pred - causal_pred).mean()
                fidelity = 1.0 - (diff / (torch.abs(original_pred).mean() + 1e-8))
                
                return float(torch.clamp(fidelity, 0.0, 1.0))
            except:
                                                          
                return 0.0
    
    def fidelity_minus(self,
                       snapshot: Dict,
                       causal_mask: torch.Tensor,
                       original_pred: torch.Tensor,
                       threshold: float = 0.5) -> float:
           
                                                                    
        
                                                                    
        
                                                                                        
        
             
                                        
                                      
                                               
                                                   
            
                
                           
           
        with torch.no_grad():
                                                
            non_causal_nodes = (causal_mask.squeeze() <= threshold).nonzero(as_tuple=True)[0]
            
            if len(non_causal_nodes) == 0:
                return 1.0                                                  
            
                                                
            sub_edge_index, _ = subgraph(
                non_causal_nodes,
                snapshot['edge_index'],
                relabel_nodes=True,
                num_nodes=snapshot['node_features'].size(0)
            )
            
                                                 
            non_causal_snapshot = {
                'node_features': snapshot['node_features'][non_causal_nodes],
                'edge_index': sub_edge_index,
                'timestamps': snapshot.get('timestamps', torch.zeros(1, device=self.device))
            }
            
                                                   
            try:
                non_causal_pred = self.target_model(non_causal_snapshot)
                
                                    
                diff = torch.abs(original_pred - non_causal_pred).mean()
                fidelity_minus = diff / (torch.abs(original_pred).mean() + 1e-8)
                
                return float(torch.clamp(fidelity_minus, 0.0, 2.0))
            except:
                return 0.0
    
    def sparsity(self, causal_mask: torch.Tensor, 
                 candidate_events: int = None,
                 threshold: float = 0.5) -> float:
           
                                                      
        
                                          
                                
        
                                                          
        
             
                                                                       
                                                                           
                                                                   
                                                                   
            
                
                                                      
           
        binary_mask = (causal_mask.squeeze() > threshold).float()
        num_selected = int(binary_mask.sum().item())
        n_cand = candidate_events if candidate_events is not None else int(binary_mask.numel())
        
        if n_cand == 0:
            return 0.0
        
        return float(num_selected) / float(n_cand)
    
    def characterization(self,
                        causal_mask: torch.Tensor,
                        edge_index: torch.Tensor,
                        threshold: float = 0.5) -> float:
           
                                                                     
        
                                                           
        
                                                    
        
             
                                      
                                                            
                                                   
            
                
                      
           
                          
        important_nodes = (causal_mask.squeeze() > threshold).nonzero(as_tuple=True)[0]
        num_important = len(important_nodes)
        
        if num_important <= 1:
            return 0.0
        
                                              
        src, dst = edge_index
        
                                                                
        important_mask = torch.zeros(causal_mask.size(0), device=causal_mask.device, dtype=torch.bool)
        important_mask[important_nodes] = True
        
        edges_in_subgraph = (important_mask[src] & important_mask[dst]).sum().item()
        
                                
        max_edges_subgraph = num_important * (num_important - 1) / 2
        density_subgraph = edges_in_subgraph / max_edges_subgraph if max_edges_subgraph > 0 else 0.0
        
                                   
        total_nodes = causal_mask.size(0)
        total_edges = edge_index.size(1)
        max_edges_total = total_nodes * (total_nodes - 1) / 2
        density_total = total_edges / max_edges_total if max_edges_total > 0 else 0.0
        
               
        char = density_subgraph / (density_total + 1e-8)
        
        return float(char)
    
    def temporal_continuity(self,
                           causal_masks: List[torch.Tensor],
                           threshold: float = 0.5) -> float:
           
                                                                        
        
                                                         
        
             
                                                                       
                                              
            
                
                                          
           
        if len(causal_masks) < 2:
            return 1.0
        
        jaccard_scores = []
        
        for i in range(len(causal_masks) - 1):
            mask_t = (causal_masks[i].squeeze() > threshold).float()
            mask_t1 = (causal_masks[i+1].squeeze() > threshold).float()
            
                                
            intersection = (mask_t * mask_t1).sum()
            union = ((mask_t + mask_t1) > 0).float().sum()
            
            jaccard = (intersection / (union + 1e-8)).item()
            jaccard_scores.append(jaccard)
        
        return np.mean(jaccard_scores)
    
    def compute_all_metrics(self,
                           snapshots: List[Dict],
                           explainer_model: nn.Module,
                           threshold: float = 0.5) -> Dict[str, float]:
           
                                                                  
        
             
                                         
                                                        
                                             
            
                
                                                
           
        explainer_model.eval()
        
        fid_plus_scores = []
        fid_minus_scores = []
        sparsity_scores = []
        char_scores = []
        causal_masks = []
        
        with torch.no_grad():
            prev_z_causal = None
            
            for snapshot in snapshots:
                                         
                snapshot = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in snapshot.items()}
                
                                       
                outputs = explainer_model(snapshot, prev_z_causal)
                causal_mask = outputs['causal_mask']
                
                                                      
                original_pred = self.target_model(snapshot)
                
                                        
                fid_plus = self.fidelity_plus(snapshot, causal_mask, original_pred, threshold)
                fid_minus = self.fidelity_minus(snapshot, causal_mask, original_pred, threshold)
                spars = self.sparsity(causal_mask, threshold)
                char = self.characterization(causal_mask, snapshot['edge_index'], threshold)
                
                fid_plus_scores.append(fid_plus)
                fid_minus_scores.append(fid_minus)
                sparsity_scores.append(spars)
                char_scores.append(char)
                causal_masks.append(causal_mask.cpu())
                
                prev_z_causal = outputs['z_causal']
        
                               
        temporal_cont = self.temporal_continuity(causal_masks, threshold)
        
                  
        metrics = {
            'fidelity_plus': np.mean(fid_plus_scores),
            'fidelity_minus': np.mean(fid_minus_scores),
            'sparsity': np.mean(sparsity_scores),
            'characterization': np.mean(char_scores),
            'temporal_continuity': temporal_cont,
            'fid_plus_std': np.std(fid_plus_scores),
            'fid_minus_std': np.std(fid_minus_scores),
            'sparsity_std': np.std(sparsity_scores),
            'char_std': np.std(char_scores)
        }
        
        return metrics
    
    def evaluate_and_save(self,
                         snapshots: List[Dict],
                         explainer_model: nn.Module,
                         save_path: str,
                         threshold: float = 0.5) -> Dict:
           
                                          
        
             
                                        
                                             
                                           
                                             
            
                
                                      
           
        print("\n📊 Évaluation des métriques d'explicabilité...")
        
        metrics = self.compute_all_metrics(snapshots, explainer_model, threshold)
        
                   
        print("\n✨ Résultats:")
        print(f"  Fidelity+      : {metrics['fidelity_plus']:.4f} ± {metrics['fid_plus_std']:.4f}")
        print(f"  Fidelity-      : {metrics['fidelity_minus']:.4f} ± {metrics['fid_minus_std']:.4f}")
        print(f"  Sparsity       : {metrics['sparsity']:.4f} ± {metrics['sparsity_std']:.4f}")
        print(f"  Characterization: {metrics['characterization']:.4f} ± {metrics['char_std']:.4f}")
        print(f"  Temporal Cont  : {metrics['temporal_continuity']:.4f}")
        
                    
        torch.save(metrics, save_path)
        print(f"\n💾 Métriques sauvegardées: {save_path}")
        
        return metrics


def create_comparative_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
       
                                            
    
         
                                                              
        
            
                                   
       
    table = "| Méthode | Dataset | Modèle | Fid+ | Fid- | Sparsity | Char |\n"
    table += "|---------|---------|--------|------|------|----------|------|\n"
    
    for method, datasets in results.items():
        for dataset, models in datasets.items():
            for model, metrics in models.items():
                fid_plus = metrics.get('fidelity_plus', 0.0)
                fid_minus = metrics.get('fidelity_minus', 0.0)
                sparsity = metrics.get('sparsity', 0.0)
                char = metrics.get('characterization', 0.0)
                
                table += f"| {method} | {dataset} | {model} | {fid_plus:.2f} | {fid_minus:.2f} | {sparsity:.2f} | {char:.2f} |\n"
    
    return table
