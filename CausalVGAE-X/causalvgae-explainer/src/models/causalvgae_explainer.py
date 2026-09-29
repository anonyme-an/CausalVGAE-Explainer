   
                                                        
                                                                 
                                            

                                   
                                 
                                  
                                    
                                  
                       
   

import torch
import torch.nn as nn
import torch.nn.functional as F
from .time_encoder import TimeEncoder
from .causal_vgae import CausalVGAE
from typing import Optional, Dict


class CausalVGAEExplainer(nn.Module):
       
                                                                  
    
                                                 
                                         
                                                                         
                                                                   
    
                                
                                      
                                                                      
                                        
                                      
                                           
       
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 latent_dim: int = 32,
                 causal_dim: int = 16,
                 time_dim: int = 16,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 target_model: Optional[nn.Module] = None):
           
             
                                                       
                                        
                                                    
                                                      
                                                      
                                                  
                                    
                                                                 
           
        super(CausalVGAEExplainer, self).__init__()
        
        self.causal_dim = causal_dim
        self.time_dim = time_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.target_model = target_model
        
                                       
        
                              
        self.time_encoder = TimeEncoder(time_dim)
        
                                                                             
        self.vgae = CausalVGAE(
            input_dim=input_dim + time_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            causal_dim=causal_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
                                                                       
        if target_model is None:
            self.target_predictor = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.target_predictor = None
    
    def forward(self, snapshot: Dict, prev_z_causal: Optional[torch.Tensor] = None) -> Dict:
           
                                                          
        
             
                                             
                                                   
                                                
                                                   
                                                                           
            
                
                                                     
           
        x = snapshot['node_features']
        edge_index = snapshot['edge_index']
        timestamps = snapshot['timestamps']
        
                                            
        if timestamps.dim() == 0:
            timestamps = timestamps.unsqueeze(0)
        
        time_emb = self.time_encoder(timestamps)
        
                                                       
        num_nodes = x.size(0)
        time_emb_expanded = time_emb.expand(num_nodes, -1)
        
                                                 
        x_temporal = torch.cat([x, time_emb_expanded], dim=-1)
        
                                                     
        vgae_output = self.vgae(x_temporal, edge_index)
        
                                                     
        if self.target_model is not None:
                                                           
            with torch.no_grad():
                if edge_index.size(1) > 0:
                                                            
                    target_embeddings = self.target_model.get_embeddings(
                        x, edge_index, timestamps
                    )
                    
                                            
                    src_nodes = edge_index[0]
                    dst_nodes = edge_index[1]
                    
                                                           
                    time_emb_edges = time_emb.expand(edge_index.size(1), -1)
                    edge_features = torch.cat([
                        target_embeddings[src_nodes],
                        target_embeddings[dst_nodes],
                        time_emb_edges
                    ], dim=-1)
                    
                    predictions = self.target_model.predictor(edge_features)
                else:
                    predictions = torch.zeros(1, 1, device=x.device)
        else:
                                            
            predictions = self.target_predictor(vgae_output['z'])
        
                                                
        causal_info_flow = self.vgae.compute_causal_information_flow(
            vgae_output['z_causal'], predictions
        )
        
                                           
        temporal_loss = self.temporal_smoothness_loss(
            vgae_output['z_causal'], prev_z_causal
        )
        
                                
        return {
            **vgae_output,                                                                                  
            'predictions': predictions,
            'causal_info_flow': causal_info_flow,
            'temporal_loss': temporal_loss,
            'time_emb': time_emb,
            'x_temporal': x_temporal
        }
    
    def temporal_smoothness_loss(self, z_causal_t: torch.Tensor,
                                z_causal_t_prev: Optional[torch.Tensor]) -> torch.Tensor:
           
                                                            
                                                     
        
             
                                           
                                                                   
            
                
                                     
           
        if z_causal_t_prev is None:
            return torch.tensor(0.0, device=z_causal_t.device)
        
                                                       
        min_size = min(z_causal_t.size(0), z_causal_t_prev.size(0))
        z_t = z_causal_t[:min_size]
        z_prev = z_causal_t_prev[:min_size]
        
                                             
        return torch.mean((z_t - z_prev) ** 2)
    
    def explain(self, snapshot: Dict, threshold: float = 0.5) -> Dict:
           
                                                       
        
             
                                          
                                                   
            
                
                                        
           
        self.eval()
        with torch.no_grad():
            output = self.forward(snapshot)
            
            causal_mask = output['causal_mask']
            binary_mask = (causal_mask > threshold).float()
            
                              
            important_nodes = torch.where(binary_mask.squeeze() > 0)[0]
            
                                 
            importance_scores = causal_mask.squeeze()
            
                                                 
            edge_importance = output.get('edge_importance', None)
            
            return {
                'causal_mask': causal_mask,
                'binary_mask': binary_mask,
                'important_nodes': important_nodes,
                'importance_scores': importance_scores,
                'edge_importance': edge_importance,
                'z_causal': output['z_causal'],
                'z_spurious': output['z_spurious'],
                'predictions': output['predictions'],
                'causal_info_flow': output['causal_info_flow'],
                'attention_weights': output.get('attention_weights', None),
                'num_important_nodes': len(important_nodes),
                'avg_importance': importance_scores.mean().item(),
                'max_importance': importance_scores.max().item()
            }
    
    def compute_fidelity_metrics(self, snapshot: Dict) -> Dict:
           
                                                    
        
             
                                         
            
                
                                                        
           
        self.eval()
        with torch.no_grad():
            output = self.forward(snapshot)
            
            original_pred = output['predictions']
            causal_mask = output['causal_mask']
            
                                                  
            masked_pred = original_pred * causal_mask[:original_pred.size(0)]
            
                                                     
            inverted_mask = 1.0 - causal_mask[:original_pred.size(0)]
            non_causal_pred = original_pred * inverted_mask
            
                                  
            from ..metrics.fidelity import compute_fidelity_plus, compute_fidelity_minus
            from ..metrics.sparsity import compute_sparsity
            from ..metrics.characterization import compute_characterization_score
            
            labels = snapshot.get('labels', None)
            
            fid_plus = compute_fidelity_plus(original_pred, masked_pred, labels)
            fid_minus = compute_fidelity_minus(original_pred, non_causal_pred, labels)
            
            num_candidates = snapshot['edge_index'].size(1)
            sparsity = compute_sparsity(causal_mask, num_candidates)
            
            char = compute_characterization_score(
                fid_plus.item() if hasattr(fid_plus, 'item') else fid_plus,
                fid_minus.item() if hasattr(fid_minus, 'item') else fid_minus
            )
            
            return {
                'fidelity_plus': fid_plus.item() if hasattr(fid_plus, 'item') else fid_plus,
                'fidelity_minus': fid_minus.item() if hasattr(fid_minus, 'item') else fid_minus,
                'sparsity': sparsity.item() if hasattr(sparsity, 'item') else sparsity,
                'characterization': char
            }
    
    def set_target_model(self, target_model: nn.Module):
           
                                           
        
             
                                                          
           
        self.target_model = target_model
        self.target_model.eval()                         
        
                                                  
        for param in self.target_model.parameters():
            param.requires_grad = False
        
        print(f"✅ Modèle cible configuré: {type(target_model).__name__}")
