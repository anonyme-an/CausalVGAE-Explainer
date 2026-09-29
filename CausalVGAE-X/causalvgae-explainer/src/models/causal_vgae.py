   
                                                 
                                           
                                                 

                                   
                                                                  
                                             
                                                
   

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
import math


class MultiHeadAttention(nn.Module):
       
                                                     
                                                           
       
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim doit être divisible par num_heads"
        
        self.scale = math.sqrt(self.head_dim)
        
                                            
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
           
                                          
        
             
                                                        
                                              
            
                
                                       
           
        batch_size = 1
        num_nodes = x.size(0)
        
                                       
        x = x.unsqueeze(0)                             
        
                             
        Q = self.q_proj(x)                             
        K = self.k_proj(x)
        V = self.v_proj(x)
        
                                                                    
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
                                                                      
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
                            
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
                                           
        context = torch.matmul(attention_weights, V)                                       
        
                                         
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.embed_dim)
        output = self.out_proj(context)
        output = self.dropout(output)
        
        return output.squeeze(0), attention_weights.squeeze(0)


class CausalEncoder(nn.Module):
       
                                                   
                                                                                     
       
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super(CausalEncoder, self).__init__()
        
                                                       
        self.gc1 = GCNConv(input_dim, hidden_dim)
        
                                                                        
        self.gat = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            dropout=dropout
        )
        
                                                                        
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
                             
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
                              
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple:
           
                                         
        
                
                                                
           
                                      
        h = F.relu(self.gc1(x, edge_index))
        h = self.dropout(h)
        
                                            
        h = F.relu(self.gat(h, edge_index))
        h = self.norm1(h)
        
                                                             
        attn_output, attention_weights = self.self_attention(h)
        h = self.norm2(h + attn_output)                       
        
                      
        h = h + self.ffn(h)                       
        
        return h, attention_weights


class CausalVGAE(nn.Module):
       
                                                                        
    
                                
                                                         
                                                      
                                    
                                  
    
                                                                     
       
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 latent_dim: int = 32,
                 causal_dim: int = 16,
                 num_heads: int = 4,
                 dropout: float = 0.2):
           
             
                                                                        
                                        
                                                           
                                                                 
                                                  
                                    
           
        super(CausalVGAE, self).__init__()
        
        self.causal_dim = causal_dim
        self.spurious_dim = latent_dim - causal_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
                                       
        self.causal_encoder = CausalEncoder(input_dim, hidden_dim, num_heads, dropout)
        
                                                   
        self.gc_mu = GATv2Conv(hidden_dim, latent_dim, heads=1, concat=False, dropout=dropout)
        self.gc_logvar = GATv2Conv(hidden_dim, latent_dim, heads=1, concat=False, dropout=dropout)
        
                       
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
                                                             
        self.causal_mask_attention = MultiHeadAttention(causal_dim, num_heads=2, dropout=dropout)
        self.causal_mask_generator = nn.Sequential(
            nn.Linear(causal_dim, hidden_dim),
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
        
                                                  
        self.causal_predictor = nn.Sequential(
            nn.Linear(causal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
                                                  
        self.edge_importance = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple:
           
                                                                       
        
             
                                                        
                                                         
            
                
                                                                             
           
                                                         
        h, attention_weights = self.causal_encoder(x, edge_index)
        
                                                      
        mu = self.gc_mu(h, edge_index)
        logvar = self.gc_logvar(h, edge_index)
        
        return mu, logvar, attention_weights
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
           
                                             
                                 
        
             
                                                
                                                         
            
                
                                                  
           
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
           
                                                          
        
             
                                                               
            
                
                                          
           
        return self.decoder(z)
    
    def compute_edge_importance(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
           
                                                  
        
             
                                                               
                                                         
            
                
                                                
           
        src, dst = edge_index
                                                            
        edge_features = torch.cat([z[src], z[dst]], dim=-1)
        return self.edge_importance(edge_features)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
           
                                                
        
             
                                                    
                                          
            
                
                                   
                                      
                                      
                                         
                                                    
                                    
                                                        
                                                    
                                                          
                                                                
           
                                               
        mu, logvar, attention_weights = self.encode(x, edge_index)
        
                                    
        z = self.reparameterize(mu, logvar)
        
                                             
        z_causal = z[:, :self.causal_dim]
        z_spurious = z[:, self.causal_dim:]
        
                                                            
                                                           
        z_causal_attn, mask_attention_weights = self.causal_mask_attention(z_causal)
        causal_mask = self.causal_mask_generator(z_causal_attn)
        
                                             
        edge_imp = self.compute_edge_importance(z, edge_index)
        
                                
        recon = self.decode(z)
        
                                    
                                                  
        causal_pred = self.causal_predictor(z_causal)
        
        return {
            'z': z,
            'z_causal': z_causal,
            'z_spurious': z_spurious,
            'mu': mu,
            'logvar': logvar,
            'recon': recon,
            'causal_mask': causal_mask,
            'edge_importance': edge_imp,
            'causal_pred': causal_pred,
            'attention_weights': attention_weights,
            'mask_attention_weights': mask_attention_weights
        }
    
    def compute_causal_information_flow(self, z_causal: torch.Tensor, 
                                       predictions: torch.Tensor) -> torch.Tensor:
           
                                                                   
                                                                 
        
             
                                              
                                              
            
                
                                         
           
                              
        z_causal_flat = z_causal.reshape(-1)
        pred_flat = predictions.reshape(-1)
        
                                     
        min_size = min(z_causal_flat.size(0), pred_flat.size(0))
        z_causal_flat = z_causal_flat[:min_size]
        pred_flat = pred_flat[:min_size]
        
                                             
        if min_size > 1:
                                 
            z_mean = z_causal_flat.mean()
            pred_mean = pred_flat.mean()
            
            z_centered = z_causal_flat - z_mean
            pred_centered = pred_flat - pred_mean
            
                         
            numerator = (z_centered * pred_centered).sum()
            denominator = torch.sqrt((z_centered ** 2).sum() * (pred_centered ** 2).sum())
            
            corr = numerator / (denominator + 1e-8)
        else:
            corr = torch.tensor(0.0, device=z_causal.device)
        
                                                       
                                       
        info_flow = -torch.log(1 - corr.abs() + 1e-8)
        
        return info_flow
    
    def get_explanation(self, x: torch.Tensor, edge_index: torch.Tensor,
                       threshold: float = 0.5) -> dict:
           
                                                   
        
             
                                 
                                          
                                                   
            
                
                                      
           
        self.eval()
        with torch.no_grad():
            output = self.forward(x, edge_index)
            
            causal_mask = output['causal_mask']
            binary_mask = (causal_mask > threshold).float()
            
            return {
                'causal_mask': causal_mask,
                'binary_mask': binary_mask,
                'z_causal': output['z_causal'],
                'important_nodes': torch.where(binary_mask.squeeze() > 0)[0],
                'importance_scores': causal_mask.squeeze()
            }
