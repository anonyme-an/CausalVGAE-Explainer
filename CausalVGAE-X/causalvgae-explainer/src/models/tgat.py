   
                                       
                                                               
                                                     

                                        
                                            
                                   
                                   
   

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import math


class TemporalAttentionLayer(nn.Module):
       
                                       
                                                                          
       
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(TemporalAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.scale = math.sqrt(self.head_dim)
        
                                                      
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.time_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None) -> tuple:
           
                                         
           
        batch_size = 1
        num_nodes = x.size(0)
        
                                   
        if time_emb is not None:
            time_bias = self.time_proj(time_emb)
            x = x + time_bias
        
        x = x.unsqueeze(0)
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
                                
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.embed_dim)
        output = self.out_proj(context)
        output = self.dropout(output)
        
        return output.squeeze(0), attention_weights.squeeze(0)


class TGAT(nn.Module):
       
                                    
                                                                           
    
                                                    
       
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 time_dim: int = 16,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.3):
           
             
                                                       
                                        
                                                      
                                                  
                                             
                                    
           
        super(TGAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
                           
        from .time_encoder import TimeEncoder
        self.time_encoder = TimeEncoder(time_dim)
        
                             
        self.input_proj = nn.Linear(input_dim + time_dim, hidden_dim)
        
                                                   
        self.gat_layers = nn.ModuleList()
        self.temporal_attention_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        
        for i in range(num_layers):
                                                
            if i == num_layers - 1:
                                                
                self.gat_layers.append(GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout
                ))
            else:
                self.gat_layers.append(GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout
                ))
            
                                        
            self.temporal_attention_layers.append(
                TemporalAttentionLayer(hidden_dim, num_heads, dropout)
            )
            
                           
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
            
                          
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            ))
        
                                                                      
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + time_dim, hidden_dim),
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
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                timestamps: torch.Tensor) -> torch.Tensor:
           
                                                                
        
             
                                                        
                                                         
                                  
            
                
                                                        
           
        return self.get_embeddings(x, edge_index, timestamps)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor,
                      timestamps: torch.Tensor) -> torch.Tensor:
           
                                                                
        
             
                                 
                                          
                                  
            
                
                                              
           
                           
        time_emb = self.time_encoder(timestamps)
        num_nodes = x.size(0)
        time_emb_expanded = time_emb.expand(num_nodes, -1)
        
                                       
        x_temporal = torch.cat([x, time_emb_expanded], dim=-1)
        
                             
        h = self.input_proj(x_temporal)
        h = self.dropout(F.gelu(h))
        
                                                                  
        all_attention_weights = []
        for i in range(self.num_layers):
            h_res = h
            
                               
            h = F.gelu(self.gat_layers[i](h, edge_index))
            
                                        
            h_attn, attn_weights = self.temporal_attention_layers[i](h, time_emb_expanded)
            all_attention_weights.append(attn_weights)
            
            h = self.norm_layers[i](h + h_attn)                   
            
                                        
            h = h + self.ffn_layers[i](h)
            
                                    
            if h.size(-1) == h_res.size(-1):
                h = h + h_res
        
        return h
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor,
                             timestamps: torch.Tensor) -> tuple:
           
                                                             
        
             
                                 
                                          
                                  
            
                
                                                      
           
        time_emb = self.time_encoder(timestamps)
        num_nodes = x.size(0)
        time_emb_expanded = time_emb.expand(num_nodes, -1)
        
        x_temporal = torch.cat([x, time_emb_expanded], dim=-1)
        h = self.input_proj(x_temporal)
        h = self.dropout(F.gelu(h))
        
        all_attention_weights = []
        for i in range(self.num_layers):
            h_res = h
            h = F.gelu(self.gat_layers[i](h, edge_index))
            h_attn, attn_weights = self.temporal_attention_layers[i](h, time_emb_expanded)
            all_attention_weights.append(attn_weights)
            h = self.norm_layers[i](h + h_attn)
            h = h + self.ffn_layers[i](h)
            if h.size(-1) == h_res.size(-1):
                h = h + h_res
        
        return h, all_attention_weights
    
    def predict_edges(self, x: torch.Tensor, edge_index: torch.Tensor,
                     timestamps: torch.Tensor, src_nodes: torch.Tensor,
                     dst_nodes: torch.Tensor) -> torch.Tensor:
           
                                                           
        
             
                                 
                                           
                                  
                                   
                                        
            
                
                                       
           
        embeddings = self.get_embeddings(x, edge_index, timestamps)
        time_emb = self.time_encoder(timestamps)
        time_emb_edges = time_emb.expand(src_nodes.size(0), -1)
        
        edge_features = torch.cat([
            embeddings[src_nodes],
            embeddings[dst_nodes],
            time_emb_edges
        ], dim=-1)
        
        return self.predictor(edge_features)
