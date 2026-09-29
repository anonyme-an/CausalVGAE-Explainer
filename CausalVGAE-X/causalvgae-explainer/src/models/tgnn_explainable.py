   
                                                          
                                                   
   

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class TimeEncoder(nn.Module):
                                               
    def __init__(self, time_dim: int):
        super(TimeEncoder, self).__init__()
        self.time_dim = time_dim
        self.linear = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
           
             
                                                                   
                
                                                  
           
                                          
        if timestamps.dim() == 0:
                                               
            timestamps = timestamps.unsqueeze(0).unsqueeze(1)
        elif timestamps.dim() == 1:
                                                   
            timestamps = timestamps.unsqueeze(1)
        return self.linear(timestamps)


class TGNNExplainable(nn.Module):
       
                                                              
                                             
       
    
    def __init__(self, 
                 num_nodes: int,
                 node_feature_dim: int,
                 embedding_dim: int = 64,
                 time_dim: int = 32,
                 num_gat_layers: int = 4,
                 gru_hidden_dim: int = 64,
                 dropout: float = 0.5):
           
             
                                            
                                                              
                                                   
                                                      
                                                 
                                                   
                                    
           
        super(TGNNExplainable, self).__init__()
        
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        
                                              
        self.initial_transform = nn.Linear(node_feature_dim, embedding_dim)
        
                              
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        
                           
        self.time_encoder = TimeEncoder(time_dim)
        
                     
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_dim = embedding_dim if i == 0 else embedding_dim * 4
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=embedding_dim,
                    heads=4,
                    concat=True,
                    dropout=dropout
                )
            )
        
                                          
        self.gru_input_dim = embedding_dim * 4
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True
        )
        
                                   
        total_dim = 2 * embedding_dim + time_dim
        self.edge_predictor = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def get_embeddings(self, node_features: torch.Tensor, 
                       edge_index: torch.Tensor,
                       timestamps: torch.Tensor = None) -> torch.Tensor:
           
                                                                  
        
             
                                                                      
                                                         
                                             
            
                
                                                          
           
                                 
        x = self.initial_transform(node_features)
        
                         
        for layer in self.gat_layers:
            x = layer(x, edge_index)
        
        return x
    
    def forward(self, source: torch.Tensor, 
                target: torch.Tensor,
                timestamps: torch.Tensor,
                edge_index: torch.Tensor,
                node_features: torch.Tensor) -> torch.Tensor:
           
                                    
        
             
                                                         
                                                        
                                                           
                                                                   
                                                                      
            
                
                                       
           
                                               
        source_emb = self.node_embedding(source)
        target_emb = self.node_embedding(target)
        
                           
                                                                  
        if timestamps.dim() == 0 or (timestamps.dim() == 1 and timestamps.size(0) == 1):
                                                                      
            batch_size = source.size(0)
            if timestamps.dim() == 0:
                timestamps = timestamps.unsqueeze(0).expand(batch_size)
            else:
                timestamps = timestamps.expand(batch_size)
        
        time_emb = self.time_encoder(timestamps)
        
                                    
        combined = torch.cat([source_emb, target_emb, time_emb], dim=1)
        
                    
        return self.edge_predictor(combined)
    
    def predict_edges(self, source: torch.Tensor, 
                      target: torch.Tensor,
                      timestamps: torch.Tensor,
                      edge_index: torch.Tensor,
                      node_features: torch.Tensor) -> torch.Tensor:
           
                                              
        
                
                                     
           
        logits = self.forward(source, target, timestamps, edge_index, node_features)
        return torch.sigmoid(logits).squeeze()


class TGNNWrapper(nn.Module):
       
                                                                          
       
    def __init__(self, tgnn_model: TGNNExplainable):
        super(TGNNWrapper, self).__init__()
        self.tgnn = tgnn_model
                                                   
        self.predictor = tgnn_model.edge_predictor
        
    def forward(self, snapshot: dict) -> torch.Tensor:
           
                                                        
        
             
                                             
                               
                            
                            
                                    
                                    
                
                
                       
           
        node_features = snapshot['node_features']
        edge_index = snapshot['edge_index']
        timestamps = snapshot.get('timestamps', torch.zeros(1, device=node_features.device))
        
                                                       
        if 'source' in snapshot and 'target' in snapshot:
            return self.tgnn.predict_edges(
                snapshot['source'],
                snapshot['target'],
                timestamps,
                edge_index,
                node_features
            )
        
                                         
        embeddings = self.tgnn.get_embeddings(node_features, edge_index, timestamps)
        return embeddings.mean(dim=0, keepdim=True)
    
    def get_embeddings(self, node_features: torch.Tensor,
                       edge_index: torch.Tensor,
                       timestamps: torch.Tensor = None) -> torch.Tensor:
                                               
        return self.tgnn.get_embeddings(node_features, edge_index, timestamps)
