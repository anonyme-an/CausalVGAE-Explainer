   
                                                             
                                           
   

import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
       
                                                                 
       
    
    def __init__(self, time_dim: int):
           
             
                                                                
           
        super(TimeEncoder, self).__init__()
        self.time_dim = time_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
           
                             
        
             
                                                             
            
                
                                                                
           
                                                   
        if timestamps.dim() == 0:
            timestamps = timestamps.unsqueeze(0)
        
                                       
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)
        
        return self.encoder(timestamps)
