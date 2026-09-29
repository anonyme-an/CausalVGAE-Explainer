   
                                                 
   

import os
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional


class WikipediaDataset(Dataset):
                                                     
    
    def __init__(self, csv_path: str):
           
             
                                               
           
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        
        self.csv_path = csv_path
        self.name = "Wikipedia"
        self.n_features = 172
        
        print(f"📊 Dataset Wikipedia chargé: {csv_path}")


class UCIMessagesDataset(Dataset):
                                                        
    
    def __init__(self, csv_path: Optional[str] = None):
           
             
                                                  
           
        if csv_path and not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        
        self.csv_path = csv_path
        self.name = "UCI-Messages"
        self.n_features = 50                              
        
        print(f"📊 Dataset UCI-Messages configuré")
        if csv_path is None:
            print("⚠️  Chemin du dataset non fourni - à configurer")
