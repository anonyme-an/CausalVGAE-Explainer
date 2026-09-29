   
                                                
                                                       
   

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler


class WikipediaSnapshotLoader:
       
                                                                  
                                        
       
    
    def __init__(self,
                 csv_path: str,
                 node_features_path: str,
                 window_size: int = 100,
                 overlap: int = 50,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15):
           
             
                                                  
                                                                 
                                                        
                                                 
                                                             
                                                          
           
        self.csv_path = csv_path
        self.node_features_path = node_features_path
        self.window_size = window_size
        self.overlap = overlap
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
                             
        self.df = None
        self.node_features = None
        self.num_nodes = 0
        
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
           
                                 
        
                
                                                           
           
        print("📂 Chargement des données Wikipedia...")
        
                     
        self.df = pd.read_csv(self.csv_path)
        print(f"  Événements: {len(self.df)}")
        print(f"  Colonnes: {list(self.df.columns)}")
        
                                                                                  
        if 'u' in self.df.columns and 'i' in self.df.columns:
            edge_index = self.df[['u', 'i']].values.T
        elif 'user_id' in self.df.columns and 'item_id' in self.df.columns:
            edge_index = self.df[['user_id', 'item_id']].values.T
        else:
            raise ValueError(f"Colonnes requises 'u'/'i' ou 'user_id'/'item_id' non trouvées. Colonnes disponibles: {list(self.df.columns)}")
        
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
                                                                    
        if 'label' in self.df.columns:
            labels = torch.tensor(self.df['label'].values, dtype=torch.float32)
        elif 'state_label' in self.df.columns:
            labels = torch.tensor(self.df['state_label'].values, dtype=torch.float32)
        else:
                                                          
            labels = torch.zeros(len(self.df), dtype=torch.float32)
        
                                                                   
        if 'ts' in self.df.columns:
            timestamps = self.df['ts'].values
        elif 'timestamp' in self.df.columns:
            timestamps = self.df['timestamp'].values
        else:
            raise ValueError(f"Colonne 'ts' ou 'timestamp' non trouvée. Colonnes disponibles: {list(self.df.columns)}")
        
                       
        timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        timestamps = torch.tensor(timestamps, dtype=torch.float32)
        
                                    
        self.node_features = np.load(self.node_features_path)
        self.num_nodes = self.node_features.shape[0]
        
                                    
        scaler = StandardScaler()
        self.node_features = scaler.fit_transform(self.node_features)
        self.node_features = torch.tensor(self.node_features, dtype=torch.float32)
        
        print(f"  Nœuds: {self.num_nodes}")
        print(f"  Feature dim: {self.node_features.shape[1]}")
        print(f"  Arêtes: {edge_index.shape[1]}")
        
        return edge_index, labels, timestamps, self.node_features
    
    def create_snapshots(self, edge_index: torch.Tensor,
                        labels: torch.Tensor,
                        timestamps: torch.Tensor) -> List[Dict]:
           
                                                   
        
             
                                      
                               
                                   
            
                
                              
           
        print(f"\n📊 Création des snapshots (window={self.window_size}, overlap={self.overlap})...")
        
        num_events = edge_index.shape[1]
        snapshots = []
        
                            
        start_idx = 0
        snapshot_id = 0
        
        while start_idx < num_events:
            end_idx = min(start_idx + self.window_size, num_events)
            
                                                      
            window_edge_index = edge_index[:, start_idx:end_idx]
            window_labels = labels[start_idx:end_idx]
            window_timestamps = timestamps[start_idx:end_idx]
            
                                             
            active_nodes = torch.unique(window_edge_index)
            
                               
            snapshot = {
                'snapshot_id': snapshot_id,
                'edge_index': window_edge_index,
                'node_features': self.node_features.clone(),                  
                'labels': window_labels,
                'timestamps': window_timestamps.mean(),                                 
                'active_nodes': active_nodes,
                'num_events': end_idx - start_idx,
                'time_range': (timestamps[start_idx].item(), timestamps[end_idx-1].item())
            }
            
                                                        
            snapshot['source'] = window_edge_index[0]
            snapshot['target'] = window_edge_index[1]
            
            snapshots.append(snapshot)
            
                                  
            start_idx += (self.window_size - self.overlap)
            snapshot_id += 1
        
        print(f"  ✅ {len(snapshots)} snapshots créés")
        
        return snapshots
    
    def split_snapshots(self, snapshots: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
           
                                              
        
             
                                                  
            
                
                                                            
           
        num_snapshots = len(snapshots)
        
        train_end = int(num_snapshots * self.train_ratio)
        val_end = int(num_snapshots * (self.train_ratio + self.val_ratio))
        
        train_snapshots = snapshots[:train_end]
        val_snapshots = snapshots[train_end:val_end]
        test_snapshots = snapshots[val_end:]
        
        print(f"\n📂 Split des snapshots:")
        print(f"  Train: {len(train_snapshots)}")
        print(f"  Val  : {len(val_snapshots)}")
        print(f"  Test : {len(test_snapshots)}")
        
        return train_snapshots, val_snapshots, test_snapshots
    
    def prepare_for_training(self) -> Dict:
           
                                                      
        
                
                                                  
           
                             
        edge_index, labels, timestamps, node_features = self.load_data()
        
                             
        snapshots = self.create_snapshots(edge_index, labels, timestamps)
        
               
        train_snapshots, val_snapshots, test_snapshots = self.split_snapshots(snapshots)
        
        return {
            'train_snapshots': train_snapshots,
            'val_snapshots': val_snapshots,
            'test_snapshots': test_snapshots,
            'num_nodes': self.num_nodes,
            'node_feature_dim': node_features.shape[1],
            'all_snapshots': snapshots
        }


def load_wikipedia_data(csv_path: str,
                       node_features_path: str,
                       window_size: int = 100,
                       overlap: int = 50) -> Dict:
       
                                                                 
    
         
                            
                                           
                                   
                              
        
            
                         
       
    loader = WikipediaSnapshotLoader(
        csv_path=csv_path,
        node_features_path=node_features_path,
        window_size=window_size,
        overlap=overlap
    )
    
    return loader.prepare_for_training()


                                            
def print_snapshot_statistics(snapshots: List[Dict]):
                                                    
    print("\n📈 Statistiques des snapshots:")
    
    num_events = [s['num_events'] for s in snapshots]
    num_active_nodes = [len(s['active_nodes']) for s in snapshots]
    
    print(f"  Nombre de snapshots: {len(snapshots)}")
    print(f"  Événements par snapshot:")
    print(f"    Moyenne: {np.mean(num_events):.1f}")
    print(f"    Min: {np.min(num_events)}")
    print(f"    Max: {np.max(num_events)}")
    print(f"  Nœuds actifs par snapshot:")
    print(f"    Moyenne: {np.mean(num_active_nodes):.1f}")
    print(f"    Min: {np.min(num_active_nodes)}")
    print(f"    Max: {np.max(num_active_nodes)}")
