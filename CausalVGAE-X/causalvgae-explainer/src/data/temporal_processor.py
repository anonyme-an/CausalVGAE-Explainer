
   
                                                            
                                                           
   

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class TemporalDataProcessor:
       
                                                                         
                                                
       
    
    def __init__(self, window_size: int = 100, overlap: int = 50):
           
             
                                                        
                                                    
           
        self.window_size = window_size
        self.overlap = overlap
        self.scaler = StandardScaler()
        
    def load_and_preprocess(self, csv_path: str) -> Dict:
           
                                              
        
             
                                                
            
                
                                                          
           
        print(f"📂 Chargement du dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Shape: {df.shape}")
        print(f"   Colonnes: {df.columns.tolist()}")
        
                                              
        user_ids = df['user_id'].values
        item_ids = df['item_id'].values
        timestamps = df['timestamp'].values
        labels = df['state_label'].values
        
                  
        features_raw = df['comma_separated_list_of_features'].values
        
                                                  
        timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        
                                                                          
        labels = (labels - labels.min()) / (labels.max() - labels.min())
                                                     
        labels = (labels > 0.5).astype(np.float32)
        
        return {
            'user_ids': user_ids,
            'item_ids': item_ids,
            'timestamps': timestamps,
            'labels': labels,
            'features': features_raw,
            'n_features': 172,                      
            'dataset_name': 'Wikipedia'
        }
    
    def create_temporal_snapshots(self, data: Dict) -> List[Dict]:
           
                                                            
        
             
                                     
            
                
                                        
           
        print("📸 Création des snapshots temporels...")
        
        timestamps = data['timestamps']
        n_windows = max(1, (len(timestamps) - self.window_size) // 
                       (self.window_size - self.overlap) + 1)
        
        snapshots = []
        for i in range(n_windows):
            start_idx = i * (self.window_size - self.overlap)
            end_idx = min(start_idx + self.window_size, len(timestamps))
            
                                                
            window_data = {
                'user_ids': data['user_ids'][start_idx:end_idx],
                'item_ids': data['item_ids'][start_idx:end_idx],
                'timestamps': data['timestamps'][start_idx:end_idx],
                'labels': data['labels'][start_idx:end_idx],
                'features': data['features'][start_idx:end_idx],
                'window_id': i,
                'time_range': (timestamps[start_idx], timestamps[end_idx-1])
            }
            
                                                     
            adj_matrix, node_features = self._create_graph_structure(window_data)
            window_data['adj_matrix'] = adj_matrix
            window_data['node_features'] = node_features
            
            snapshots.append(window_data)
        
        print(f"   ✅ {len(snapshots)} snapshots créés")
        return snapshots
    
    def _create_graph_structure(self, window_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
           
                                                                         
        
             
                                                         
            
                
                                                           
           
        user_ids = window_data['user_ids']
        item_ids = window_data['item_ids']
        features_str = window_data['features']
        
                             
        parsed_features = []
        for feat_str in features_str:
            if isinstance(feat_str, str):
                try:
                    feat_array = np.array([float(x) for x in feat_str.split(',')])
                except:
                    feat_array = np.zeros(172)
            else:
                feat_array = np.zeros(172)
                feat_array[0] = float(feat_str) if not np.isnan(float(feat_str)) else 0.0
            parsed_features.append(feat_array)
        
        features = np.array(parsed_features)
        
                                                           
        if features.shape[1] != 172:
            new_features = np.zeros((features.shape[0], 172))
            min_cols = min(features.shape[1], 172)
            new_features[:, :min_cols] = features[:, :min_cols]
            features = new_features
        
                                    
        all_nodes = np.unique(np.concatenate([user_ids, item_ids]))
        node_mapping = {node: idx for idx, node in enumerate(all_nodes)}
        n_nodes = len(all_nodes)
        
                                                  
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for u, i in zip(user_ids, item_ids):
            u_idx = node_mapping[u]
            i_idx = node_mapping[i]
            adj_matrix[u_idx, i_idx] = 1
            adj_matrix[i_idx, u_idx] = 1
        
                                       
        node_features = np.zeros((n_nodes, features.shape[1]))
        for node, idx in node_mapping.items():
            user_mask = user_ids == node
            item_mask = item_ids == node
            node_mask = user_mask | item_mask
            if np.any(node_mask):
                node_features[idx] = features[node_mask].mean(axis=0)
        
        return adj_matrix, node_features
    
    def prepare_for_pytorch(self, snapshots: List[Dict], device: torch.device) -> List[Dict]:
           
                                                   
        
             
                                               
                                                
            
                
                                      
           
        print("🔧 Conversion en tenseurs PyTorch...")
        
        prepared_snapshots = []
        for snapshot in snapshots:
            adj_matrix = snapshot['adj_matrix']
            node_features = snapshot['node_features']
            labels = snapshot['labels']
            timestamps = snapshot['timestamps']
            
                                                            
            edge_index = []
            for i in range(adj_matrix.shape[0]):
                for j in range(adj_matrix.shape[1]):
                    if adj_matrix[i, j] > 0:
                        edge_index.append([i, j])
            
            if len(edge_index) == 0:
                                                    
                edge_index = [[0, 0]]
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)
            
                                                                                    
                                                                          
            user_ids = snapshot['user_ids']
            item_ids = snapshot['item_ids']
            all_nodes = np.unique(np.concatenate([user_ids, item_ids]))
            node_mapping = {node: idx for idx, node in enumerate(all_nodes)}
            
                                                                
            pred_src = torch.tensor([node_mapping[u] for u in user_ids], dtype=torch.long).to(device)
            pred_dst = torch.tensor([node_mapping[i] for i in item_ids], dtype=torch.long).to(device)
            
            prepared_snapshot = {
                'node_features': torch.tensor(node_features, dtype=torch.float32).to(device),
                'edge_index': edge_index,
                'labels': torch.tensor(labels, dtype=torch.float32).to(device),
                'timestamps': torch.tensor(timestamps.mean(), dtype=torch.float32).to(device),
                'adj_matrix': adj_matrix,
                'window_id': snapshot['window_id'],
                'time_range': snapshot['time_range'],
                'pred_src': pred_src,                                 
                'pred_dst': pred_dst                                       
            }
            
            prepared_snapshots.append(prepared_snapshot)
        
        print(f"   ✅ {len(prepared_snapshots)} snapshots PyTorch prêts")
        return prepared_snapshots
