   
                                            
                                                     
   

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


class TargetModelTrainer:
       
                                                            
                                                                 
       
    
    def __init__(self, model: nn.Module, device: torch.device):
           
             
                                      
                                        
           
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        self.criterion = nn.BCELoss()
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_auc': [],
            'val_auc': []
        }
    
    def train_epoch(self, snapshots: list) -> tuple:
           
                                         
        
             
                                                        
            
                
                                             
           
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for snapshot in tqdm(snapshots, desc="Training"):
            self.optimizer.zero_grad()
            
            x = snapshot['node_features']
            edge_index = snapshot['edge_index']
            timestamps = snapshot['timestamps']
            labels = snapshot['labels']
            
                                                             
            node_embeddings = self.model.get_embeddings(x, edge_index, timestamps)
            
                                                                
            src_nodes = snapshot['pred_src']
            dst_nodes = snapshot['pred_dst']
            
                                                           
            edge_features = torch.cat([
                node_embeddings[src_nodes],
                node_embeddings[dst_nodes]
            ], dim=-1)
            
                                                             
            predictions = self.model.predictor(edge_features)
            
                                
            loss = self.criterion(predictions, labels.unsqueeze(-1))
            
                      
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
                                    
            all_predictions.extend(predictions.detach().cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
        
                              
        avg_loss = total_loss / len(snapshots)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        binary_preds = (all_predictions > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, binary_preds)
        
        try:
            auc = roc_auc_score(all_labels, all_predictions)
        except:
            auc = 0.5
        
        return avg_loss, accuracy, auc
    
    def validate(self, val_snapshots: list) -> tuple:
           
                        
        
             
                                                  
            
                
                                       
           
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for snapshot in val_snapshots:
                x = snapshot['node_features']
                edge_index = snapshot['edge_index']
                timestamps = snapshot['timestamps']
                labels = snapshot['labels']
                
                                                              
                node_embeddings = self.model.get_embeddings(x, edge_index, timestamps)
                src_nodes = snapshot['pred_src']
                dst_nodes = snapshot['pred_dst']
                
                edge_features = torch.cat([
                    node_embeddings[src_nodes],
                    node_embeddings[dst_nodes]
                ], dim=-1)
                
                predictions = self.model.predictor(edge_features)
                loss = self.criterion(predictions, labels.unsqueeze(-1))
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(val_snapshots)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        binary_preds = (all_predictions > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, binary_preds)
        
        try:
            auc = roc_auc_score(all_labels, all_predictions)
        except:
            auc = 0.5
        
        return avg_loss, accuracy, auc
    
    def train(self, train_snapshots: list, val_snapshots: list, 
             epochs: int = 50, save_path: str = 'target_model.pth'):
           
                                      
        
             
                                                     
                                                  
                                    
                                                     
           
        print(f"\n{'='*60}")
        print(f"ENTRAÎNEMENT DU MODÈLE CIBLE: {type(self.model).__name__}")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
                          
            train_loss, train_acc, train_auc = self.train_epoch(train_snapshots)
            
                        
            val_loss, val_acc, val_auc = self.validate(val_snapshots)
            
                        
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
                       
            self.scheduler.step(val_loss)
            
                        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, save_path)
                print(f"✅ Modèle sauvegardé à l'époque {epoch+1}")
            
                       
            print(f"Époque {epoch+1}/{epochs}")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, AUC={train_auc:.4f}")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 60)
        
        print(f"\n✅ Entraînement terminé! Meilleur val_loss: {best_val_loss:.4f}\n")
    
    def save_predictions(self, snapshots: list, save_path: str = 'predictions.npy'):
           
                                            
        
             
                                          
                                           
           
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for snapshot in snapshots:
                x = snapshot['node_features']
                edge_index = snapshot['edge_index']
                timestamps = snapshot['timestamps']
                
                predictions = self.model(x, edge_index, timestamps)
                all_predictions.append(predictions.cpu().numpy())
        
        np.save(save_path, np.concatenate(all_predictions, axis=0))
        print(f"✅ Prédictions sauvegardées: {save_path}")
