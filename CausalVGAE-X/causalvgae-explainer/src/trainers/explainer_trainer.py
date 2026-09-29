   
                                           
                                                    
   

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('..')

from ..metrics.fidelity import compute_fidelity_plus, compute_fidelity_minus
from ..metrics.sparsity import compute_sparsity
from ..metrics.characterization import compute_characterization_score
from ..metrics.aufsc import compute_aufsc_plus, compute_aufsc_minus
from ..metrics.temporal_continuity import compute_temporal_continuity


class DynamicExplainerTrainer:
       
                                                                       
    
                               
                                                                
                                                         
       
    
    def __init__(self, model: nn.Module, device: torch.device):
           
             
                                             
                                        
           
        self.model = model.to(device)
        self.device = device
        
                    
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
                   
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
                                  
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'fidelity_plus': [],
            'fidelity_minus': [],
            'sparsity': [],
            'char_scores': [],
            'aufsc_plus': [],
            'aufsc_minus': [],
            'temporal_continuity': [],
            'causal_info_flow': [],
            'learning_rate': []
        }
        
                                       
        self.loss_weights = {
            'recon': 1.0,
            'kl': 0.1,
            'causal': 0.3,
            'fidelity': 0.4,
            'sparsity': 0.2,
            'temporal': 0.1,
            'char': 0.3
        }
    
    def compute_combined_loss(self, output: dict, snapshot: dict, epoch: int) -> dict:
           
                                                             
        
             
                                    
                                     
                                                    
            
                
                                               
           
        labels = snapshot['labels']
        node_features = snapshot['node_features']
        
                                                                     
                                                                                       
        recon_loss = nn.MSELoss()(output['recon'], node_features)
        
                                        
        mu, logvar = output['mu'], output['logvar']
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.size(0)                 
        
                                                              
        causal_loss = -output['causal_info_flow']
        
                                      
                                                                                   
        node_embeddings = output['z']                           
        
                                                                
        if self.model.target_predictor is not None:
                                            
            src_nodes = snapshot['pred_src']
            dst_nodes = snapshot['pred_dst']
            edge_features = torch.cat([
                node_embeddings[src_nodes],
                node_embeddings[dst_nodes]
            ], dim=-1)
                                                            
            original_pred = torch.sigmoid(edge_features.mean(dim=-1, keepdim=True))
            masked_pred = original_pred * output['causal_mask'][src_nodes].mean(dim=-1, keepdim=True)
        else:
                                                                               
                                                              
            original_pred = output['predictions']
            if original_pred.size(0) != len(labels):
                                                  
                src_nodes = snapshot['pred_src']
                dst_nodes = snapshot['pred_dst']
                original_pred = output['predictions'][src_nodes]
            masked_pred = original_pred * 0.5                           
        
        fid_plus = compute_fidelity_plus(original_pred, masked_pred, labels)
        fid_minus = compute_fidelity_minus(original_pred, masked_pred, labels)
        
                                                   
        fidelity_loss = -(fid_plus - 0.1 * fid_minus)
        
                                      
        total_edges = snapshot['edge_index'].size(1)
        current_sparsity = compute_sparsity(output['causal_mask'], total_edges)
        target_sparsity = 0.45
        sparsity_loss = torch.abs(current_sparsity - target_sparsity)
        
                                     
        temporal_loss = output['temporal_loss']
        
                                             
        char_score = compute_characterization_score(fid_plus, fid_minus)
        char_loss = -char_score                  
        
                              
        total_loss = (
            self.loss_weights['recon'] * recon_loss +
            self.loss_weights['kl'] * kl_loss +
            self.loss_weights['causal'] * causal_loss +
            self.loss_weights['fidelity'] * fidelity_loss +
            self.loss_weights['sparsity'] * sparsity_loss +
            self.loss_weights['temporal'] * temporal_loss +
            self.loss_weights['char'] * char_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'causal_loss': causal_loss.item(),
            'fidelity_loss': fidelity_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'temporal_loss': temporal_loss.item(),
            'char_loss': char_loss if isinstance(char_loss, float) else char_loss.item(),
            'fid_plus': fid_plus.item() if hasattr(fid_plus, 'item') else fid_plus,
            'fid_minus': fid_minus.item() if hasattr(fid_minus, 'item') else fid_minus,
            'sparsity': current_sparsity.item() if hasattr(current_sparsity, 'item') else current_sparsity,
            'char_score': char_score
        }
    
    def train_epoch(self, snapshots: list, epoch: int) -> dict:
           
                                         
        
             
                                         
                                  
            
                
                                     
           
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0,
            'fid_plus': [],
            'fid_minus': [],
            'sparsity': [],
            'char_scores': [],
            'causal_info_flow': []
        }
        
        prev_z_causal = None
        epoch_masks = []
        
        for snapshot in tqdm(snapshots, desc=f"Époque {epoch+1}"):
            self.optimizer.zero_grad()
            
                          
            output = self.model(snapshot, prev_z_causal)
            
                               
            losses = self.compute_combined_loss(output, snapshot, epoch)
            
                      
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
                                 
            epoch_metrics['total_loss'] += losses['total_loss'].item()
            epoch_metrics['fid_plus'].append(losses['fid_plus'])
            epoch_metrics['fid_minus'].append(losses['fid_minus'])
            epoch_metrics['sparsity'].append(losses['sparsity'])
            epoch_metrics['char_scores'].append(losses['char_score'])
            epoch_metrics['causal_info_flow'].append(output['causal_info_flow'].item())
            
            epoch_masks.append(output['causal_mask'].detach())
            
                                                           
            prev_z_causal = output['z_causal'].detach()
        
                  
        n = len(snapshots)
        avg_loss = epoch_metrics['total_loss'] / n
        avg_fid_plus = np.mean(epoch_metrics['fid_plus'])
        avg_fid_minus = np.mean(epoch_metrics['fid_minus'])
        avg_sparsity = np.mean(epoch_metrics['sparsity'])
        avg_char = np.mean(epoch_metrics['char_scores'])
        avg_causal_flow = np.mean(epoch_metrics['causal_info_flow'])
        
                      
        fid_plus_array = np.array(epoch_metrics['fid_plus'])
        fid_minus_array = np.array(epoch_metrics['fid_minus'])
        sparsity_array = np.array(epoch_metrics['sparsity'])
        
        aufsc_plus = compute_aufsc_plus(fid_plus_array, sparsity_array)
        aufsc_minus = compute_aufsc_minus(fid_minus_array, sparsity_array)
        
                               
        temporal_continuity = compute_temporal_continuity(epoch_masks)
        
        return {
            'avg_loss': avg_loss,
            'avg_fid_plus': avg_fid_plus,
            'avg_fid_minus': avg_fid_minus,
            'avg_sparsity': avg_sparsity,
            'avg_char': avg_char,
            'avg_causal_flow': avg_causal_flow,
            'aufsc_plus': aufsc_plus,
            'aufsc_minus': aufsc_minus,
            'temporal_continuity': temporal_continuity.item()
        }
    
    def validate(self, val_snapshots: list) -> dict:
           
                            
        
             
                                                  
            
                
                                   
           
        self.model.eval()
        
        val_metrics = {
            'total_loss': 0,
            'fid_plus': [],
            'fid_minus': [],
            'sparsity': [],
            'char_scores': []
        }
        
        prev_z_causal = None
        val_masks = []
        
        with torch.no_grad():
            for snapshot in val_snapshots:
                output = self.model(snapshot, prev_z_causal)
                losses = self.compute_combined_loss(output, snapshot, epoch=0)
                
                val_metrics['total_loss'] += losses['total_loss'].item()
                val_metrics['fid_plus'].append(losses['fid_plus'])
                val_metrics['fid_minus'].append(losses['fid_minus'])
                val_metrics['sparsity'].append(losses['sparsity'])
                val_metrics['char_scores'].append(losses['char_score'])
                
                val_masks.append(output['causal_mask'])
                prev_z_causal = output['z_causal']
        
                  
        n = len(val_snapshots)
        avg_loss = val_metrics['total_loss'] / n
        avg_fid_plus = np.mean(val_metrics['fid_plus'])
        avg_fid_minus = np.mean(val_metrics['fid_minus'])
        avg_sparsity = np.mean(val_metrics['sparsity'])
        avg_char = np.mean(val_metrics['char_scores'])
        
        temporal_continuity = compute_temporal_continuity(val_masks)
        
        return {
            'avg_loss': avg_loss,
            'avg_fid_plus': avg_fid_plus,
            'avg_fid_minus': avg_fid_minus,
            'avg_sparsity': avg_sparsity,
            'avg_char': avg_char,
            'temporal_continuity': temporal_continuity.item()
        }
    
    def train(self, train_snapshots: list, val_snapshots: list, 
             epochs: int = 50, save_path: str = 'causalvgae_explainer.pth'):
           
                                      
        
             
                                                     
                                                  
                                    
                                           
           
        print(f"\n{'='*70}")
        print(f"ENTRAÎNEMENT CAUSALVGAE-EXPLAINER EXPLAINER")
        print(f"{'='*70}\n")
        print(f"Train snapshots: {len(train_snapshots)}")
        print(f"Val snapshots: {len(val_snapshots)}")
        print(f"Epochs: {epochs}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
                          
            train_metrics = self.train_epoch(train_snapshots, epoch)
            
                        
            val_metrics = self.validate(val_snapshots)
            
                                    
            self.history['train_loss'].append(train_metrics['avg_loss'])
            self.history['val_loss'].append(val_metrics['avg_loss'])
            self.history['fidelity_plus'].append(train_metrics['avg_fid_plus'])
            self.history['fidelity_minus'].append(train_metrics['avg_fid_minus'])
            self.history['sparsity'].append(train_metrics['avg_sparsity'])
            self.history['char_scores'].append(train_metrics['avg_char'])
            self.history['aufsc_plus'].append(train_metrics['aufsc_plus'])
            self.history['aufsc_minus'].append(train_metrics['aufsc_minus'])
            self.history['temporal_continuity'].append(train_metrics['temporal_continuity'])
            self.history['causal_info_flow'].append(train_metrics['avg_causal_flow'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
                       
            self.scheduler.step(val_metrics['avg_loss'])
            
                                           
            if val_metrics['avg_loss'] < best_val_loss:
                best_val_loss = val_metrics['avg_loss']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'history': self.history
                }, save_path)
                print(f"✅ Modèle sauvegardé à l'époque {epoch+1}")
            
                       
            print(f"\nÉpoque {epoch+1}/{epochs}")
            print(f"  Train: Loss={train_metrics['avg_loss']:.4f}")
            print(f"  Val:   Loss={val_metrics['avg_loss']:.4f}")
            print(f"  Fidelity+: {train_metrics['avg_fid_plus']:.4f}")
            print(f"  Fidelity-: {train_metrics['avg_fid_minus']:.4f}")
            print(f"  Sparsity: {train_metrics['avg_sparsity']:.4f}")
            print(f"  Char Score: {train_metrics['avg_char']:.4f}")
            print(f"  AUFSC+: {train_metrics['aufsc_plus']:.4f}")
            print(f"  AUFSC-: {train_metrics['aufsc_minus']:.4f}")
            print(f"  Temporal Continuity: {train_metrics['temporal_continuity']:.4f}")
            print(f"  Causal Info Flow: {train_metrics['avg_causal_flow']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 70)
        
        print(f"\n✅ Entraînement terminé! Meilleur val_loss: {best_val_loss:.4f}\n")
        
        return self.history
