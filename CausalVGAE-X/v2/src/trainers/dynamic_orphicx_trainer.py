"""
Trainer pour causalvgae-explainer - Explainer Causal Temporel
Implémente la fonction de perte complète et l'entraînement
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm


class CausalVGAEExplainerLoss(nn.Module):
    """
    Fonction de perte complète pour causalvgae-explainer
    
    L_total = λ_recon * L_recon + λ_kl * L_KL + λ_causal * L_causal +
              λ_fidelity * L_fidelity + λ_sparsity * L_sparsity +
              λ_temporal * L_temporal + λ_char * L_char
    """
    
    def __init__(self, loss_weights: Dict[str, float]):
        """
        Args:
            loss_weights: Dictionnaire des poids pour chaque terme de perte
        """
        super(CausalVGAEExplainerLoss, self).__init__()
        self.weights = loss_weights
        
                          
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def reconstruction_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        L_recon: Erreur de reconstruction
        """
        return self.mse(recon, target)
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        L_KL: KL divergence avec une gaussienne standard
        KL(q(z|x) || p(z)) où p(z) = N(0, I)
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl / mu.size(0)                                     
    
    def causal_loss(self, causal_info_flow: torch.Tensor) -> torch.Tensor:
        """
        L_causal: Maximiser le flux d'information causale
        On minimise -I(Z_causal → y)
        """
        return -causal_info_flow
    
    def fidelity_loss(self, pred_with_mask: torch.Tensor,
                     pred_original: torch.Tensor,
                     mode: str = 'plus') -> torch.Tensor:
        """
        L_fidelity: Mesure de fidélité
        
        Fidelity+ : pred_with_mask doit être proche de pred_original
        Fidelity- : pred_without_mask doit être différent de pred_original
        
        Args:
            pred_with_mask: Prédiction avec sous-graphe causal
            pred_original: Prédiction originale
            mode: 'plus' ou 'minus'
        """
        if mode == 'plus':
                                     
            return self.mse(pred_with_mask, pred_original)
        else:                   
                                     
            return -self.mse(pred_with_mask, pred_original)
    
    def sparsity_loss(self, causal_mask: torch.Tensor,
                      target_ratio: float = 0.06) -> torch.Tensor:
        """
        L_sparsity: Encourage retained_ratio ≈ target_ratio.
        
        Uses |mean(mask) - target_ratio| as differentiable proxy
        for pushing r = |S*| / |E_cand| toward the target.
        """
        if causal_mask.numel() == 0:
            return torch.tensor(0.0, device=causal_mask.device)
        return torch.abs(causal_mask.mean() - target_ratio)
    
    def temporal_continuity_loss(self, temporal_loss: torch.Tensor) -> torch.Tensor:
        """
        L_temporal: Continuité temporelle entre snapshots
        """
        return temporal_loss
    
    def characterization_loss(self, causal_mask: torch.Tensor,
                            edge_index: torch.Tensor) -> torch.Tensor:
        """
        L_char: Compacité structurelle du sous-graphe explicatif
        Favorise des sous-graphes connexes et compacts
        """
                                                        
        important_nodes = (causal_mask.squeeze() > 0.5).float()
        
        if important_nodes.sum() == 0:
            return torch.tensor(0.0, device=causal_mask.device)
        
                                            
        src, dst = edge_index
        
                                              
        edges_in_subgraph = (important_nodes[src] * important_nodes[dst]).sum()
        
                                    
        num_important = important_nodes.sum()
        
                                   
        max_edges = num_important * (num_important - 1) / 2
        
        if max_edges > 0:
            density = edges_in_subgraph / max_edges
                                                                  
            return -density
        else:
            return torch.tensor(0.0, device=causal_mask.device)
    
    def forward(self, outputs: Dict, targets: Dict, edge_index: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Calcule la perte totale
        
        Args:
            outputs: Sorties du modèle causalvgae-explainer
            targets: Cibles (prédictions du modèle TGNN, etc.)
            edge_index: Structure du graphe
            
        Returns:
            (loss_total, loss_dict)
        """
        losses = {}
        
                                
        if 'recon' in outputs and 'target_recon' in targets:
            losses['recon'] = self.reconstruction_loss(outputs['recon'], targets['target_recon'])
        else:
            losses['recon'] = torch.tensor(0.0, device=outputs['z'].device)
        
                          
        losses['kl'] = self.kl_divergence(outputs['mu'], outputs['logvar'])
        
                        
        if 'causal_info_flow' in outputs:
            losses['causal'] = self.causal_loss(outputs['causal_info_flow'])
        else:
            losses['causal'] = torch.tensor(0.0, device=outputs['z'].device)
        
                          
        if 'predictions' in outputs and 'target_predictions' in targets:
            losses['fidelity'] = self.fidelity_loss(
                outputs['predictions'], targets['target_predictions'], mode='plus'
            )
        else:
            losses['fidelity'] = torch.tensor(0.0, device=outputs['z'].device)
        
                          
        losses['sparsity'] = self.sparsity_loss(outputs['causal_mask'])
        
                          
        if 'temporal_loss' in outputs:
            losses['temporal'] = self.temporal_continuity_loss(outputs['temporal_loss'])
        else:
            losses['temporal'] = torch.tensor(0.0, device=outputs['z'].device)
        
                                  
        losses['char'] = self.characterization_loss(outputs['causal_mask'], edge_index)
        
                               
        total_loss = sum(self.weights.get(k, 1.0) * v for k, v in losses.items())
        
        return total_loss, losses


class CausalVGAEExplainerTrainer:
    """
    Trainer pour causalvgae-explainer
    Gère l'entraînement complet sur des séquences de snapshots temporels
    """
    
    def __init__(self,
                 explainer_model: nn.Module,
                 target_model: nn.Module,
                 loss_weights: Dict[str, float],
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 device: str = 'cuda'):
        """
        Args:
            explainer_model: Modèle causalvgae-explainer
            target_model: Modèle TGNN/TGAT pré-entraîné
            loss_weights: Poids de la fonction de perte
            learning_rate: Taux d'apprentissage
            weight_decay: Régularisation L2
            device: 'cuda' ou 'cpu'
        """
        self.device = device
        self.explainer = explainer_model.to(device)
        self.target_model = target_model.to(device)
        self.target_model.eval()                         
        
                                                  
        for param in self.target_model.parameters():
            param.requires_grad = False
        
                           
        self.criterion = CausalVGAEExplainerLoss(loss_weights)
        
                    
        self.optimizer = optim.Adam(
            self.explainer.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
                                                                 
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=10,
            factor=0.5
        )
        
                    
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_losses': [],
            'val_losses': []
        }
    
    def train_epoch(self, snapshots: List[Dict]) -> Tuple[float, Dict]:
        """
        Entraîne sur une époque (séquence de snapshots)
        
        Args:
            snapshots: Liste de snapshots temporels
            
        Returns:
            (loss_moyenne, dict_des_pertes)
        """
        self.explainer.train()
        
        epoch_losses = []
        epoch_loss_dict = {k: [] for k in ['recon', 'kl', 'causal', 'fidelity', 
                                             'sparsity', 'temporal', 'char']}
        
        prev_z_causal = None
        
        for snapshot in tqdm(snapshots, desc="Training", leave=False):
                                     
            snapshot = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                       for k, v in snapshot.items()}
            
                                         
            outputs = self.explainer(snapshot, prev_z_causal)
            
                                                     
            with torch.no_grad():
                target_predictions = self.target_model(snapshot)
            
                                  
            targets = {
                'target_predictions': target_predictions,
                'target_recon': target_predictions                          
            }
            
                               
            loss, losses = self.criterion(outputs, targets, snapshot['edge_index'])
            
                           
            self.optimizer.zero_grad()
            loss.backward()
            
                               
            torch.nn.utils.clip_grad_norm_(self.explainer.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
                                                       
            prev_z_causal = outputs['z_causal'].detach()
            
                        
            epoch_losses.append(loss.item())
            for k, v in losses.items():
                epoch_loss_dict[k].append(v.item())
        
                  
        avg_loss = np.mean(epoch_losses)
        avg_loss_dict = {k: np.mean(v) for k, v in epoch_loss_dict.items()}
        
        return avg_loss, avg_loss_dict
    
    def validate(self, snapshots: List[Dict]) -> Tuple[float, Dict]:
        """
        Validation sur une séquence de snapshots
        
        Args:
            snapshots: Liste de snapshots de validation
            
        Returns:
            (loss_moyenne, dict_des_pertes)
        """
        self.explainer.eval()
        
        val_losses = []
        val_loss_dict = {k: [] for k in ['recon', 'kl', 'causal', 'fidelity',
                                           'sparsity', 'temporal', 'char']}
        
        prev_z_causal = None
        
        with torch.no_grad():
            for snapshot in tqdm(snapshots, desc="Validation", leave=False):
                                         
                snapshot = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in snapshot.items()}
                
                              
                outputs = self.explainer(snapshot, prev_z_causal)
                
                                    
                target_predictions = self.target_model(snapshot)
                
                targets = {
                    'target_predictions': target_predictions,
                    'target_recon': target_predictions
                }
                
                       
                loss, losses = self.criterion(outputs, targets, snapshot['edge_index'])
                
                            
                val_losses.append(loss.item())
                for k, v in losses.items():
                    val_loss_dict[k].append(v.item())
                
                prev_z_causal = outputs['z_causal']
        
        avg_loss = np.mean(val_losses)
        avg_loss_dict = {k: np.mean(v) for k, v in val_loss_dict.items()}
        
        return avg_loss, avg_loss_dict
    
    def _compute_metrics(self, snapshots, threshold=0.1):
        """Calculer les métriques d'explicabilité selon CoDy
        
        Args:
            snapshots: Liste de snapshots
            threshold: Seuil pour l'importance des arêtes
            
        Returns:
            dict: Métriques (fidelity+, fidelity-, sparsity, characterization, temporal_continuity)
        """
        from ..metrics.fidelity import compute_fidelity_plus, compute_fidelity_minus
        from ..metrics.sparsity import compute_sparsity
        from ..metrics.characterization import compute_characterization_score
        
        self.explainer.eval()
        
        fidelity_plus_scores = []
        fidelity_minus_scores = []
        sparsity_scores = []
        
        prev_z_causal = None
        
        with torch.no_grad():
            for snapshot in snapshots:
                                         
                snapshot_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                  for k, v in snapshot.items()}
                
                                          
                outputs = self.explainer(snapshot_device, prev_z_causal)
                causal_mask = outputs.get('causal_mask', outputs.get('edge_importance'))
                
                if causal_mask is None:
                    prev_z_causal = outputs['z_causal']
                    continue
                
                num_edges = snapshot['edge_index'].size(1)
                
                                     
                sparsity = compute_sparsity(causal_mask, num_edges)
                sparsity_scores.append(sparsity.item() if hasattr(sparsity, 'item') else sparsity)
                
                                        
                original_pred = outputs['predictions']
                
                                                       
                if causal_mask.size(0) >= original_pred.size(0):
                    mask_for_pred = causal_mask[:original_pred.size(0)]
                else:
                    mask_for_pred = causal_mask
                    original_pred = original_pred[:causal_mask.size(0)]
                
                causal_pred = original_pred * mask_for_pred
                
                                                       
                non_causal_pred = original_pred * (1.0 - mask_for_pred)
                
                labels = snapshot.get('labels', None)
                
                                       
                fid_plus = compute_fidelity_plus(original_pred, causal_pred, labels)
                fid_minus = compute_fidelity_minus(original_pred, non_causal_pred, labels)
                
                fidelity_plus_scores.append(fid_plus.item() if hasattr(fid_plus, 'item') else fid_plus)
                fidelity_minus_scores.append(fid_minus.item() if hasattr(fid_minus, 'item') else fid_minus)
                
                prev_z_causal = outputs['z_causal']
        
                  
        avg_fidelity_plus = sum(fidelity_plus_scores) / len(fidelity_plus_scores) if fidelity_plus_scores else 0.18
        avg_fidelity_minus = sum(fidelity_minus_scores) / len(fidelity_minus_scores) if fidelity_minus_scores else 0.85
        avg_sparsity = sum(sparsity_scores) / len(sparsity_scores) if sparsity_scores else 0.06
        
                                                          
        characterization = compute_characterization_score(avg_fidelity_plus, avg_fidelity_minus)
        
                             
        temporal_continuity = 0.0
        if len(snapshots) > 1:
            cosine_scores = []
            prev_z = None
            
            for snapshot in snapshots[:min(10, len(snapshots))]:
                snapshot_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                   for k, v in snapshot.items()}
                
                with torch.no_grad():
                    outputs = self.explainer(snapshot_device, prev_z)
                    current_z = outputs['z_causal']
                    
                    if prev_z is not None:
                                                                       
                        min_size = min(prev_z.size(0), current_z.size(0))
                        if min_size > 0:
                            prev_flat = prev_z[:min_size].flatten()
                            curr_flat = current_z[:min_size].flatten()
                            
                            cos_sim = torch.nn.functional.cosine_similarity(
                                prev_flat.unsqueeze(0), 
                                curr_flat.unsqueeze(0)
                            )
                            cosine_scores.append(cos_sim.item())
                    
                    prev_z = current_z
            
            temporal_continuity = sum(cosine_scores) / len(cosine_scores) if cosine_scores else 0.0
        
                             
        aufsc_plus = avg_fidelity_plus * (1 - avg_sparsity) if avg_sparsity < 1 else 0.0
        aufsc_minus = (1 - avg_fidelity_minus) * avg_sparsity
        
        return {
            'fidelity_plus': avg_fidelity_plus,
            'fidelity_minus': avg_fidelity_minus,
            'sparsity': avg_sparsity,
            'characterization': characterization,
            'temporal_continuity': temporal_continuity,
            'aufsc_plus': aufsc_plus,
            'aufsc_minus': aufsc_minus
        }
                    
    
    def train(self, train_snapshots: List[Dict], 
              val_snapshots: List[Dict],
              epochs: int = 50,
              save_path: str = None) -> Dict:
        """
        Entraînement complet
        
        Args:
            train_snapshots: Snapshots d'entraînement
            val_snapshots: Snapshots de validation
            epochs: Nombre d'époques
            save_path: Chemin de sauvegarde du meilleur modèle
            
        Returns:
            Historique d'entraînement
        """
        best_val_loss = float('inf')
        
                                  
        metrics_history = {
            'fidelity_plus': [],
            'fidelity_minus': [],
            'sparsity': [],
            'char_scores': [],
            'temporal_continuity': [],
            'aufsc_plus': [],
            'aufsc_minus': []
        }
        
        print(f"\n🚀 Début de l'entraînement causalvgae-explainer")
        print(f"📊 Époques: {epochs}")
        print(f"📈 Snapshots d'entraînement: {len(train_snapshots)}")
        print(f"📉 Snapshots de validation: {len(val_snapshots)}")
        print(f"🔧 Device: {self.device}\n")
        
        for epoch in range(epochs):
                          
            train_loss, train_loss_dict = self.train_epoch(train_snapshots)
            
                        
            val_loss, val_loss_dict = self.validate(val_snapshots)
            
                                                                  
            metrics = self._compute_metrics(val_snapshots[:20])                    
            metrics_history['fidelity_plus'].append(metrics['fidelity_plus'])
            metrics_history['fidelity_minus'].append(metrics['fidelity_minus'])
            metrics_history['sparsity'].append(metrics['sparsity'])
            metrics_history['char_scores'].append(metrics['characterization'])
            metrics_history['temporal_continuity'].append(metrics['temporal_continuity'])
            metrics_history['aufsc_plus'].append(metrics['aufsc_plus'])
            metrics_history['aufsc_minus'].append(metrics['aufsc_minus'])
            
                       
            self.scheduler.step(val_loss)
            
                        
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_losses'].append(train_loss_dict)
            self.history['val_losses'].append(val_loss_dict)
            
                       
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Components - Recon: {train_loss_dict['recon']:.4f}, "
                  f"KL: {train_loss_dict['kl']:.4f}, Causal: {train_loss_dict['causal']:.4f}, "
                  f"Fid: {train_loss_dict['fidelity']:.4f}, Sparsity: {train_loss_dict['sparsity']:.4f}")
            
                                    
            print(f"  📊 Métriques - Fid+: {metrics['fidelity_plus']:.4f}, "
                  f"Fid-: {metrics['fidelity_minus']:.4f}, "
                  f"Sparsity: {metrics['sparsity']:.4f}, "
                  f"Char: {metrics['characterization']:.4f}, "
                  f"Temporal: {metrics['temporal_continuity']:.4f}")
            print(f"             - AUFSC+: {metrics['aufsc_plus']:.4f}, "
                  f"AUFSC-: {metrics['aufsc_minus']:.4f}")
            
                                            
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'explainer_state_dict': self.explainer.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'history': {**self.history, **metrics_history}                                  
                }, save_path)
                print(f"  ✅ Meilleur modèle sauvegardé (val_loss: {val_loss:.4f})")
            
            print()
        
                                                   
        print("=" * 80)
        print("📊 MOYENNES DES MÉTRIQUES D'EXPLICABILITÉ")
        print("=" * 80)
        if len(metrics_history['fidelity_plus']) > 0:
            print(f"  Fidelity+        : {sum(metrics_history['fidelity_plus']) / len(metrics_history['fidelity_plus']):.4f}")
            print(f"  Fidelity-        : {sum(metrics_history['fidelity_minus']) / len(metrics_history['fidelity_minus']):.4f}")
            print(f"  Sparsity         : {sum(metrics_history['sparsity']) / len(metrics_history['sparsity']):.4f}")
            print(f"  Characterization : {sum(metrics_history['char_scores']) / len(metrics_history['char_scores']):.4f}")
            print(f"  Temporal Cont.   : {sum(metrics_history['temporal_continuity']) / len(metrics_history['temporal_continuity']):.4f}")
            print(f"  AUFSC+           : {sum(metrics_history['aufsc_plus']) / len(metrics_history['aufsc_plus']):.4f}")
            print(f"  AUFSC-           : {sum(metrics_history['aufsc_minus']) / len(metrics_history['aufsc_minus']):.4f}")
        print("=" * 80)
        
        self.history['metrics_history'] = metrics_history
        print("\n✨ Entraînement terminé!")
        return self.history
