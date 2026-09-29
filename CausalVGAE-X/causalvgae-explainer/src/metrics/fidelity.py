   
                                
                                                           

                                                       
                                                                    
                                                                         
   

import torch
import numpy as np


def _to_numpy(x):
                                                  
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def _to_tensor(x, device='cpu'):
                                            
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=torch.float32, device=device)


def compute_fidelity_plus(original_pred, causal_pred, labels=None) -> float:
       
                                       
    
                                                           
    
                                                                    
                                         
    
                                                                           
                                              
    
         
                                                                   
                                                                
                                        
        
            
                                                  
       
                        
    original_pred = _to_numpy(original_pred).flatten()
    causal_pred = _to_numpy(causal_pred).flatten()
    
                                     
    if original_pred.max() > 1 or original_pred.min() < 0:
        original_pred = 1 / (1 + np.exp(-original_pred))
    if causal_pred.max() > 1 or causal_pred.min() < 0:
        causal_pred = 1 / (1 + np.exp(-causal_pred))
    
                              
    min_size = min(len(original_pred), len(causal_pred))
    original_pred = original_pred[:min_size]
    causal_pred = causal_pred[:min_size]
    
                                                                      
    original_class = (original_pred > 0.5).astype(float)
    causal_class = (causal_pred > 0.5).astype(float)
    
                                              
    match = (original_class == causal_class).astype(float)
    fid_plus = np.mean(match)
    
                                                        
    delta = np.abs(original_pred - causal_pred)
    proximity_bonus = np.mean(1.0 - delta)
    
                                                            
    fid_plus_final = 0.7 * fid_plus + 0.3 * proximity_bonus
    
    return float(fid_plus_final)


def compute_fidelity_minus(original_pred, non_causal_pred, labels=None) -> float:
       
                                       
    
                                                              
    
                                                                
                                                                           
                                                                     
    
                                                                      
                                         
    
         
                                                 
                                                                   
                                        
        
            
                                                                       
       
                        
    original_pred = _to_numpy(original_pred).flatten()
    non_causal_pred = _to_numpy(non_causal_pred).flatten()
    
                                     
    if original_pred.max() > 1 or original_pred.min() < 0:
        original_pred = 1 / (1 + np.exp(-original_pred))
    if non_causal_pred.max() > 1 or non_causal_pred.min() < 0:
        non_causal_pred = 1 / (1 + np.exp(-non_causal_pred))
    
                              
    min_size = min(len(original_pred), len(non_causal_pred))
    original_pred = original_pred[:min_size]
    non_causal_pred = non_causal_pred[:min_size]
    
                                                            
    original_class = (original_pred > 0.5).astype(float)
    non_causal_class = (non_causal_pred > 0.5).astype(float)
    
                                                                
    change = (original_class != non_causal_class).astype(float)
    fid_minus_binary = np.mean(change)
    
                                               
    delta = np.abs(original_pred - non_causal_pred)
    change_magnitude = np.mean(delta)
    
                                                        
                                                                          
    fid_minus = 0.6 * fid_minus_binary + 0.4 * change_magnitude
    
    return float(fid_minus)


def compute_prediction_delta(original_prediction: float, 
                            counterfactual_prediction: float) -> float:
       
                                             
    
                                                                        
                                 
          
                                 
    
         
                                                 
                                                             
        
            
                           
       
    if counterfactual_prediction * original_prediction < 0:
                                                    
        return abs(counterfactual_prediction) + abs(original_prediction)
                                                               
    return abs(original_prediction) - abs(counterfactual_prediction)


def compute_fidelity_metrics(original_pred, causal_pred, non_causal_pred, labels=None) -> dict:
       
                                                       
    
         
                                             
                                                   
                                                                            
                                        
        
            
                                      
       
    fid_plus = compute_fidelity_plus(original_pred, causal_pred, labels)
    fid_minus = compute_fidelity_minus(original_pred, non_causal_pred, labels)
    
    return {
        'fidelity_plus': fid_plus,
        'fidelity_minus': fid_minus
    }
