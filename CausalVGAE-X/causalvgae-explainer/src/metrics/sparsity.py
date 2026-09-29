                                                                 

            
                                      
                            

                                                  
                                                  
   

import torch
import numpy as np


def _to_numpy(x):
                                                  
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_retained_ratio(explanation_edges, candidate_edges) -> float:
                                                                        
    
         
                                                                              
                                                                     
        
            
                               
       
    n_cand = len(candidate_edges)
    if n_cand == 0:
        return 0.0
    return float(len(explanation_edges)) / float(n_cand)


def compute_compactness(explanation_edges, candidate_edges) -> float:
                                                               
    
         
                                                                              
                                                                     
        
            
                                  
       
    return 1.0 - compute_retained_ratio(explanation_edges, candidate_edges)


def compute_sparsity(causal_mask, candidate_events: int, threshold: float = None) -> float:
                                                                     
    
                                            
    
                                               
                                            
    
         
                                                                 
                                                                   
                                                                        
        
            
                               
       
    if candidate_events == 0:
        return 0.0
    
    causal_mask = _to_numpy(causal_mask).flatten()
    
                                                       
    if threshold is None:
        k = max(1, int(0.06 * len(causal_mask)))
        if len(causal_mask) > 0:
            sorted_mask = np.sort(causal_mask)[::-1]
            threshold = sorted_mask[min(k, len(sorted_mask) - 1)]
        else:
            threshold = 0.5
    
                                        
    num_selected = int(np.sum(causal_mask >= threshold))
    
                                      
    retained_ratio = float(num_selected) / float(candidate_events)
    
    return float(retained_ratio)


def compute_sparsity_from_explanation(explanation_events: np.ndarray,
                                      candidate_events: np.ndarray) -> float:
                                                       
    
                       
    
         
                                                                 
                                                              
        
            
                               
       
    if len(candidate_events) == 0:
        return 0.0
    
    return float(len(explanation_events)) / float(len(candidate_events))


def compute_ratio_loss(edge_scores, target_ratio: float = 0.06) -> torch.Tensor:
                                              
    
                                                  
    
                                                                  
    
         
                                      
                                                                   
        
            
                          
       
    if edge_scores.numel() == 0:
        return torch.tensor(0.0, device=edge_scores.device)
    return torch.abs(edge_scores.mean() - target_ratio)


def compute_retained_ratio_from_topk(edge_scores, k: int) -> float:
                                                      
    
                    
    
         
                                                           
                                    
        
            
                               
       
    scores = _to_numpy(edge_scores).flatten()
    n = len(scores)
    if n == 0:
        return 0.0
    k = max(1, min(int(k), n))
    return float(k) / float(n)


def compute_sparsity_adaptive(causal_mask, candidate_events: int,
                             target_sparsity: float = 0.06) -> dict:
                                                   
    
         
                                
                                                             
                                                                      
        
            
                                                                  
       
    retained_ratio = compute_sparsity(causal_mask, candidate_events)
    compactness = 1.0 - retained_ratio
    
                                             
    sparsity_loss = abs(retained_ratio - target_sparsity)
    
    return {
        'retained_ratio': retained_ratio,
        'sparsity': retained_ratio,                                 
        'compactness': compactness,
        'sparsity_loss': sparsity_loss,
        'deviation_from_target': retained_ratio - target_sparsity
    }


def compute_sparsity_range(causal_masks: list, candidate_events_list: list) -> dict:
                                                                 
    
         
                                          
                                                             
        
            
                       
       
    ratios = []
    
    for mask, candidates in zip(causal_masks, candidate_events_list):
        r = compute_sparsity(mask, candidates)
        ratios.append(r)
    
    ratios = np.array(ratios)
    compactness = 1.0 - ratios
    
    return {
        'mean_retained_ratio': float(np.mean(ratios)),
        'mean_compactness': float(np.mean(compactness)),
        'mean_sparsity': float(np.mean(ratios)),                              
        'std_sparsity': float(np.std(ratios)),
        'min_sparsity': float(np.min(ratios)),
        'max_sparsity': float(np.max(ratios)),
        'median_sparsity': float(np.median(ratios)),
        'in_target_range': float(np.sum((ratios >= 0.04) & (ratios <= 0.07)) / len(ratios))
    }
