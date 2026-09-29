   
                                          
                                                              

                                                       
                                                    
   

import torch
import numpy as np


def compute_characterization_score(fid_plus: float,
                                   fid_minus: float,
                                   w_plus: float = 0.5,
                                   w_minus: float = 0.5) -> float:
       
                                                  
    
                                                           
    
                                                          
                                            
                                                                          
    
    
         
                                             
                                              
                                             
                                              
        
            
                                                                   
       
                                    
    if isinstance(fid_plus, torch.Tensor):
        fid_plus = fid_plus.item()
    if isinstance(fid_minus, torch.Tensor):
        fid_minus = fid_minus.item()
    
                              
    eps = 1e-8
    fid_plus = max(fid_plus, eps)
    fid_minus = max(fid_minus, eps)
    
                                              
    numerator = w_plus + w_minus
    denominator = (w_plus / fid_plus) + (w_minus / fid_minus)
    
    if denominator == 0:
        return 0.0
    
    char_score = numerator / denominator
    
    return float(char_score)


def compute_characterization_from_metrics(fid_plus: float,
                                          fid_minus: float,
                                          sparsity: float,
                                          alpha: float = 0.5) -> float:
       
                                                               
    
                                                             
    
         
                            
                             
                                   
                                          
        
            
                                       
       
    char = compute_characterization_score(fid_plus, fid_minus)
    sparsity_bonus = 1.0 - sparsity                                
    
    return alpha * char + (1 - alpha) * sparsity_bonus


def compute_characterization_batch(fid_plus_scores: list,
                                   fid_minus_scores: list,
                                   w_plus: float = 0.5,
                                   w_minus: float = 0.5) -> dict:
       
                                                    
    
         
                                             
                                              
                               
                                
        
            
                                              
       
    char_scores = []
    for fid_p, fid_m in zip(fid_plus_scores, fid_minus_scores):
        char = compute_characterization_score(fid_p, fid_m, w_plus, w_minus)
        char_scores.append(char)
    
    char_scores = np.array(char_scores)
    
    return {
        'mean_char': float(np.mean(char_scores)),
        'std_char': float(np.std(char_scores)),
        'min_char': float(np.min(char_scores)),
        'max_char': float(np.max(char_scores)),
        'median_char': float(np.median(char_scores))
    }


def optimize_weights(fid_plus_scores: list,
                     fid_minus_scores: list) -> tuple:
       
                                                   
    
         
                                             
                                              
        
            
                                               
       
    from scipy.optimize import minimize
    
    def objective(weights):
        w_plus, w_minus = weights
        char_scores = []
        for fid_p, fid_m in zip(fid_plus_scores, fid_minus_scores):
            char = compute_characterization_score(fid_p, fid_m, w_plus, w_minus)
            char_scores.append(char)
                                          
        return -np.mean(char_scores)
    
                                                
    constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1}
    bounds = [(0, 1), (0, 1)]
    
    result = minimize(objective, [0.5, 0.5], method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return result.x[0], result.x[1]
