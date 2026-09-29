   
                     
                                                       
   

import torch
import numpy as np


def compute_temporal_continuity(masks: list) -> torch.Tensor:
       
                                                        
    
                                                          
                                                                                       
    
         
                                                                     
        
            
                                               
       
    if len(masks) < 2:
        return torch.tensor(0.0)
    
    continuity_scores = []
    
    for i in range(len(masks) - 1):
        mask1 = masks[i].flatten()
        mask2 = masks[i+1].flatten()
        
                                     
        min_size = min(mask1.size(0), mask2.size(0))
        
        if min_size > 0:
            mask1_sync = mask1[:min_size]
            mask2_sync = mask2[:min_size]
            
            similarity = torch.cosine_similarity(
                mask1_sync.unsqueeze(0),
                mask2_sync.unsqueeze(0),
                dim=1
            )
            continuity_scores.append(similarity.item())
        else:
            continuity_scores.append(0.0)
    
    if len(continuity_scores) == 0:
        return torch.tensor(0.0)
    
                             
    avg_continuity = np.mean(continuity_scores)
    
    return torch.tensor(avg_continuity)


def compute_temporal_continuity_detailed(masks: list) -> dict:
       
                                                                    
    
         
                                       
        
            
                                      
       
    if len(masks) < 2:
        return {
            'mean_continuity': 0.0,
            'std_continuity': 0.0,
            'min_continuity': 0.0,
            'max_continuity': 0.0,
            'num_transitions': 0
        }
    
    continuity_scores = []
    
    for i in range(len(masks) - 1):
        mask1 = masks[i].flatten()
        mask2 = masks[i+1].flatten()
        
        min_size = min(mask1.size(0), mask2.size(0))
        
        if min_size > 0:
            mask1_sync = mask1[:min_size]
            mask2_sync = mask2[:min_size]
            
            similarity = torch.cosine_similarity(
                mask1_sync.unsqueeze(0),
                mask2_sync.unsqueeze(0),
                dim=1
            )
            continuity_scores.append(similarity.item())
    
    if len(continuity_scores) == 0:
        return {
            'mean_continuity': 0.0,
            'std_continuity': 0.0,
            'min_continuity': 0.0,
            'max_continuity': 0.0,
            'num_transitions': 0
        }
    
    continuity_scores = np.array(continuity_scores)
    
    return {
        'mean_continuity': np.mean(continuity_scores),
        'std_continuity': np.std(continuity_scores),
        'min_continuity': np.min(continuity_scores),
        'max_continuity': np.max(continuity_scores),
        'median_continuity': np.median(continuity_scores),
        'num_transitions': len(continuity_scores)
    }


def compute_smooth_transitions(masks: list, threshold: float = 0.1) -> dict:
       
                                                           
    
         
                               
                                                            
        
            
                                                     
       
    if len(masks) < 2:
        return {
            'smooth_transitions': 0,
            'abrupt_transitions': 0,
            'smooth_ratio': 0.0
        }
    
    smooth_count = 0
    abrupt_count = 0
    
    for i in range(len(masks) - 1):
        mask1 = masks[i].flatten()
        mask2 = masks[i+1].flatten()
        
        min_size = min(mask1.size(0), mask2.size(0))
        
        if min_size > 0:
            mask1_sync = mask1[:min_size]
            mask2_sync = mask2[:min_size]
            
                                
            diff = torch.mean(torch.abs(mask1_sync - mask2_sync)).item()
            
            if diff < threshold:
                smooth_count += 1
            else:
                abrupt_count += 1
    
    total = smooth_count + abrupt_count
    smooth_ratio = smooth_count / total if total > 0 else 0.0
    
    return {
        'smooth_transitions': smooth_count,
        'abrupt_transitions': abrupt_count,
        'smooth_ratio': smooth_ratio,
        'total_transitions': total
    }
