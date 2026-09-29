   
                                          
                                                     
   

import numpy as np
import torch


def compute_aufsc_plus(fidelity_plus_scores: np.ndarray,
                      sparsity_scores: np.ndarray) -> float:
       
                                                        
    
                                                 
                                                     
    
         
                                                     
                                               
        
            
                                  
       
    if len(fidelity_plus_scores) < 2:
        return 0.0
    
                                           
    sorted_indices = np.argsort(sparsity_scores)
    sorted_sparsity = sparsity_scores[sorted_indices]
    sorted_fidelity = fidelity_plus_scores[sorted_indices]
    
                                        
    unique_sparsity, unique_indices = np.unique(sorted_sparsity, return_index=True)
    unique_fidelity = sorted_fidelity[unique_indices]
    
    if len(unique_sparsity) > 1:
                                                       
        aufsc_plus = np.trapz(unique_fidelity, unique_sparsity)
    else:
        aufsc_plus = 0.0
    
    return aufsc_plus


def compute_aufsc_minus(fidelity_minus_scores: np.ndarray,
                       sparsity_scores: np.ndarray) -> float:
       
                                                        
    
                                                                   
                                           
    
                                       
    
         
                                                      
                                               
        
            
                                  
       
    if len(fidelity_minus_scores) < 2:
        return 0.0
    
                                                       
                                      
    fidelity_transformed = fidelity_minus_scores
    
                        
    sorted_indices = np.argsort(sparsity_scores)
    sorted_sparsity = sparsity_scores[sorted_indices]
    sorted_fidelity_transformed = fidelity_transformed[sorted_indices]
    
                            
    unique_sparsity, unique_indices = np.unique(sorted_sparsity, return_index=True)
    unique_fidelity_transformed = sorted_fidelity_transformed[unique_indices]
    
    if len(unique_sparsity) > 1:
                                                          
        aufsc_minus = np.trapz(unique_fidelity_transformed, unique_sparsity)
    else:
        aufsc_minus = 0.0
    
    return aufsc_minus


def compute_aufsc_metrics(fidelity_plus_scores: np.ndarray,
                         fidelity_minus_scores: np.ndarray,
                         sparsity_scores: np.ndarray) -> dict:
       
                                      
    
         
                                             
                                              
                                               
        
            
                                          
       
    aufsc_plus = compute_aufsc_plus(fidelity_plus_scores, sparsity_scores)
    aufsc_minus = compute_aufsc_minus(fidelity_minus_scores, sparsity_scores)
    
    return {
        'aufsc_plus': aufsc_plus,
        'aufsc_minus': aufsc_minus,
        'aufsc_combined': (aufsc_plus + aufsc_minus) / 2               
    }


def plot_fidelity_sparsity_curves(fidelity_plus_scores: np.ndarray,
                                  fidelity_minus_scores: np.ndarray,
                                  sparsity_scores: np.ndarray,
                                  save_path: str = 'fidelity_sparsity_curves.png'):
       
                                       
    
         
                                         
                                          
                                           
                                       
       
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
                                  
    ax1.scatter(sparsity_scores, fidelity_plus_scores, alpha=0.6, c='green')
    ax1.set_xlabel('Sparsité')
    ax1.set_ylabel('Fidélité+')
    ax1.set_title('Courbe Fidélité+-Sparsité')
    ax1.grid(True, alpha=0.3)
    
                                  
    ax2.scatter(sparsity_scores, fidelity_minus_scores, alpha=0.6, c='orange')
    ax2.set_xlabel('Sparsité')
    ax2.set_ylabel('Fidélité-')
    ax2.set_title('Courbe Fidélité--Sparsité')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Courbes sauvegardées: {save_path}")
