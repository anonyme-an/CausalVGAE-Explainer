   
                                       
                                              
   

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
                                                    
    input_dim: int = 172                      
    hidden_dim: int = 64
    latent_dim: int = 32
    causal_dim: int = 16
    time_dim: int = 16
    
                  
    num_gat_heads: int = 4
    dropout: float = 0.3
    
            
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingConfig:
                                                    
    target_model_epochs: int = 50
    explainer_epochs: int = 50
    
                         
    validation_split: float = 0.2
    
                
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
               
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
                       
    max_grad_norm: float = 1.0
    
                                   
    loss_weights: dict = None
    
    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = {
                'recon': 1.0,
                'kl': 0.1,
                'causal': 0.3,
                'fidelity': 0.4,
                'sparsity': 0.2,
                'temporal': 0.1,
                'char': 0.3
            }


@dataclass
class DataConfig:
                                                  
    wikipedia_path: str = r"C:\Users\omara\OneDrive\Desktop\wikipedia.csv"
    uci_messages_path: Optional[str] = None
    
                   
    window_size: int = 100
    overlap: int = 50
    
              
    wikipedia_features: int = 172
    uci_features: int = 50


@dataclass
class ExperimentConfig:
                                                                           
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    
                           
    results_dir: str = "results"
    checkpoints_dir: str = "models_checkpoints"
    figures_dir: str = "results/figures"
    
                            
    run_tgnn: bool = True
    run_tgat: bool = True
    run_wikipedia: bool = True
    run_uci_messages: bool = False
    
                   
    num_snapshots_viz: int = 4
    save_figures: bool = True
    
                      
    random_seed: int = 42
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()


def get_default_config() -> ExperimentConfig:
       
                                        
    
            
                              
       
    return ExperimentConfig()


def print_config(config: ExperimentConfig):
       
                            
    
         
                                        
       
    print("\n" + "="*70)
    print("CONFIGURATION CAUSALVGAE-EXPLAINER")
    print("="*70)
    
    print("\n📊 MODÈLE:")
    print(f"   Input dim: {config.model.input_dim}")
    print(f"   Hidden dim: {config.model.hidden_dim}")
    print(f"   Latent dim: {config.model.latent_dim}")
    print(f"   Causal dim: {config.model.causal_dim}")
    print(f"   Time dim: {config.model.time_dim}")
    print(f"   Device: {config.model.device}")
    
    print("\n🎯 ENTRAÎNEMENT:")
    print(f"   Target model epochs: {config.training.target_model_epochs}")
    print(f"   Explainer epochs: {config.training.explainer_epochs}")
    print(f"   Learning rate: {config.training.learning_rate}")
    print(f"   Validation split: {config.training.validation_split}")
    
    print("\n📁 DONNÉES:")
    print(f"   Wikipedia: {config.data.wikipedia_path}")
    print(f"   Window size: {config.data.window_size}")
    print(f"   Overlap: {config.data.overlap}")
    
    print("\n🔬 EXPÉRIENCES:")
    print(f"   TGNN: {'✅' if config.run_tgnn else '❌'}")
    print(f"   TGAT: {'✅' if config.run_tgat else '❌'}")
    print(f"   Wikipedia: {'✅' if config.run_wikipedia else '❌'}")
    print(f"   UCI-Messages: {'✅' if config.run_uci_messages else '❌'}")
    
    print("\n" + "="*70 + "\n")
