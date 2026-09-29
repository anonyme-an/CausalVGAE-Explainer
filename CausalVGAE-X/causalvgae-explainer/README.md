# causalvgae-explainer - Temporal Graph Explainability

**Explainability Framework for Temporal Graph Neural Networks**

Implementation of causalvgae-explainer for explaining predictions of temporal graph models (TGNN and TGAT) on dynamic networks.

---

## ðŸ“ Project Structure

```
V1/
â”œâ”€â”€ main.py                 # ðŸ”µ TGNN - Main training script
â”œâ”€â”€ main_tgat.py            # ðŸŸ  TGAT - Main training script
â”œâ”€â”€ requirements.txt        # Dependencies
â”œâ”€â”€ README.md               # This file
â”‚
â”œâ”€â”€ configs/                # Configuration
â”‚   â””â”€â”€ config.py           # Hyperparameters
â”‚
â”œâ”€â”€ src/                    # Source modules
â”‚   â”œâ”€â”€ data/               # Data loading & processing
â”‚   â”œâ”€â”€ models/             # Model architectures
â”‚   â”œâ”€â”€ metrics/            # Evaluation metrics
â”‚   â”œâ”€â”€ trainers/           # Training logic
â”‚   â””â”€â”€ visualizers/        # Visualization tools
â”‚
â”œâ”€â”€ models_checkpoints/     # Saved model weights
â”‚
â””â”€â”€ results/                # Training results & figures
    â”œâ”€â”€ figures/
    â””â”€â”€ *.pth, *.csv
```

---

## ðŸš€ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train TGNN Model
```bash
python main.py
```

### 3. Train TGAT Model
```bash
python main_tgat.py
```

### Ã‰tude de sensibilitÃ© et d'ablation

Le script `sensitivity_ablation.py` recharge un checkpoint causalvgae-explainer et
calcule les courbes sur les sorties rÃ©elles du modÃ¨le. Il produit les fichiers
CSV, JSON et les figures PNG/PDF/SVG dans `results/sensitivity_ablation`.

```bash
cd v2
python sensitivity_ablation.py \
    --data ..\resources\datasets\raw\wikipedia.csv \
    --checkpoint best_causalvgae_explainer_causalvgae_explainer.pth \
    --max-snapshots 12 --device cuda
```

Les courbes de sensibilitÃ© varient le pourcentage de nÅ“uds conservÃ©s et le
coefficient de dÃ©croissance temporelle. L'ablation compare `full`,
`no_temporal`, `random`, `edge_importance` et `uniform`. Les CSV `*_raw`
conservent les scores par snapshot; les CSV `*_summary` donnent moyenne,
Ã©cart-type et IC95. Les dÃ©finitions exactes des mÃ©triques sont enregistrÃ©es
dans `study_config.json`.

---

## ðŸ“Š Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Fidelity+** | `fid+ = 1/N Î£ ðŸ™(p(f(G),Îµ) = p(f(X_Îµ),Îµ))` | Agreement with explanation |
| **Fidelity-** | `fid- = 1 - 1/N Î£ ðŸ™(p(f(G),Îµ) = p(f(G\X_Îµ),Îµ))` | Disagreement without explanation |
| **Sparsity** | `\|X_Îµ\| / \|C(G,Îµ,k,nmax)\|` | Explanation compactness |
| **Char Score** | `(w+ + w-) / (fid+/w+ + fid-/w-)` | Weighted characterization |
| **Temporal Continuity** | Jaccard similarity | Explanation stability over time |
| **AUFSC** | Area under Fidelity-Sparsity curve | Overall performance |

---

## ðŸŽ¯ Target Results

| Metric | TGNN Target | TGAT Target |
|--------|-------------|-------------|
| Fidelity+ | > 0.75 | > 0.70 |
| Fidelity- | > 0.65 | > 0.60 |
| Sparsity | 0.30-0.50 | 0.30-0.50 |
| Char Score | > 0.30 | > 0.25 |
| Temporal | > 0.95 | > 0.90 |

---

## ðŸ“ˆ Model Architectures

### TGNN (Temporal Graph Neural Network)
- GCN-based encoder
- Temporal encoding with sinusoidal functions
- Causal VGAE for explanation generation

### TGAT (Temporal Graph Attention Network)
- GATv2Conv with multi-head attention
- 4 attention heads
- Residual connections with LayerNorm

---

## ðŸ“‚ Dataset

**Wikipedia** temporal interaction network:
- 157,474 events
- 9,228 unique nodes
- 172-dimensional features
- Binary state labels

Path: `C:\Users\omara\OneDrive\Desktop\wikipedia.csv`

---

## ðŸ“œ License

MIT License

