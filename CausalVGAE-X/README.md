# CausalVGAE-X

CausalVGAE-X is a temporal graph explanation project with TGNN and TGAT architectures. It supports the Wikipedia and Reddit datasets from one unified entry point.

## Structure

- `train.py`: unified training launcher
- `causalvgae-explainer/tgnn.py`: TGNN architecture and training loop
- `causalvgae-explainer/tgat.py`: TGAT architecture and training loop
- `causalvgae-explainer/src/`: reusable data, model, metric, trainer, and visualization modules
- `resources/datasets/raw/`: `wikipedia.csv` and `reddit.csv`
- `results/`: generated training results

## Installation

Use Python 3.11 or newer, create an environment, and install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Training

Run commands from the `CausalVGAE-X` directory:

```powershell
python train.py --model tgnn --dataset wikipedia
python train.py --model tgnn --dataset reddit
python train.py --model tgat --dataset wikipedia
python train.py --model tgat --dataset reddit
```

The default model is `tgnn` and the default dataset is `wikipedia`.

The same launcher is available from the `causalvgae-explainer` directory:

```powershell
cd causalvgae-explainer
python train.py --model tgat --dataset reddit
```

Results, checkpoints, and metrics are stored with model and dataset names so runs remain separate.
