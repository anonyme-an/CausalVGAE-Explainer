                                                

import argparse
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_SCRIPTS = {
    "tgnn": PROJECT_ROOT / "causalvgae-explainer" / "tgnn.py",
    "tgat": PROJECT_ROOT / "causalvgae-explainer" / "tgat.py",
}


def load_training_module(model_name):
    script_path = MODEL_SCRIPTS[model_name]
    spec = importlib.util.spec_from_file_location(f"causalvgae_x_{model_name}", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load training module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Train CausalVGAE-X on Wikipedia or Reddit."
    )
    parser.add_argument("--model", choices=MODEL_SCRIPTS, default="tgnn")
    parser.add_argument(
        "--dataset", choices=("wikipedia", "reddit"), default="wikipedia"
    )
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / "resources" / "datasets" / "raw" / f"{args.dataset}.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"CausalVGAE-X | model={args.model} | dataset={args.dataset}")
    load_training_module(args.model).main(args.dataset)


if __name__ == "__main__":
    main()
