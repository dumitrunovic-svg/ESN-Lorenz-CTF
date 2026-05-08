"""
ESN_Tuned: Echo State Network with evolved hyperparameters.
CTF for Science compatible model runner.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ctf4science.data_module import load_dataset, get_prediction_timesteps, parse_pair_ids, get_applicable_plots
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization

from esn import ESN

CTF_ROOT = Path("/opt/projects/ctf4science")


MODEL_NAME = "ESN_Tuned"


def build_esn_from_config(config: dict, seed: int = 42) -> ESN:
    """Construct ESN from config dict (model section)."""
    m = config.get("model", {})
    return ESN(
        reservoir_size=int(m.get("reservoir_size", 500)),
        spectral_radius=float(m.get("spectral_radius", 0.9)),
        input_scaling=float(m.get("input_scaling", 0.1)),
        leaking_rate=float(m.get("leaking_rate", 1.0)),
        ridge_alpha=float(m.get("ridge_alpha", 1e-6)),
        washout=int(m.get("washout", 100)),
        seed=seed,
    )


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    pair_ids = parse_pair_ids(config["dataset"])

    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    batch_results = {
        "batch_id": batch_id,
        "model": MODEL_NAME,
        "dataset": dataset_name,
        "pairs": [],
    }

    viz = Visualization()
    applicable_plots = get_applicable_plots(dataset_name)

    for pair_id in pair_ids:
        train_data, init_data = load_dataset(dataset_name, pair_id, transpose=True)

        n_steps = get_prediction_timesteps(dataset_name, pair_id).shape[0]

        esn = build_esn_from_config(config, seed=42 + pair_id)

        if init_data is not None:
            # Pairs 8, 9: multi-trajectory training + separate init
            X_full = np.concatenate(train_data, axis=1).T
            X_init = init_data.T
            esn.fit(X_full)
            pred = esn.predict(n_steps, init_data=X_init)
        else:
            X_train = np.concatenate(train_data, axis=1).T
            pred = esn.fit_predict(X_train, n_steps)

        # Detect if test data exists locally (dev set) or is hidden (official set)
        test_dir = CTF_ROOT / "data" / dataset_name / "test"
        has_local_test = test_dir.exists() and any(test_dir.iterdir())

        if has_local_test:
            results = evaluate(dataset_name, pair_id, pred)
        else:
            # Official set: skip evaluation, predictions only
            results = {"submission_only": True}

        results_directory = save_results(
            dataset_name, MODEL_NAME, batch_id, pair_id, config, pred, results
        )

        batch_results["pairs"].append({"pair_id": pair_id, "metrics": results})

        print(f"  Pair {pair_id}: {results}")

        for plot_type in applicable_plots:
            try:
                fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
                viz.save_figure_results(fig, dataset_name, MODEL_NAME, batch_id, pair_id, plot_type, results_directory)
            except Exception:
                pass

    with open(results_directory.parent / "batch_results.yaml", "w") as f:
        yaml.dump(batch_results, f)

    # Print summary
    all_scores = []
    print(f"\n=== {MODEL_NAME} on {dataset_name} ===")
    for p in batch_results["pairs"]:
        for m, v in p["metrics"].items():
            if isinstance(v, (int, float)):
                print(f"  E{p['pair_id']} {m}: {v:.2f}")
                all_scores.append(v)
            else:
                print(f"  E{p['pair_id']} {m}: {v}")
    if all_scores:
        print(f"  Mean score: {np.mean(all_scores):.2f}")
    print(f"Results saved in: {results_directory.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
