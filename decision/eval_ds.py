import argparse
import json
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch

from decision.analyze_latent import extract_latents
from decision.metrics import straightness_score
from decision.probe import run_decision_probes


def get_model(run_name):
    model = swm.policy.AutoCostModel(run_name)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model.requires_grad_(False)
    return model.model if hasattr(model, "model") else model


def get_loader(dataset_name, batch_size=64, num_steps=4):
    dataset = swm.data.HDF5Dataset(
        name=dataset_name,
        num_steps=num_steps,
        keys_to_load=["pixels", "action", "state"],
        keys_to_cache=["action", "state"],
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def evaluate_ckpt(run_name, dataset_name):
    loader = get_loader(dataset_name)
    model = get_model(run_name)
    latents = extract_latents(model, loader, device="cuda" if torch.cuda.is_available() else "cpu")

    metrics = {}
    metrics.update(straightness_score(latents["z"]))
    metrics.update(run_decision_probes(latents))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--dataset", default="pusht_expert_train")
    parser.add_argument("--output", default="decision_metrics.json")
    args = parser.parse_args()

    results = {}
    for ckpt in args.ckpts:
        results[ckpt] = evaluate_ckpt(ckpt, args.dataset)
        print(f"[{ckpt}] {results[ckpt]}")

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved metrics to {out.resolve()}")


if __name__ == "__main__":
    main()
