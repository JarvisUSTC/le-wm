import numpy as np
import torch


@torch.no_grad()
def extract_latents(model, loader, device="cuda"):
    model.eval()
    all_z, all_a, all_state, all_goal = [], [], [], []

    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        out = model.encode(batch)
        all_z.append(out["emb"].detach().cpu().numpy())

        if "action" in batch:
            all_a.append(batch["action"].detach().cpu().numpy())
        if "state" in batch:
            all_state.append(batch["state"].detach().cpu().numpy())
        if "goal_state" in batch:
            all_goal.append(batch["goal_state"].detach().cpu().numpy())

    result = {"z": np.concatenate(all_z, axis=0)}
    if all_a:
        result["action"] = np.concatenate(all_a, axis=0)
    if all_state:
        result["state"] = np.concatenate(all_state, axis=0)
    if all_goal:
        result["goal_state"] = np.concatenate(all_goal, axis=0)
    return result
