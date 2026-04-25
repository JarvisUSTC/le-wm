import numpy as np
import torch

from decision.losses import compute_reward_from_pusht_state, discounted_returns
from decision.metrics import linear_probe_fit_predict, regression_scores


def flatten_bt(x):
    b, t = x.shape[:2]
    return x.reshape(b * t, *x.shape[2:])


def run_decision_probes(latents):
    z = latents["z"]
    state = latents.get("state")
    action = latents.get("action")
    goal_state = latents.get("goal_state")

    metrics = {}

    if state is not None:
        y_true, y_pred = linear_probe_fit_predict(flatten_bt(z), flatten_bt(state))
        state_scores = regression_scores(y_true, y_pred)
        metrics["state_probe_r2"] = state_scores["r2"]
        metrics["state_probe_mse"] = state_scores["mse"]

    if action is not None:
        z_t = z[:, :-1]
        z_tp1 = z[:, 1:]
        x = np.concatenate([flatten_bt(z_t), flatten_bt(z_tp1)], axis=-1)
        y = flatten_bt(action[:, :-1])
        y_true, y_pred = linear_probe_fit_predict(x, y)
        scores = regression_scores(y_true, y_pred)
        metrics["action_probe_r2"] = scores["r2"]
        metrics["action_probe_mse"] = scores["mse"]

    if state is not None and action is not None:
        rew = compute_reward_from_pusht_state(
            torch.from_numpy(state[:, :-1]),
            None if goal_state is None else torch.from_numpy(goal_state[:, :-1]),
        ).numpy()
        x_rew = np.concatenate([flatten_bt(z[:, :-1]), flatten_bt(action[:, :-1])], axis=-1)
        y_true, y_pred = linear_probe_fit_predict(x_rew, rew.reshape(-1, 1))
        scores = regression_scores(y_true, y_pred)
        metrics["reward_probe_r2"] = scores["r2"]
        metrics["reward_probe_mse"] = scores["mse"]

        ret = discounted_returns(torch.from_numpy(rew), gamma=0.99).numpy()
        y_true, y_pred = linear_probe_fit_predict(flatten_bt(z[:, :-1]), ret.reshape(-1, 1))
        scores = regression_scores(y_true, y_pred)
        metrics["value_probe_r2"] = scores["r2"]
        metrics["value_probe_mse"] = scores["mse"]

    return metrics
