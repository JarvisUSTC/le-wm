import torch
import torch.nn.functional as F


def _angle_distance(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b)).abs()


def compute_reward_from_pusht_state(state, goal_state=None):
    """Proxy reward from state/goal tensors using position and angle.

    Assumes first dims of state encode xy + angle (PushT default).
    """
    if goal_state is None:
        return torch.zeros(state.shape[:-1], device=state.device, dtype=state.dtype)

    pos_err = torch.norm(state[..., :2] - goal_state[..., :2], dim=-1)
    ang_err = _angle_distance(state[..., 2], goal_state[..., 2])
    return -(pos_err + 0.1 * ang_err)


def discounted_returns(rewards, gamma=0.99):
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(rewards.shape[1])):
        running = rewards[:, t] + gamma * running
        returns[:, t] = running
    return returns


def decision_losses(heads, emb, action, state=None, goal_state=None, gamma=0.99):
    """Compute decision-sufficiency auxiliary losses."""
    z_t = emb[:, :-1]
    z_tp1 = emb[:, 1:]
    a_t = action[:, :-1]

    flat_z_t = z_t.reshape(-1, z_t.size(-1))
    flat_z_tp1 = z_tp1.reshape(-1, z_tp1.size(-1))
    flat_a_t = a_t.reshape(-1, a_t.size(-1))

    losses = {}
    pred_a = heads.inverse_action(flat_z_t, flat_z_tp1)
    losses["action_loss"] = F.mse_loss(pred_a, flat_a_t)

    if state is not None:
        rewards = compute_reward_from_pusht_state(state[:, :-1], None if goal_state is None else goal_state[:, :-1])
        returns = discounted_returns(rewards, gamma=gamma)

        pred_r = heads.predict_reward(flat_z_t, flat_a_t)
        pred_v = heads.predict_value(flat_z_t)
        losses["reward_loss"] = F.mse_loss(pred_r, rewards.reshape(-1))
        losses["value_loss"] = F.mse_loss(pred_v, returns.reshape(-1))
    else:
        zero = emb.sum() * 0.0
        losses["reward_loss"] = zero
        losses["value_loss"] = zero

    return losses
