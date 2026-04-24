import numpy as np


def linear_probe_fit_predict(x, y, l2=1e-4):
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.shape[0]
    split = int(n * 0.8)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    xtx = x_train.T @ x_train
    w = np.linalg.solve(xtx + l2 * np.eye(xtx.shape[0]), x_train.T @ y_train)
    pred = x_test @ w
    return y_test, pred


def regression_scores(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = ((y_true - y_pred) ** 2).mean()
    var = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).mean() + 1e-8
    r2 = 1.0 - mse / var
    return {"mse": float(mse), "r2": float(r2)}


def straightness_score(z_seq):
    z = np.asarray(z_seq)
    dz1 = z[:, 1:-1] - z[:, :-2]
    dz2 = z[:, 2:] - z[:, 1:-1]
    dot = (dz1 * dz2).sum(-1)
    denom = np.linalg.norm(dz1, axis=-1) * np.linalg.norm(dz2, axis=-1) + 1e-8
    cos = dot / denom
    curvature = np.linalg.norm(dz2 - dz1, axis=-1)
    return {
        "straightness_cos": float(np.nanmean(cos)),
        "curvature": float(np.nanmean(curvature)),
    }


def auc_from_scores(normal_scores, invalid_scores):
    scores = np.concatenate([normal_scores, invalid_scores])
    labels = np.concatenate([
        np.zeros_like(normal_scores, dtype=np.int64),
        np.ones_like(invalid_scores, dtype=np.int64),
    ])
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores)) + 1
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    rank_pos = ranks[labels == 1].sum()
    auc = (rank_pos - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)
    return float(auc)


def spearman_rank_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx.astype(np.float64)
    ry = ry.astype(np.float64)
    rx = (rx - rx.mean()) / (rx.std() + 1e-8)
    ry = (ry - ry.mean()) / (ry.std() + 1e-8)
    return float((rx * ry).mean())
