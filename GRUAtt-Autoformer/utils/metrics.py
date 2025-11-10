import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def R2(pred, true):
    ss_total = np.sum(np.square(true - np.mean(true)))
    ss_residual = np.sum(np.square(true - pred))
    r2 = 1 - (ss_residual / ss_total)
    return r2


def DS(pred, true):
    """计算方向对称性 (Directional Symmetry)"""
    N = len(true)
    if N < 2:
        return np.nan  # 数据点不足，无法计算
    a_t = []
    for t in range(1, N):
        y_diff = pred[t] - pred[t-1]
        d_diff = true[t] - true[t-1]
        a_t.append(1 if (y_diff * d_diff) >= 0 else 0)
    return 100 * np.mean(a_t) if a_t else np.nan


def DP(pred, true):
    """计算上升趋势方向精度 (Directional Precision for up trend)"""
    N = len(true)
    if N < 2:
        return np.nan
    a_t = []
    for t in range(1, N):
        d_diff = true[t] - true[t-1]
        if d_diff > 0:  # 真实值处于上升趋势
            y_diff = pred[t] - pred[t-1]
            a_t.append(1 if (y_diff * d_diff) >= 0 and y_diff > 0 else 0)
    N1 = len(a_t)
    return 100 * np.mean(a_t) if N1 > 0 else np.nan


def CD(pred, true):
    """计算下降趋势方向精度 (Directional Precision for down trend)"""
    N = len(true)
    if N < 2:
        return np.nan
    a_t = []
    for t in range(1, N):
        d_diff = true[t] - true[t-1]
        if d_diff < 0:  # 真实值处于下降趋势
            y_diff = pred[t] - pred[t-1]
            a_t.append(1 if (y_diff * d_diff) >= 0 and y_diff < 0 else 0)
    N2 = len(a_t)
    return 100 * np.mean(a_t) if N2 > 0 else np.nan


def metric(pred, true):
    """返回所有评估指标，包括新增的DS、DP、CD"""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    r2 = R2(pred, true)
    ds = DS(pred, true)
    dp = DP(pred, true)
    cd = CD(pred, true)
    return mae, mse, rmse, mape, r2, ds, dp, cd
