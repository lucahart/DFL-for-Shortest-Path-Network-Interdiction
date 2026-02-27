
import torch
import numpy as np

def feature_cost_mapping(w):
    """
    Docstring for feature_cost_mapping

    :param w: Description
    :return: Description
    """
    if torch.is_tensor(w):
        w_t = w.squeeze(-1) if w.ndim > 0 and w.shape[-1] == 1 else w
        return torch.stack((w_t, w_t, -w_t, -w_t), dim=-1)
    if isinstance(w, np.ndarray):
        w_arr = np.squeeze(w, axis=-1) if w.ndim > 0 and w.shape[-1] == 1 else w
        return np.stack((w_arr, w_arr, -w_arr, -w_arr), axis=-1)
    return [w, w, -w, -w]

def optimizer(c):
    """
    Docstring for optimizer
    
    :param c: Description
    :return: Description
    """
    if torch.is_tensor(c):
        lhs = c[..., 0] + c[..., 1]
        rhs = c[..., 2] + c[..., 3]
        cond = lhs <= rhs
        ones = torch.ones_like(lhs)
        zeros = torch.zeros_like(lhs)
        y_left = torch.stack((ones, ones, zeros, zeros), dim=-1)
        y_right = torch.stack((zeros, zeros, ones, ones), dim=-1)
        return torch.where(cond.unsqueeze(-1), y_left, y_right)
    if isinstance(c, np.ndarray):
        lhs = c[..., 0] + c[..., 1]
        rhs = c[..., 2] + c[..., 3]
        cond = lhs <= rhs
        ones = np.ones_like(lhs)
        zeros = np.zeros_like(lhs)
        y_left = np.stack((ones, ones, zeros, zeros), axis=-1)
        y_right = np.stack((zeros, zeros, ones, ones), axis=-1)
        return np.where(cond[..., None], y_left, y_right)
    if c[0] + c[1] <= c[2] + c[3]:
        return [1, 1, 0, 0]
    return [0, 0, 1, 1]

def sol_value(c, y):
    """
    Docstring for sol_value
    
    :param c: Description
    :param y: Description
    :return: Description
    """
    if torch.is_tensor(c) and torch.is_tensor(y):
        return torch.sum(c * y, dim=-1)
    if isinstance(c, np.ndarray) and isinstance(y, np.ndarray):
        return np.sum(c * y, axis=-1)
    return sum(c[i] * y[i] for i in range(len(c)))
    
def interdictor(c, d=3):
    """
    Docstring for interdictor
    
    :param c: Description
    :return: Description
    """
    if torch.is_tensor(c):
        lhs = c[..., 0] + c[..., 1]
        rhs = c[..., 2] + c[..., 3]
        cond = lhs <= rhs
        zeros = torch.zeros_like(lhs)
        intds = torch.full_like(lhs, d)
        g_left = torch.stack((zeros, intds, zeros, zeros), dim=-1)
        g_right = torch.stack((zeros, zeros, zeros, intds), dim=-1)
        return torch.where(cond.unsqueeze(-1), g_left, g_right)
    if isinstance(c, np.ndarray):
        lhs = c[..., 0] + c[..., 1]
        rhs = c[..., 2] + c[..., 3]
        cond = lhs <= rhs
        zeros = np.zeros_like(lhs)
        intds = np.full_like(lhs, d)
        g_left = np.stack((zeros, intds, zeros, zeros), axis=-1)
        g_right = np.stack((zeros, zeros, zeros, intds), axis=-1)
        return np.where(cond[..., None], g_left, g_right)
    if c[0] + c[1] <= c[2] + c[3]:
        return [0, d, 0, 0]
    return [0, 0, 0, d]
    
def test_intd_pipeline_volatile():
    w = -1
    print(f"Input                w:  {w}")
    c = feature_cost_mapping(w)
    print(f"Cost                 c: {c}")
    y = optimizer(c)
    print(f"Opt. sol.            y: {y}")
    g_c = interdictor(c)
    print(f"Intd.              g_c: {g_c}")
    c_intd = [c[i] + g_c[i] for i in range(len(c))]
    print(f"Intd. cost     c + g_c: {c_intd}")
    y_intd = optimizer(c_intd)
    print(f"Intd. opt. sol. y_intd: {y_intd}")
    pass
