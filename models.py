import numpy as np


def sigmoid(x, x50, s):
    x = np.asarray(x, dtype=float)
    s = max(float(s), 1e-9)
    return 1.0 / (1.0 + np.exp(-(x - x50) / s))


def cu_model(BV, x50, s, leak):
    return leak + (1.0 - leak) * sigmoid(BV, x50, s)


def co_model(BV, x50, s, amp, peak, width, leak):
    base = leak + (1.0 - leak) * sigmoid(BV, x50, s)
    width = max(float(width), 1e-9)
    rollup = amp * np.exp(-0.5 * ((np.asarray(BV) - peak) / width) ** 2)
    return base + rollup
