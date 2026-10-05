import numpy as np

REF_LAB = {
    "red":    np.array([54.29, 80.81, 69.89]),
    "orange": np.array([70.21, 39.79, 76.09]),
    "yellow": np.array([97.61, -15.75,  93.39]),
    "green":  np.array([46.28, -47.56, 48.58]),
    "cyan":   np.array([91.11, -48.08, -14.12]),
    "blue":   np.array([29.57, 68.30, -112.03]),
    "purple": np.array([69.62, 53.30, -36.54]),
    # 黑白灰建议单独判断，这里作为理论参考
    "black":  np.array([0, 0, 0]),
    "white":  np.array([100, 0, 0]),
    "gray":   np.array([53.59, 0.00, -0.00]),
}