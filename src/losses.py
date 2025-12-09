import numpy as np
from typing import Sequence

def method2_loss(
    de_new: Sequence[float],
    de_old: Sequence[float],
    constraint: Sequence[float],
    verify: Sequence[float],
    x: float = 1.0,
    y: float = 0.3,
    z: float = 0.3,
    f: str = "relu",
) -> list[float]:
    """
    L = ( x*(de(s,t) - de(s0,t0)) + y*constraint(s) + z*verify(s) ) / (x+y+z)
    """
    de_new = np.array(de_new, dtype=float)
    de_old = np.array(de_old, dtype=float)
    cons = np.array(constraint, dtype=float)
    ver = np.array(verify, dtype=float)

    delta = de_new - de_old
    if f == "relu":
        delta = np.maximum(0.0, delta)
    elif f == "sigmoid":
        delta = 1/(1 + np.exp(-delta))

    wsum = x + y + z
    L = (x*delta + y*cons + z*ver) / wsum
    return L.tolist()
