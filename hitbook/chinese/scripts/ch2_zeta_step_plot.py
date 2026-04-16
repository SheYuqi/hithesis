#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYDEPS = ROOT / ".pydeps"
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ch2_simulation import SIM_FIGSIZE, configure_matplotlib, rk4_step, save_figure, style_axes


FIG_DIR = ROOT / "hitbook" / "chinese" / "figures"


def simulate_step_response(zeta: float, omega_n: float = 4.0, duration: float = 10.0, dt: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, duration + dt, dt)
    x = np.zeros((t.size, 2), dtype=float)

    def rhs(_t: float, xx: np.ndarray) -> np.ndarray:
        x1, x2 = xx
        dx1 = x2
        dx2 = -2.0 * zeta * omega_n * x2 - (omega_n**2) * x1 + (omega_n**2)
        return np.array([dx1, dx2], dtype=float)

    for k in range(t.size - 1):
        x[k + 1] = rk4_step(rhs, float(t[k]), x[k], dt)
    return t, x[:, 0]


def main() -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=SIM_FIGSIZE)

    cases = [
        (0.2, "black", (0, (5.0, 2.0, 1.0, 2.0))),
        (0.4, "#FF66CC", (0, (1.0, 2.0))),
        (0.8, "#FF0000", "--"),
        (1.0, "#0000FF", "-."),
        (2.0, "#00CC00", "-"),
    ]

    for zeta, color, linestyle in cases:
        t, y = simulate_step_response(zeta)
        label = rf"$\zeta={zeta:g}$"
        ax.plot(t, y, color=color, linestyle=linestyle, linewidth=2.0, label=label)

    style_axes(ax, "时间 (s)", "输出")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 1.6)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])

    leg = ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="0.35")
    leg.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.96)
    save_figure(fig, FIG_DIR / "ch2_zeta_step_response")


if __name__ == "__main__":
    main()
