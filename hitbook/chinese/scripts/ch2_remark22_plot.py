#!/usr/bin/env python3
from __future__ import annotations

import math
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

from ch2_simulation import configure_matplotlib, save_figure, style_axes


FIG_DIR = ROOT / "hitbook" / "chinese" / "figures"


def rk4_step(rhs, t: float, x: np.ndarray, dt: float) -> np.ndarray:
    k1 = rhs(t, x)
    k2 = rhs(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = rhs(t + dt, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_response(m1: float, m2: float, forcing, duration: float = 8.0, dt: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, duration + dt, dt)
    x = np.zeros((t.size, 2), dtype=float)
    x[0] = np.array([-1.0, 0.0], dtype=float)

    def rhs(tt: float, xx: np.ndarray) -> np.ndarray:
        e1, e1_dot = xx
        e1_ddot = forcing(tt) - m2 * e1_dot - m1 * e1
        return np.array([e1_dot, e1_ddot], dtype=float)

    for k in range(t.size - 1):
        x[k + 1] = rk4_step(rhs, float(t[k]), x[k], dt)
    return t, x[:, 0]


def make_case_plot(name: str, m1: float, m2: float, zeta_text: str, out_name: str) -> None:
    t, e_hat = simulate_response(m1, m2, lambda _: 0.2)
    _, e_main = simulate_response(m1, m2, lambda tt: 0.2 * math.sin(5.0 * tt))
    _, e_check = simulate_response(m1, m2, lambda _: -0.2)

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.plot(t, e_hat, color="#FF6A6A", linestyle="--", linewidth=2.0, label=r"$\hat e_1$")
    ax.plot(t, e_main, color="#1F77B4", linestyle="-", linewidth=2.0, label=r"$e_1$")
    ax.plot(t, e_check, color="#66AA55", linestyle="-.", linewidth=2.0, label=r"$\check e_1$")

    style_axes(ax, "时间 (s)", r"误差 $e_1$")
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(-1.0, 0.42)
    ax.set_yticks(np.arange(-1.0, 0.41, 0.2))
    ax.set_title(fr"{name}: $m_2={m2:g},\,m_1={m1:g},\,\zeta {zeta_text}$", fontsize=12, pad=8)

    leg = ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="0.35")
    leg.get_frame().set_linewidth(0.8)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.88)
    save_figure(fig, FIG_DIR / out_name)


def main() -> None:
    make_case_plot("情形 A", 2.25, 3.0, "=1", "ch2_remark22_caseA")
    make_case_plot("情形 B", 2.0, 2.0, r"\approx0.707", "ch2_remark22_caseB")


if __name__ == "__main__":
    main()
