#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np

from ch2_simulation import (
    ROOT,
    B_GAIN,
    K1_GAIN,
    DampingCase,
    SIM_FIGSIZE,
    SINE_PERIOD,
    add_zoom_inset,
    basis_vector,
    configure_matplotlib,
    equivalent_damping_ratio,
    save_figure,
    sine_reference,
    step_reference,
    style_axes,
    true_nonlinearity,
    rk4_step,
)

import matplotlib.pyplot as plt


FIG_DIR = ROOT / "hitbook" / "chinese" / "figures" / "ch4_sim"
W_CF = np.array([0.35, -0.28, 0.40, 0.08, 0.12, 0.05], dtype=float)
FILTER_ZETA = 0.90
FILTER_WN = 80.0
ROBUST_GAIN = 0.04
SAT_WIDTH = 0.18
U_MAX = 0.14
AW_LAMBDA = 12.0
AW_GAIN = -6.0


def sat(x: float) -> float:
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return x


def clip_u(u: float) -> float:
    return max(-U_MAX, min(U_MAX, u))


def simulate_saturation_case(
    zeta: float,
    omega_n: float,
    reference: Callable[[float], Tuple[float, float, float]],
    duration: float = SINE_PERIOD,
    dt: float = 0.001,
    anti_windup: bool = True,
) -> Dict[str, np.ndarray]:
    times = np.arange(0.0, duration + dt, dt)
    state = np.zeros((times.size, 6), dtype=float)  # x1, x2, xi1, xi2, q, xi_aw
    u_cmd = np.zeros(times.size, dtype=float)
    u_act = np.zeros(times.size, dtype=float)
    yd = np.zeros(times.size, dtype=float)
    m1 = omega_n**2
    m2 = 2.0 * zeta * omega_n

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        x1, x2, xi1, xi2, q, xi_aw = x
        ref, ref_dot, _ = reference(t)
        z1 = x1 - ref
        alpha10 = ref_dot - K1_GAIN * z1
        z2 = x2 - xi1
        s = z2 + (m2 - K1_GAIN) * z1 + m1 * q
        h_est = float(np.dot(W_CF, basis_vector(x1, x2)))
        u_star = -((m2 - K1_GAIN) * z2 + (m1 - K1_GAIN * (m2 - K1_GAIN)) * z1 + h_est - xi2 + ROBUST_GAIN * sat(s / SAT_WIDTH)) / B_GAIN
        u_c = u_star - (AW_GAIN / B_GAIN) * xi_aw if anti_windup else u_star
        u = clip_u(u_c)
        delta_u = u - u_c
        dx1 = x2
        dx2 = B_GAIN * u + true_nonlinearity(x1, x2)
        dxi1 = xi2
        dxi2 = -2.0 * FILTER_ZETA * FILTER_WN * xi2 - FILTER_WN**2 * (xi1 - alpha10)
        dq = z1
        dxi_aw = -AW_LAMBDA * xi_aw + B_GAIN * delta_u if anti_windup else 0.0
        return np.array([dx1, dx2, dxi1, dxi2, dq, dxi_aw], dtype=float)

    for idx, t in enumerate(times[:-1]):
        ref, ref_dot, _ = reference(t)
        yd[idx] = ref
        x1, x2, xi1, xi2, q, xi_aw = state[idx]
        z1 = x1 - ref
        z2 = x2 - xi1
        s = z2 + (m2 - K1_GAIN) * z1 + m1 * q
        h_est = float(np.dot(W_CF, basis_vector(x1, x2)))
        u_star = -((m2 - K1_GAIN) * z2 + (m1 - K1_GAIN * (m2 - K1_GAIN)) * z1 + h_est - xi2 + ROBUST_GAIN * sat(s / SAT_WIDTH)) / B_GAIN
        u_c = u_star - (AW_GAIN / B_GAIN) * xi_aw if anti_windup else u_star
        u_cmd[idx] = u_c
        u_act[idx] = clip_u(u_c)
        state[idx + 1] = rk4_step(rhs, t, state[idx], dt)

    ref, ref_dot, _ = reference(times[-1])
    yd[-1] = ref
    x1, x2, xi1, xi2, q, xi_aw = state[-1]
    z1 = x1 - ref
    z2 = x2 - xi1
    s = z2 + (m2 - K1_GAIN) * z1 + m1 * q
    h_est = float(np.dot(W_CF, basis_vector(x1, x2)))
    u_star = -((m2 - K1_GAIN) * z2 + (m1 - K1_GAIN * (m2 - K1_GAIN)) * z1 + h_est - xi2 + ROBUST_GAIN * sat(s / SAT_WIDTH)) / B_GAIN
    u_c = u_star - (AW_GAIN / B_GAIN) * xi_aw if anti_windup else u_star
    u_cmd[-1] = u_c
    u_act[-1] = clip_u(u_c)

    return {"t": times, "y": state[:, 0], "e": state[:, 0] - yd, "u": u_act, "u_cmd": u_cmd, "yd": yd}



def transient_overshoot_from_trace(
    t: np.ndarray,
    y: np.ndarray,
    yd: np.ndarray,
    t0: float = 0.5,
    window: float = 0.8,
) -> float:
    mask = (t >= t0) & (t <= t0 + window)
    if not np.any(mask):
        return 0.0
    masked_e = y[mask] - yd[mask]
    masked_yd = yd[mask]
    idx = int(np.argmax(masked_e))
    peak_e = float(masked_e[idx])
    ref_at_peak = float(masked_yd[idx])
    if abs(ref_at_peak) < 1e-9:
        return 0.0
    return max(0.0, peak_e / abs(ref_at_peak) * 100.0)



def make_summary_table(
    cases: Iterable[DampingCase],
    omega_n: float,
    step_noaw: Dict[float, Dict[str, np.ndarray]],
    step_aw: Dict[float, Dict[str, np.ndarray]],
    sine_noaw: Dict[float, Dict[str, np.ndarray]],
    sine_aw: Dict[float, Dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    csv_path = output_dir / "ch4_damping_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal", "alpha", "beta", "m1", "m2", "zeta_d", "zeta_sim_noaw", "zeta_sim_aw"])
        for signal_name, noaw_map, aw_map in [("y_d1", step_noaw, step_aw), ("y_d2", sine_noaw, sine_aw)]:
            for case in cases:
                alpha = case.zeta * omega_n
                beta = 0.0 if case.zeta >= 1.0 else omega_n * math.sqrt(max(0.0, 1.0 - case.zeta * case.zeta))
                zeta_noaw = equivalent_damping_ratio(transient_overshoot_from_trace(noaw_map[case.zeta]["t"], noaw_map[case.zeta]["y"], noaw_map[case.zeta]["yd"]))
                zeta_aw = equivalent_damping_ratio(transient_overshoot_from_trace(aw_map[case.zeta]["t"], aw_map[case.zeta]["y"], aw_map[case.zeta]["yd"]))
                writer.writerow([signal_name, f"{alpha:.3f}", f"{beta:.3f}", f"{omega_n**2:.3f}", f"{2*alpha:.3f}", f"{case.zeta:.3f}", f"{zeta_noaw:.3f}", f"{zeta_aw:.3f}"])




def plot_family(
    filename: str,
    result_map: Dict[float, Dict[str, np.ndarray]],
    key: str,
    ylabel: str,
    zoom_xlim: Tuple[float, float],
    output_dir: Path,
    y_strategy: str = "full",
    focus_quantile: float = 0.70,
) -> None:
    fig, ax = plt.subplots(figsize=SIM_FIGSIZE, constrained_layout=True)
    cases = [
        DampingCase(1.000, "#0000FF", "-", r"$\zeta = 1.000$"),
        DampingCase(1.0 / math.sqrt(2.0), "#FF0000", "--", r"$\zeta = 0.707$"),
        DampingCase(0.625, "#00CC00", "-.", r"$\zeta = 0.625$"),
    ]
    t = next(iter(result_map.values()))["t"]
    ref_style = {"color": "black", "linewidth": 2.0, "label": r"参考信号 $y_d$"}
    if key == "y":
        ax.plot(t, next(iter(result_map.values()))["yd"], **ref_style)
    if key == "u":
        ax.axhline(U_MAX, color="#555555", linestyle=":", linewidth=1.4)
        ax.axhline(-U_MAX, color="#555555", linestyle=":", linewidth=1.4)
    plotted = []
    for case in cases:
        style = {"color": case.color, "linestyle": case.linestyle, "linewidth": 2.0, "label": case.label}
        ax.plot(t, result_map[case.zeta][key], **style)
        plotted.append((result_map[case.zeta][key], style))
    style_axes(ax, "时间 (s)", ylabel)
    ax.set_xlim(0.0, float(t[-1]))
    leg = ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="0.35")
    leg.get_frame().set_linewidth(0.8)
    add_zoom_inset(
        ax,
        t,
        plotted,
        zoom_xlim,
        ref_series=(next(iter(result_map.values()))["yd"], ref_style) if key == "y" else None,
        y_strategy=y_strategy,
        focus_quantile=focus_quantile,
    )
    save_figure(fig, output_dir / filename)


def build_figures(output_dir: Path, duration: float = SINE_PERIOD, dt: float = 0.001, omega_n: float = 35.0) -> None:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        DampingCase(1.000, "#0000FF", "-", r"$\zeta = 1.000$"),
        DampingCase(1.0 / math.sqrt(2.0), "#FF0000", "--", r"$\zeta = 0.707$"),
        DampingCase(0.625, "#00CC00", "-.", r"$\zeta = 0.625$"),
    ]

    step_noaw = {case.zeta: simulate_saturation_case(case.zeta, omega_n, step_reference, duration, dt, False) for case in cases}
    step_aw = {case.zeta: simulate_saturation_case(case.zeta, omega_n, step_reference, duration, dt, True) for case in cases}
    sine_noaw = {case.zeta: simulate_saturation_case(case.zeta, omega_n, sine_reference, duration, dt, False) for case in cases}
    sine_aw = {case.zeta: simulate_saturation_case(case.zeta, omega_n, sine_reference, duration, dt, True) for case in cases}

    make_summary_table(cases, omega_n, step_noaw, step_aw, sine_noaw, sine_aw, output_dir)

    plot_family("ch4_step_noaw_response", step_noaw, "y", "跟踪输出", (0.48, 0.75), output_dir, y_strategy="upper_band", focus_quantile=0.05)
    plot_family("ch4_step_aw_response", step_aw, "y", "跟踪输出", (0.48, 0.75), output_dir, y_strategy="upper_band", focus_quantile=0.05)
    plot_family("ch4_step_noaw_error", step_noaw, "e", "跟踪误差", (0.48, 0.75), output_dir, y_strategy="upper_band", focus_quantile=0.05)
    plot_family("ch4_step_aw_error", step_aw, "e", "跟踪误差", (0.48, 0.75), output_dir, y_strategy="upper_band", focus_quantile=0.05)
    plot_family("ch4_step_noaw_control", step_noaw, "u", "控制输入", (0.48, 0.75), output_dir, y_strategy="upper_band", focus_quantile=0.05)
    plot_family("ch4_step_aw_control", step_aw, "u", "控制输入", (0.48, 0.75), output_dir, y_strategy="upper_band", focus_quantile=0.05)

    plot_family("ch4_sine_noaw_response", sine_noaw, "y", "跟踪输出", (0.48, 0.75), output_dir, y_strategy="upper_band", focus_quantile=0.05)
    plot_family("ch4_sine_aw_response", sine_aw, "y", "跟踪输出", (0.48, 0.75), output_dir, y_strategy="upper_band", focus_quantile=0.05)
    plot_family("ch4_sine_noaw_error", sine_noaw, "e", "跟踪误差", (0.48, 0.75), output_dir, y_strategy="upper_band", focus_quantile=0.05)
    plot_family("ch4_sine_aw_error", sine_aw, "e", "跟踪误差", (0.48, 0.75), output_dir, y_strategy="upper_band", focus_quantile=0.05)
    plot_family("ch4_sine_noaw_control", sine_noaw, "u", "控制输入", (0.48, 0.75), output_dir, y_strategy="upper_band", focus_quantile=0.05)
    plot_family("ch4_sine_aw_control", sine_aw, "u", "控制输入", (0.48, 0.75), output_dir, y_strategy="upper_band", focus_quantile=0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chapter 4 simulation figures.")
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--duration", type=float, default=SINE_PERIOD)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--omega-n", type=float, default=35.0)
    args = parser.parse_args()
    build_figures(args.output_dir, args.duration, args.dt, args.omega_n)


if __name__ == "__main__":
    main()
