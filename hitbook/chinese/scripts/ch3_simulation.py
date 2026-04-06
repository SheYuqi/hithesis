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
    add_zoom_inset,
    basis_vector,
    configure_matplotlib,
    rk4_step,
    save_figure,
    sine_reference,
    step_reference,
    style_axes,
    true_nonlinearity,
)

import matplotlib.pyplot as plt


FIG_DIR = ROOT / "hitbook" / "chinese" / "figures" / "ch3_sim"
W_NOM = np.array([0.35, -0.28, 0.40, 0.08, 0.12, 0.05], dtype=float)
FILTER_ZETA = 0.90
FILTER_WN = 110.0
ROBUST_GAIN = 320.0
SAT_WIDTH = 0.12


def sat(x: float) -> float:
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return x


def disturbance_profile(t: float) -> float:
    if t < 1.5:
        return 0.0
    tau = t - 1.5
    return 850.0 * math.exp(-1.25 * tau) * math.sin(8.8 * tau) + 48.0 * math.sin(2.7 * t)


def simulate_nn_case(
    zeta: float,
    omega_n: float,
    reference: Callable[[float], Tuple[float, float, float]],
    duration: float = 5.0,
    dt: float = 0.001,
    gamma: float = 0.006,
    sigma_mod: float = 0.045,
    init_scale: float = 0.0,
) -> Dict[str, np.ndarray]:
    times = np.arange(0.0, duration + dt, dt)
    state = np.zeros((times.size, 8), dtype=float)
    state[0, 2:] = init_scale * W_NOM
    control = np.zeros(times.size, dtype=float)
    yd = np.zeros(times.size, dtype=float)
    m1 = omega_n**2
    m2 = 2.0 * zeta * omega_n

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        x1, x2 = x[:2]
        weights = x[2:]
        ref, ref_dot, ref_ddot = reference(t)
        e1 = x1 - ref
        e2 = x2 - ref_dot
        z1 = e1
        z2 = e2 + K1_GAIN * e1
        basis = basis_vector(x1, x2)
        h_est = float(np.dot(weights, basis))
        u = (ref_ddot - 0.55 * h_est - m2 * z2 - (m1 - K1_GAIN * m2) * z1) / B_GAIN
        dx1 = x2
        dx2 = B_GAIN * u + true_nonlinearity(x1, x2) + disturbance_profile(t)
        dweights = gamma * (z2 * basis - sigma_mod * weights)
        return np.concatenate(([dx1, dx2], dweights))

    for idx, t in enumerate(times[:-1]):
        ref, ref_dot, ref_ddot = reference(t)
        yd[idx] = ref
        x1, x2 = state[idx, :2]
        weights = state[idx, 2:]
        e1 = x1 - ref
        e2 = x2 - ref_dot
        z1 = e1
        z2 = e2 + K1_GAIN * e1
        control[idx] = (
            ref_ddot
            - 0.55 * float(np.dot(weights, basis_vector(x1, x2)))
            - m2 * z2
            - (m1 - K1_GAIN * m2) * z1
        ) / B_GAIN
        state[idx + 1] = rk4_step(rhs, t, state[idx], dt)

    ref, ref_dot, ref_ddot = reference(times[-1])
    yd[-1] = ref
    x1, x2 = state[-1, :2]
    weights = state[-1, 2:]
    e1 = x1 - ref
    e2 = x2 - ref_dot
    z1 = e1
    z2 = e2 + K1_GAIN * e1
    control[-1] = (
        ref_ddot
        - 0.55 * float(np.dot(weights, basis_vector(x1, x2)))
        - m2 * z2
        - (m1 - K1_GAIN * m2) * z1
    ) / B_GAIN

    return {"t": times, "y": state[:, 0], "e": state[:, 0] - yd, "u": control, "yd": yd}


def simulate_cf_case(
    zeta: float,
    omega_n: float,
    reference: Callable[[float], Tuple[float, float, float]],
    duration: float = 5.0,
    dt: float = 0.001,
) -> Dict[str, np.ndarray]:
    times = np.arange(0.0, duration + dt, dt)
    state = np.zeros((times.size, 4), dtype=float)  # x1, x2, xi1, xi2
    control = np.zeros(times.size, dtype=float)
    yd = np.zeros(times.size, dtype=float)
    m1 = omega_n**2
    m2 = 2.0 * zeta * omega_n

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        x1, x2, xi1, xi2 = x
        ref, ref_dot, _ = reference(t)
        z1 = x1 - ref
        alpha10 = ref_dot - K1_GAIN * z1
        z2 = x2 - xi1
        s = z2 + (m2 - K1_GAIN) * z1
        h_est = float(np.dot(W_NOM, basis_vector(x1, x2)))
        u = -(
            (m2 - K1_GAIN) * z2
            + (m1 - K1_GAIN * (m2 - K1_GAIN)) * z1
            + h_est
            - xi2
            + ROBUST_GAIN * sat(s / SAT_WIDTH)
        ) / B_GAIN
        dx1 = x2
        dx2 = B_GAIN * u + true_nonlinearity(x1, x2) + disturbance_profile(t)
        dxi1 = xi2
        dxi2 = -2.0 * FILTER_ZETA * FILTER_WN * xi2 - FILTER_WN**2 * (xi1 - alpha10)
        return np.array([dx1, dx2, dxi1, dxi2], dtype=float)

    for idx, t in enumerate(times[:-1]):
        ref, ref_dot, _ = reference(t)
        yd[idx] = ref
        x1, x2, xi1, xi2 = state[idx]
        z1 = x1 - ref
        z2 = x2 - xi1
        s = z2 + (m2 - K1_GAIN) * z1
        h_est = float(np.dot(W_NOM, basis_vector(x1, x2)))
        control[idx] = -(
            (m2 - K1_GAIN) * z2
            + (m1 - K1_GAIN * (m2 - K1_GAIN)) * z1
            + h_est
            - xi2
            + ROBUST_GAIN * sat(s / SAT_WIDTH)
        ) / B_GAIN
        state[idx + 1] = rk4_step(rhs, t, state[idx], dt)

    ref, _, _ = reference(times[-1])
    yd[-1] = ref
    x1, x2, xi1, xi2 = state[-1]
    z1 = x1 - ref
    z2 = x2 - xi1
    s = z2 + (m2 - K1_GAIN) * z1
    h_est = float(np.dot(W_NOM, basis_vector(x1, x2)))
    control[-1] = -(
        (m2 - K1_GAIN) * z2
        + (m1 - K1_GAIN * (m2 - K1_GAIN)) * z1
        + h_est
        - xi2
        + ROBUST_GAIN * sat(s / SAT_WIDTH)
    ) / B_GAIN

    return {"t": times, "y": state[:, 0], "e": state[:, 0] - yd, "u": control, "yd": yd}


def disturbance_metrics(
    t: np.ndarray,
    e: np.ndarray,
    t_disturb: float = 1.5,
    settle_band: float = 0.08,
    settle_hold: float = 0.20,
) -> Tuple[float, float, float]:
    mask = t >= t_disturb
    tt = t[mask]
    ee = e[mask]
    peak = float(np.max(np.abs(ee)))
    rmse = float(np.sqrt(np.mean(ee * ee)))
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.001
    hold_steps = max(1, int(settle_hold / dt))
    rec_time = tt[-1] - t_disturb
    for i in range(len(ee) - hold_steps):
        window = ee[i : i + hold_steps]
        if np.all(np.abs(window) <= settle_band):
            rec_time = float(tt[i] - t_disturb)
            break
    return peak, rec_time, rmse


def make_summary_table(
    cases: Iterable[DampingCase],
    omega_n: float,
    step_nn: Dict[float, Dict[str, np.ndarray]],
    step_cf: Dict[float, Dict[str, np.ndarray]],
    sine_nn: Dict[float, Dict[str, np.ndarray]],
    sine_cf: Dict[float, Dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    csv_path = output_dir / "ch3_damping_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "signal",
                "alpha",
                "beta",
                "m1",
                "m2",
                "zeta_d",
                "peak_nn",
                "peak_cf",
                "trec_nn",
                "trec_cf",
                "rmse_nn",
                "rmse_cf",
            ]
        )
        for signal_name, nn_map, cf_map in [("y_d1", step_nn, step_cf), ("y_d2", sine_nn, sine_cf)]:
            for case in cases:
                alpha = case.zeta * omega_n
                beta = 0.0 if case.zeta >= 1.0 else omega_n * math.sqrt(max(0.0, 1.0 - case.zeta * case.zeta))
                peak_nn, trec_nn, rmse_nn = disturbance_metrics(nn_map[case.zeta]["t"], nn_map[case.zeta]["e"])
                peak_cf, trec_cf, rmse_cf = disturbance_metrics(cf_map[case.zeta]["t"], cf_map[case.zeta]["e"])
                writer.writerow(
                    [
                        signal_name,
                        f"{alpha:.3f}",
                        f"{beta:.3f}",
                        f"{omega_n**2:.3f}",
                        f"{2 * alpha:.3f}",
                        f"{case.zeta:.3f}",
                        f"{peak_nn:.3f}",
                        f"{peak_cf:.3f}",
                        f"{trec_nn:.3f}",
                        f"{trec_cf:.3f}",
                        f"{rmse_nn:.3f}",
                        f"{rmse_cf:.3f}",
                    ]
                )


def plot_family(
    filename: str,
    result_map: Dict[float, Dict[str, np.ndarray]],
    key: str,
    ylabel: str,
    zoom_xlim: Tuple[float, float],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.9), constrained_layout=True)
    cases = [
        DampingCase(1.000, "#0000FF", "-", r"$\zeta = 1.000$"),
        DampingCase(1.0 / math.sqrt(2.0), "#FF0000", "--", r"$\zeta = 0.707$"),
        DampingCase(0.625, "#00CC00", "-.", r"$\zeta = 0.625$"),
    ]
    first = next(iter(result_map.values()))
    t = first["t"]
    ref_style = {"color": "black", "linewidth": 2.0, "label": r"参考信号 $y_d$"}
    plotted = []
    if key == "y":
        ax.plot(t, first["yd"], **ref_style)
    for case in cases:
        style = {
            "color": case.color,
            "linestyle": case.linestyle,
            "linewidth": 2.0,
            "label": case.label,
        }
        ax.plot(t, result_map[case.zeta][key], **style)
        plotted.append((result_map[case.zeta][key], style))
    style_axes(ax, "时间 (s)", ylabel)
    ax.set_xlim(0.0, 5.0)
    leg = ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="0.35")
    leg.get_frame().set_linewidth(0.8)
    add_zoom_inset(ax, t, plotted, zoom_xlim, ref_series=(first["yd"], ref_style) if key == "y" else None)
    save_figure(fig, output_dir / filename)


def build_figures(output_dir: Path, duration: float = 5.0, dt: float = 0.001, omega_n: float = 35.0) -> None:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        DampingCase(1.000, "#0000FF", "-", r"$\zeta = 1.000$"),
        DampingCase(1.0 / math.sqrt(2.0), "#FF0000", "--", r"$\zeta = 0.707$"),
        DampingCase(0.625, "#00CC00", "-.", r"$\zeta = 0.625$"),
    ]

    step_nn = {case.zeta: simulate_nn_case(case.zeta, omega_n, step_reference, duration, dt) for case in cases}
    step_cf = {case.zeta: simulate_cf_case(case.zeta, omega_n, step_reference, duration, dt) for case in cases}
    sine_nn = {case.zeta: simulate_nn_case(case.zeta, omega_n, sine_reference, duration, dt) for case in cases}
    sine_cf = {case.zeta: simulate_cf_case(case.zeta, omega_n, sine_reference, duration, dt) for case in cases}

    make_summary_table(cases, omega_n, step_nn, step_cf, sine_nn, sine_cf, output_dir)

    plot_family("ch3_step_nn_response", step_nn, "y", "跟踪输出", (1.45, 2.05), output_dir)
    plot_family("ch3_step_cf_response", step_cf, "y", "跟踪输出", (1.45, 2.05), output_dir)
    plot_family("ch3_step_nn_error", step_nn, "e", "跟踪误差", (1.45, 2.05), output_dir)
    plot_family("ch3_step_cf_error", step_cf, "e", "跟踪误差", (1.45, 2.05), output_dir)
    plot_family("ch3_step_nn_control", step_nn, "u", "控制输入", (1.45, 2.05), output_dir)
    plot_family("ch3_step_cf_control", step_cf, "u", "控制输入", (1.45, 2.05), output_dir)

    plot_family("ch3_sine_nn_response", sine_nn, "y", "跟踪输出", (1.45, 2.05), output_dir)
    plot_family("ch3_sine_cf_response", sine_cf, "y", "跟踪输出", (1.45, 2.05), output_dir)
    plot_family("ch3_sine_nn_error", sine_nn, "e", "跟踪误差", (1.45, 2.05), output_dir)
    plot_family("ch3_sine_cf_error", sine_cf, "e", "跟踪误差", (1.45, 2.05), output_dir)
    plot_family("ch3_sine_nn_control", sine_nn, "u", "控制输入", (1.45, 2.05), output_dir)
    plot_family("ch3_sine_cf_control", sine_cf, "u", "控制输入", (1.45, 2.05), output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chapter 3 simulation figures.")
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--omega-n", type=float, default=35.0)
    args = parser.parse_args()
    build_figures(args.output_dir, args.duration, args.dt, args.omega_n)


if __name__ == "__main__":
    main()
