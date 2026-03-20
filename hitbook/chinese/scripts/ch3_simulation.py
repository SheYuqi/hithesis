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
    equivalent_damping_ratio,
    save_figure,
    sine_reference,
    step_reference,
    style_axes,
    true_nonlinearity,
    rk4_step,
)

import matplotlib.pyplot as plt


FIG_DIR = ROOT / "hitbook" / "chinese" / "figures" / "ch3_sim"
W_CF = np.array([0.35, -0.28, 0.40, 0.08, 0.12, 0.05], dtype=float)
FILTER_ZETA = 0.90
FILTER_WN = 80.0
ROBUST_GAIN = 0.55
SAT_WIDTH = 0.18


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
    return 900.0 * math.exp(-1.2 * tau) * math.sin(8.5 * tau) + 55.0 * math.sin(2.6 * t)


def simulate_command_filter_case(
    zeta: float,
    omega_n: float,
    reference: Callable[[float], Tuple[float, float, float]],
    duration: float = 5.0,
    dt: float = 0.001,
    anti_disturbance: bool = True,
) -> Dict[str, np.ndarray]:
    times = np.arange(0.0, duration + dt, dt)
    state = np.zeros((times.size, 5), dtype=float)  # x1, x2, xi1, xi2, q
    control = np.zeros(times.size, dtype=float)
    yd = np.zeros(times.size, dtype=float)
    m1 = omega_n**2
    m2 = 2.0 * zeta * omega_n
    robust_gain = ROBUST_GAIN if anti_disturbance else 0.0
    weight_scale = 1.0 if anti_disturbance else 0.45
    filter_wn = FILTER_WN if anti_disturbance else 24.0

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        x1, x2, xi1, xi2, q = x
        ref, ref_dot, _ = reference(t)
        z1 = x1 - ref
        alpha10 = ref_dot - K1_GAIN * z1
        z2 = x2 - xi1
        s = z2 + (m2 - K1_GAIN) * z1 + m1 * q
        h_est = float(np.dot(weight_scale * W_CF, basis_vector(x1, x2)))
        u = -(
            (m2 - K1_GAIN) * z2
            + (m1 - K1_GAIN * (m2 - K1_GAIN)) * z1
            + h_est
            - xi2
            + robust_gain * sat(s / SAT_WIDTH)
        ) / B_GAIN
        dx1 = x2
        dx2 = B_GAIN * u + true_nonlinearity(x1, x2) + disturbance_profile(t)
        dxi1 = xi2
        dxi2 = -2.0 * FILTER_ZETA * filter_wn * xi2 - filter_wn**2 * (xi1 - alpha10)
        dq = z1
        return np.array([dx1, dx2, dxi1, dxi2, dq], dtype=float)

    for idx, t in enumerate(times[:-1]):
        ref, ref_dot, _ = reference(t)
        yd[idx] = ref
        x1, x2, xi1, xi2, q = state[idx]
        z1 = x1 - ref
        z2 = x2 - xi1
        s = z2 + (m2 - K1_GAIN) * z1 + m1 * q
        h_est = float(np.dot(weight_scale * W_CF, basis_vector(x1, x2)))
        control[idx] = -(
            (m2 - K1_GAIN) * z2
            + (m1 - K1_GAIN * (m2 - K1_GAIN)) * z1
            + h_est
            - xi2
            + robust_gain * sat(s / SAT_WIDTH)
        ) / B_GAIN
        state[idx + 1] = rk4_step(rhs, t, state[idx], dt)

    ref, _, _ = reference(times[-1])
    yd[-1] = ref
    x1, x2, xi1, xi2, q = state[-1]
    z1 = x1 - ref
    z2 = x2 - xi1
    s = z2 + (m2 - K1_GAIN) * z1 + m1 * q
    h_est = float(np.dot(W_CF, basis_vector(x1, x2)))
    control[-1] = -(
        (m2 - K1_GAIN) * z2
        + (m1 - K1_GAIN * (m2 - K1_GAIN)) * z1
        + h_est
        - xi2
        + robust_gain * sat(s / SAT_WIDTH)
    ) / B_GAIN

    return {"t": times, "y": state[:, 0], "e": state[:, 0] - yd, "u": control, "yd": yd}


def transient_overshoot_from_trace(
    t: np.ndarray,
    y: np.ndarray,
    yd: np.ndarray,
    t0: float = 0.5,
    window: float = 0.6,
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
    step_noad: Dict[float, Dict[str, np.ndarray]],
    step_ad: Dict[float, Dict[str, np.ndarray]],
    sine_noad: Dict[float, Dict[str, np.ndarray]],
    sine_ad: Dict[float, Dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    csv_path = output_dir / "ch3_damping_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signal", "alpha", "beta", "m1", "m2", "zeta_d", "zeta_sim_noad", "zeta_sim_ad"])
        for signal_name, noad_map, ad_map in [("y_d1", step_noad, step_ad), ("y_d2", sine_noad, sine_ad)]:
            for case in cases:
                alpha = case.zeta * omega_n
                beta = 0.0 if case.zeta >= 1.0 else omega_n * math.sqrt(max(0.0, 1.0 - case.zeta * case.zeta))
                zeta_noad = equivalent_damping_ratio(
                    transient_overshoot_from_trace(noad_map[case.zeta]["t"], noad_map[case.zeta]["y"], noad_map[case.zeta]["yd"])
                )
                zeta_ad = equivalent_damping_ratio(
                    transient_overshoot_from_trace(ad_map[case.zeta]["t"], ad_map[case.zeta]["y"], ad_map[case.zeta]["yd"])
                )
                writer.writerow(
                    [
                        signal_name,
                        f"{alpha:.3f}",
                        f"{beta:.3f}",
                        f"{omega_n**2:.3f}",
                        f"{2 * alpha:.3f}",
                        f"{case.zeta:.3f}",
                        f"{zeta_noad:.3f}",
                        f"{zeta_ad:.3f}",
                    ]
                )


def plot_family(
    filename: str,
    result_map: Dict[float, Dict[str, np.ndarray]],
    key: str,
    ylabel: str,
    zoom_xlim: Tuple[float, float],
    output_dir: Path,
    title_prefix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.9), constrained_layout=True)
    cases = [
        DampingCase(1.000, "#1f4aff", "-", r"$\zeta = 1.000$"),
        DampingCase(1.0 / math.sqrt(2.0), "#d62728", "--", r"$\zeta = 0.707$"),
        DampingCase(0.625, "#2ca02c", "-.", r"$\zeta = 0.625$"),
    ]
    first = next(iter(result_map.values()))
    t = first["t"]
    ref_style = {"color": "black", "linewidth": 1.4, "label": r"参考信号 $y_d$"}
    plotted = []
    if key == "y":
        ax.plot(t, first["yd"], **ref_style)
    for case in cases:
        style = {
            "color": case.color,
            "linestyle": case.linestyle,
            "linewidth": 1.8,
            "label": case.label,
        }
        ax.plot(t, result_map[case.zeta][key], **style)
        plotted.append((result_map[case.zeta][key], style))
    style_axes(ax, "时间 (s)", ylabel)
    ax.set_xlim(0.0, 5.0)
    ax.legend(loc="best", frameon=True, edgecolor="black")
    add_zoom_inset(ax, t, plotted, zoom_xlim, ref_series=(first["yd"], ref_style) if key == "y" else None)
    save_figure(fig, output_dir / filename)


def build_figures(output_dir: Path, duration: float = 5.0, dt: float = 0.001, omega_n: float = 35.0) -> None:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        DampingCase(1.000, "#1f4aff", "-", r"$\zeta = 1.000$"),
        DampingCase(1.0 / math.sqrt(2.0), "#d62728", "--", r"$\zeta = 0.707$"),
        DampingCase(0.625, "#2ca02c", "-.", r"$\zeta = 0.625$"),
    ]

    step_noad = {case.zeta: simulate_command_filter_case(case.zeta, omega_n, step_reference, duration, dt, False) for case in cases}
    step_ad = {case.zeta: simulate_command_filter_case(case.zeta, omega_n, step_reference, duration, dt, True) for case in cases}
    sine_noad = {case.zeta: simulate_command_filter_case(case.zeta, omega_n, sine_reference, duration, dt, False) for case in cases}
    sine_ad = {case.zeta: simulate_command_filter_case(case.zeta, omega_n, sine_reference, duration, dt, True) for case in cases}

    make_summary_table(cases, omega_n, step_noad, step_ad, sine_noad, sine_ad, output_dir)

    plot_family("ch3_step_noad_response", step_noad, "y", "跟踪输出", (0.50, 0.80), output_dir, "无抗扰")
    plot_family("ch3_step_ad_response", step_ad, "y", "跟踪输出", (0.50, 0.80), output_dir, "有抗扰")
    plot_family("ch3_step_noad_error", step_noad, "e", "跟踪误差", (0.50, 0.95), output_dir, "无抗扰")
    plot_family("ch3_step_ad_error", step_ad, "e", "跟踪误差", (0.50, 0.95), output_dir, "有抗扰")
    plot_family("ch3_step_noad_control", step_noad, "u", "控制输入", (0.50, 0.95), output_dir, "无抗扰")
    plot_family("ch3_step_ad_control", step_ad, "u", "控制输入", (0.50, 0.95), output_dir, "有抗扰")

    plot_family("ch3_sine_noad_response", sine_noad, "y", "跟踪输出", (0.50, 0.95), output_dir, "无抗扰")
    plot_family("ch3_sine_ad_response", sine_ad, "y", "跟踪输出", (0.50, 0.95), output_dir, "有抗扰")
    plot_family("ch3_sine_noad_error", sine_noad, "e", "跟踪误差", (0.50, 0.95), output_dir, "无抗扰")
    plot_family("ch3_sine_ad_error", sine_ad, "e", "跟踪误差", (0.50, 0.95), output_dir, "有抗扰")
    plot_family("ch3_sine_noad_control", sine_noad, "u", "控制输入", (0.50, 0.95), output_dir, "无抗扰")
    plot_family("ch3_sine_ad_control", sine_ad, "u", "控制输入", (0.50, 0.95), output_dir, "有抗扰")


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
