#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[3]
PYDEPS = ROOT / ".pydeps"
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


FIG_DIR = ROOT / "hitbook" / "chinese" / "figures" / "ch2_sim"
B_GAIN = 8000.0
K1_GAIN = 5.0
W_STAR = np.array([0.35, -0.28, 0.40, 0.08, 0.12, 0.05], dtype=float)


@dataclass(frozen=True)
class DampingCase:
    zeta: float
    color: str
    linestyle: str
    label: str


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Noto Serif CJK JP",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
        }
    )


def step_reference(t: float, t0: float = 0.5, amplitude: float = 5.0) -> Tuple[float, float, float]:
    if t < t0:
        return 0.0, 0.0, 0.0
    return amplitude, 0.0, 0.0


def sine_reference(
    t: float,
    t0: float = 0.5,
    offset: float = 5.0,
    amplitude: float = 1.2,
    freq_hz: float = 0.35,
) -> Tuple[float, float, float]:
    if t < t0:
        return 0.0, 0.0, 0.0
    omega = 2.0 * math.pi * freq_hz
    tau = t - t0
    yd = offset + amplitude * math.sin(omega * tau)
    yd_dot = amplitude * omega * math.cos(omega * tau)
    yd_ddot = -amplitude * omega * omega * math.sin(omega * tau)
    return yd, yd_dot, yd_ddot


def rk4_step(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    state: np.ndarray,
    dt: float,
) -> np.ndarray:
    k1 = rhs(t, state)
    k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = rhs(t + dt, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_ideal_closed_loop(
    zeta: float,
    omega_n: float,
    reference: Callable[[float], Tuple[float, float, float]],
    duration: float = 5.0,
    dt: float = 0.001,
) -> Dict[str, np.ndarray]:
    times = np.arange(0.0, duration + dt, dt)
    state = np.zeros((times.size, 2), dtype=float)
    control = np.zeros(times.size, dtype=float)
    yd = np.zeros(times.size, dtype=float)
    yd_dot = np.zeros(times.size, dtype=float)

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        ref, ref_dot, ref_ddot = reference(t)
        e1 = x[0] - ref
        e2 = x[1] - ref_dot
        u = (ref_ddot - 2.0 * zeta * omega_n * e2 - (omega_n**2) * e1) / B_GAIN
        return np.array([x[1], B_GAIN * u], dtype=float)

    for idx, t in enumerate(times[:-1]):
        ref, ref_dot, ref_ddot = reference(t)
        yd[idx] = ref
        yd_dot[idx] = ref_dot
        e1 = state[idx, 0] - ref
        e2 = state[idx, 1] - ref_dot
        control[idx] = (ref_ddot - 2.0 * zeta * omega_n * e2 - (omega_n**2) * e1) / B_GAIN
        state[idx + 1] = rk4_step(rhs, t, state[idx], dt)

    yd[-1], yd_dot[-1], ref_ddot = reference(times[-1])
    e1 = state[-1, 0] - yd[-1]
    e2 = state[-1, 1] - yd_dot[-1]
    control[-1] = (ref_ddot - 2.0 * zeta * omega_n * e2 - (omega_n**2) * e1) / B_GAIN

    return {
        "t": times,
        "y": state[:, 0],
        "y_dot": state[:, 1],
        "yd": yd,
        "yd_dot": yd_dot,
        "e": state[:, 0] - yd,
        "u": control,
    }


def overshoot_percent(zeta: float) -> float:
    if zeta >= 1.0:
        return 0.0
    return math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta * zeta)) * 100.0


def equivalent_damping_ratio(sigma_percent: float) -> float:
    if sigma_percent <= 0.1:
        return 1.0
    sigma = sigma_percent / 100.0
    log_term = abs(math.log(sigma))
    return log_term / math.sqrt(math.pi * math.pi + log_term * log_term)


def basis_vector(x1: float, x2: float) -> np.ndarray:
    return np.array(
        [
            x1 / 5.0,
            x2 / 5.0,
            math.sin(math.radians(x1)),
            (x1 * x2) / 25.0,
            math.tanh(x2 / 3.0),
            1.0,
        ],
        dtype=float,
    )


def true_nonlinearity(x1: float, x2: float) -> float:
    return float(np.dot(W_STAR, basis_vector(x1, x2)))


def simulate_nn_closed_loop(
    zeta: float,
    omega_n: float,
    reference: Callable[[float], Tuple[float, float, float]],
    duration: float = 5.0,
    dt: float = 0.001,
    gamma: float = 0.04,
    sigma_mod: float = 0.01,
    init_scale: float = 0.85,
) -> Dict[str, np.ndarray]:
    times = np.arange(0.0, duration + dt, dt)
    state = np.zeros((times.size, 8), dtype=float)
    state[0, 2:] = init_scale * W_STAR
    control = np.zeros(times.size, dtype=float)
    yd = np.zeros(times.size, dtype=float)
    yd_dot = np.zeros(times.size, dtype=float)
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
        u = (ref_ddot - h_est - m2 * z2 - (m1 - K1_GAIN * m2) * z1) / B_GAIN
        dx1 = x2
        dx2 = B_GAIN * u + true_nonlinearity(x1, x2)
        dweights = gamma * (z2 * basis - sigma_mod * weights)
        return np.concatenate(([dx1, dx2], dweights))

    for idx, t in enumerate(times[:-1]):
        ref, ref_dot, ref_ddot = reference(t)
        yd[idx] = ref
        yd_dot[idx] = ref_dot
        x1, x2 = state[idx, :2]
        weights = state[idx, 2:]
        e1 = x1 - ref
        e2 = x2 - ref_dot
        z1 = e1
        z2 = e2 + K1_GAIN * e1
        control[idx] = (
            ref_ddot
            - float(np.dot(weights, basis_vector(x1, x2)))
            - m2 * z2
            - (m1 - K1_GAIN * m2) * z1
        ) / B_GAIN
        state[idx + 1] = rk4_step(rhs, t, state[idx], dt)

    yd[-1], yd_dot[-1], ref_ddot = reference(times[-1])
    x1, x2 = state[-1, :2]
    weights = state[-1, 2:]
    e1 = x1 - yd[-1]
    e2 = x2 - yd_dot[-1]
    z1 = e1
    z2 = e2 + K1_GAIN * e1
    control[-1] = (
        ref_ddot
        - float(np.dot(weights, basis_vector(x1, x2)))
        - m2 * z2
        - (m1 - K1_GAIN * m2) * z1
    ) / B_GAIN

    return {
        "t": times,
        "y": state[:, 0],
        "y_dot": state[:, 1],
        "yd": yd,
        "yd_dot": yd_dot,
        "e": state[:, 0] - yd,
        "u": control,
        "weights": state[:, 2:],
    }


def simulate_standard_backstepping_nn(
    reference: Callable[[float], Tuple[float, float, float]],
    duration: float = 5.0,
    dt: float = 0.001,
    k1: float = 25.0,
    k2: float = 25.0,
    gamma: float = 0.04,
    sigma_mod: float = 0.01,
    init_scale: float = 0.85,
) -> Dict[str, np.ndarray]:
    times = np.arange(0.0, duration + dt, dt)
    state = np.zeros((times.size, 8), dtype=float)
    state[0, 2:] = init_scale * W_STAR
    control = np.zeros(times.size, dtype=float)
    yd = np.zeros(times.size, dtype=float)
    yd_dot = np.zeros(times.size, dtype=float)

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        x1, x2 = x[:2]
        weights = x[2:]
        ref, ref_dot, ref_ddot = reference(t)
        e1 = x1 - ref
        e2 = x2 - ref_dot
        alpha1 = ref_dot - k1 * e1
        z2 = x2 - alpha1
        basis = basis_vector(x1, x2)
        h_est = float(np.dot(weights, basis))
        u = (ref_ddot - k1 * e2 - e1 - k2 * z2 - h_est) / B_GAIN
        dx1 = x2
        dx2 = B_GAIN * u + true_nonlinearity(x1, x2)
        dweights = gamma * (z2 * basis - sigma_mod * weights)
        return np.concatenate(([dx1, dx2], dweights))

    for idx, t in enumerate(times[:-1]):
        ref, ref_dot, ref_ddot = reference(t)
        yd[idx] = ref
        yd_dot[idx] = ref_dot
        x1, x2 = state[idx, :2]
        weights = state[idx, 2:]
        e1 = x1 - ref
        e2 = x2 - ref_dot
        alpha1 = ref_dot - k1 * e1
        z2 = x2 - alpha1
        control[idx] = (ref_ddot - k1 * e2 - e1 - k2 * z2 - float(np.dot(weights, basis_vector(x1, x2)))) / B_GAIN
        state[idx + 1] = rk4_step(rhs, t, state[idx], dt)

    yd[-1], yd_dot[-1], ref_ddot = reference(times[-1])
    x1, x2 = state[-1, :2]
    weights = state[-1, 2:]
    e1 = x1 - yd[-1]
    e2 = x2 - yd_dot[-1]
    alpha1 = yd_dot[-1] - k1 * e1
    z2 = x2 - alpha1
    control[-1] = (ref_ddot - k1 * e2 - e1 - k2 * z2 - float(np.dot(weights, basis_vector(x1, x2)))) / B_GAIN

    return {
        "t": times,
        "y": state[:, 0],
        "y_dot": state[:, 1],
        "yd": yd,
        "yd_dot": yd_dot,
        "e": state[:, 0] - yd,
        "u": control,
        "weights": state[:, 2:],
    }


def step_overshoot_from_trace(t: np.ndarray, y: np.ndarray, yd: np.ndarray, t0: float = 0.5) -> float:
    mask = t >= t0
    if not np.any(mask):
        return 0.0
    steady = yd[mask][-1]
    if abs(steady) < 1e-9:
        return 0.0
    peak = float(np.max(y[mask]))
    return max(0.0, (peak - steady) / steady * 100.0)


def transient_overshoot_from_trace(
    t: np.ndarray,
    y: np.ndarray,
    yd: np.ndarray,
    t0: float = 0.5,
    window: float = 0.35,
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


def step_peak_metrics(
    t: np.ndarray,
    y: np.ndarray,
    yd: np.ndarray,
    t0: float = 0.5,
) -> Tuple[float, float, float]:
    mask = t >= t0
    masked_t = t[mask]
    masked_y = y[mask]
    idx = int(np.argmax(masked_y))
    peak_t = float(masked_t[idx])
    peak_y = float(masked_y[idx])
    steady = float(yd[mask][-1])
    if abs(steady) < 1e-9:
        sigma = 0.0
    else:
        sigma = max(0.0, (peak_y - steady) / steady * 100.0)
    return peak_t, peak_y, sigma


def make_summary_table(
    cases: Iterable[DampingCase],
    omega_n: float,
    step_results: Dict[float, Dict[str, np.ndarray]],
    sine_results: Dict[float, Dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ch2_damping_summary.csv"
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
                "zeta_sim",
            ]
        )
        for signal_name, result_map in [("y_d1", step_results), ("y_d2", sine_results)]:
            for case in cases:
                result = result_map[case.zeta]
                alpha = case.zeta * omega_n
                beta = 0.0 if case.zeta >= 1.0 else omega_n * math.sqrt(max(0.0, 1.0 - case.zeta * case.zeta))
                sigma_sim = transient_overshoot_from_trace(result["t"], result["y"], result["yd"])
                writer.writerow(
                    [
                        signal_name,
                        f"{alpha:.3f}",
                        f"{beta:.3f}",
                        f"{omega_n**2:.3f}",
                        f"{2.0 * alpha:.3f}",
                        f"{case.zeta:.3f}",
                        f"{equivalent_damping_ratio(sigma_sim):.3f}",
                    ]
                )


def add_zoom_inset(
    ax: plt.Axes,
    t: np.ndarray,
    series: List[Tuple[np.ndarray, Dict[str, object]]],
    xlim: Tuple[float, float],
    ylabel: str | None = None,
    ref_series: Tuple[np.ndarray, Dict[str, object]] | None = None,
) -> plt.Axes:
    inset = inset_axes(ax, width="55%", height="45%", loc="center left", borderpad=1.6)
    if ref_series is not None:
        inset.plot(t, ref_series[0], **ref_series[1])
    for values, style in series:
        inset.plot(t, values, **style)
    inset.set_xlim(*xlim)

    mask = (t >= xlim[0]) & (t <= xlim[1])
    stacked = []
    if ref_series is not None:
        stacked.append(ref_series[0][mask])
    for values, _ in series:
        stacked.append(values[mask])
    y_min = min(np.min(values) for values in stacked)
    y_max = max(np.max(values) for values in stacked)
    pad = 0.08 * max(1e-6, y_max - y_min)
    inset.set_ylim(y_min - pad, y_max + pad)
    inset.grid(True, alpha=0.35)
    inset.tick_params(direction="in")
    if ylabel:
        inset.set_ylabel(ylabel, fontsize=8)
    inset.set_xlabel("时间 (s)", fontsize=8)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="black", lw=0.8)
    return inset


def annotate_step_overshoot(ax: plt.Axes, result_map: Dict[float, Dict[str, np.ndarray]], cases: Iterable[DampingCase]) -> None:
    offsets = {
        1.000: (10, 8),
        0.707: (-58, -10),
        0.625: (12, -14),
    }
    for case in cases:
        result = result_map[case.zeta]
        peak_t, peak_y, sigma = step_peak_metrics(result["t"], result["y"], result["yd"])
        dx, dy = offsets.get(round(case.zeta, 3), (10, 6))
        ax.plot([peak_t], [peak_y], marker="s", markersize=3.8, color=case.color)
        ax.annotate(
            rf"$\sigma={sigma:.2f}\%$",
            xy=(peak_t, peak_y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            color=case.color,
            arrowprops={"arrowstyle": "->", "color": case.color, "lw": 0.8},
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": case.color, "linewidth": 0.6},
        )


def style_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    ax.tick_params(direction="in", which="both", top=True, right=True)


def save_figure(fig: plt.Figure, basepath: Path) -> None:
    fig.savefig(basepath.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(basepath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_figures(
    output_dir: Path,
    duration: float = 5.0,
    dt: float = 0.001,
    omega_n: float = 35.0,
    mode: str = "nn",
) -> None:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        DampingCase(1.000, "#1f4aff", "-", r"$\zeta = 1.000$"),
        DampingCase(1.0 / math.sqrt(2.0), "#d62728", "--", r"$\zeta = 0.707$"),
        DampingCase(0.625, "#2ca02c", "-.", r"$\zeta = 0.625$"),
    ]

    simulator = simulate_nn_closed_loop if mode == "nn" else simulate_ideal_closed_loop
    step_results = {case.zeta: simulator(case.zeta, omega_n, step_reference, duration, dt) for case in cases}
    sine_results = {case.zeta: simulator(case.zeta, omega_n, sine_reference, duration, dt) for case in cases}
    bs_step = simulate_standard_backstepping_nn(step_reference, duration, dt)
    bs_sine = simulate_standard_backstepping_nn(sine_reference, duration, dt)

    make_summary_table(cases, omega_n, step_results, sine_results, output_dir)

    common_styles = {
        case.zeta: {
            "color": case.color,
            "linestyle": case.linestyle,
            "linewidth": 1.8,
            "label": case.label,
        }
        for case in cases
    }
    ref_style = {"color": "black", "linewidth": 1.4, "label": r"参考信号 $y_d$"}
    bs_style = {"color": "#4d4d4d", "linestyle": ":", "linewidth": 2.0, "label": "BS"}

    panels = [
        (
            "ch2_step_response",
            step_results,
            "y",
            "跟踪输出",
            (0.50, 0.72),
            "跟踪常值信号的跟踪响应",
        ),
        (
            "ch2_step_error",
            step_results,
            "e",
            "跟踪误差",
            (0.50, 0.72),
            "跟踪常值信号的跟踪误差",
        ),
        (
            "ch2_step_control",
            step_results,
            "u",
            "控制输入",
            (0.50, 0.72),
            "跟踪常值信号的控制输入",
        ),
        (
            "ch2_sine_response",
            sine_results,
            "y",
            "跟踪输出",
            (0.50, 0.65),
            "跟踪变值信号的跟踪响应",
        ),
        (
            "ch2_sine_error",
            sine_results,
            "e",
            "跟踪误差",
            (0.50, 0.65),
            "跟踪变值信号的跟踪误差",
        ),
        (
            "ch2_sine_control",
            sine_results,
            "u",
            "控制输入",
            (0.50, 0.65),
            "跟踪变值信号的控制输入",
        ),
    ]

    for filename, result_map, key, ylabel, zoom_xlim, title in panels:
        fig, ax = plt.subplots(figsize=(6.4, 3.9), constrained_layout=True)
        first = next(iter(result_map.values()))
        t = first["t"]
        show_reference = key == "y"
        if show_reference:
            ax.plot(t, first["yd"], **ref_style)
        plotted = []
        ax.plot(t, (bs_step if "step" in filename else bs_sine)[key], **bs_style)
        plotted.append(((bs_step if "step" in filename else bs_sine)[key], bs_style))
        for case in cases:
            data = result_map[case.zeta]
            style = common_styles[case.zeta]
            ax.plot(t, data[key], **style)
            plotted.append((data[key], style))
        style_axes(ax, "时间 (s)", ylabel)
        ax.set_xlim(0.0, duration)
        ax.legend(loc="best", frameon=True, edgecolor="black")
        inset = add_zoom_inset(ax, t, plotted, zoom_xlim, ref_series=(first["yd"], ref_style) if show_reference else None)
        save_figure(fig, output_dir / filename)

    overview_order = [
        "ch2_step_response",
        "ch2_step_error",
        "ch2_step_control",
        "ch2_sine_response",
        "ch2_sine_error",
        "ch2_sine_control",
    ]
    title_map = {item[0]: item[5] for item in panels}
    result_key_map = {item[0]: item[2] for item in panels}
    result_map_lookup = {
        "ch2_step_response": step_results,
        "ch2_step_error": step_results,
        "ch2_step_control": step_results,
        "ch2_sine_response": sine_results,
        "ch2_sine_error": sine_results,
        "ch2_sine_control": sine_results,
    }
    ylabel_map = {item[0]: item[3] for item in panels}
    zoom_map = {item[0]: item[4] for item in panels}

    fig, axes = plt.subplots(3, 2, figsize=(11.0, 12.0), constrained_layout=True)
    axes = axes.reshape(3, 2)
    for ax, name in zip(axes.flat, overview_order):
        results = result_map_lookup[name]
        key = result_key_map[name]
        first = next(iter(results.values()))
        t = first["t"]
        show_reference = key == "y"
        if show_reference:
            ax.plot(t, first["yd"], **ref_style)
        plotted = []
        ax.plot(t, (bs_step if "step" in name else bs_sine)[key], **bs_style)
        plotted.append(((bs_step if "step" in name else bs_sine)[key], bs_style))
        for case in cases:
            data = results[case.zeta]
            style = common_styles[case.zeta]
            ax.plot(t, data[key], **style)
            plotted.append((data[key], style))
        style_axes(ax, "时间 (s)", ylabel_map[name])
        ax.set_xlim(0.0, duration)
        ax.legend(loc="best", frameon=True, edgecolor="black")
        inset = add_zoom_inset(ax, t, plotted, zoom_map[name], ref_series=(first["yd"], ref_style) if show_reference else None)
    save_figure(fig, output_dir / "ch2_damping_overview")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chapter 2 damping-ratio simulation figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIG_DIR,
        help="Directory for generated figures and CSV summary.",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Simulation duration in seconds.")
    parser.add_argument("--dt", type=float, default=0.001, help="Simulation time step.")
    parser.add_argument("--omega-n", type=float, default=35.0, help="Natural frequency used to build m1 and m2.")
    parser.add_argument(
        "--mode",
        choices=("nn", "ideal"),
        default="nn",
        help="nn uses the neural-network controller closed loop; ideal uses the exact second-order template.",
    )
    args = parser.parse_args()
    build_figures(args.output_dir, duration=args.duration, dt=args.dt, omega_n=args.omega_n, mode=args.mode)


if __name__ == "__main__":
    main()
