#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ch2_simulation import SIM_FIGSIZE, configure_matplotlib, save_figure, style_axes


FIG_DIR = ROOT / "hitbook" / "chinese" / "figures" / "ch3_sim"


def benchmark_signal(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.cos(t)
    z_dot = -np.sin(t)
    return z, z_dot


def basis_vector(z: float, eta: float) -> np.ndarray:
    return np.array(
        [
            z,
            eta,
            z * eta,
            math.sin(z),
            math.tanh(eta),
            1.0,
        ],
        dtype=float,
    )


def simulate_second_order_command_filter(
    t: np.ndarray,
    z: np.ndarray,
    h1: float = 20.0,
    h2: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r1 = np.zeros_like(t)
    r2 = np.zeros_like(t)
    y_r = np.zeros_like(t)
    r1[0] = z[0]
    r2[0] = z[0]
    dt = float(t[1] - t[0])

    for k in range(len(t) - 1):
        dr2 = -h2 * (r2[k] - z[k])
        dr1 = -h1 * (r1[k] - z[k]) + dr2
        r2[k + 1] = r2[k] + dt * dr2
        r1[k + 1] = r1[k] + dt * dr1
        y_r[k] = dr1

    y_r[-1] = -h1 * (r1[-1] - z[-1]) - h2 * (r2[-1] - z[-1])
    return r1, r2, y_r


def simulate_nn_estimator(
    t: np.ndarray,
    z: np.ndarray,
    z_dot: np.ndarray,
    lambda_eta: float = 1.5,
    gamma: float = 8.0,
    sigma_mod: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta = np.zeros_like(t)
    eta[0] = z[0]
    weights = np.zeros((len(t), 6), dtype=float)
    y_hat = np.zeros_like(t)
    dt = float(t[1] - t[0])

    for k in range(len(t) - 1):
        basis = basis_vector(float(z[k]), float(eta[k]))
        y_hat[k] = float(weights[k] @ basis)
        e_nn = float(z_dot[k] - y_hat[k])
        d_eta = -lambda_eta * eta[k] + z[k]
        d_w = gamma * (e_nn * basis - sigma_mod * weights[k])
        eta[k + 1] = eta[k] + dt * d_eta
        weights[k + 1] = weights[k] + dt * d_w

    y_hat[-1] = float(weights[-1] @ basis_vector(float(z[-1]), float(eta[-1])))
    return eta, weights, y_hat


def error_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    error = estimate - reference
    mean_abs = float(np.mean(np.abs(error)))
    mse = float(np.mean(error * error))
    rmse = float(np.sqrt(mse))
    return {"MEAN": mean_abs, "MSE": mse, "RMSE": rmse}


def write_metrics_csv(output_dir: Path, nn_metrics: dict[str, float], cf_metrics: dict[str, float]) -> None:
    csv_path = output_dir / "ch3_filter_nn_compare_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "nn", "cf"])
        for key in ["MEAN", "MSE", "RMSE"]:
            writer.writerow([key, f"{nn_metrics[key]:.6f}", f"{cf_metrics[key]:.6f}"])


def plot_output_response(
    output_dir: Path,
    t: np.ndarray,
    z_dot: np.ndarray,
    y_nn: np.ndarray,
    y_cf: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=SIM_FIGSIZE)
    ax.plot(t, z_dot, color="black", linestyle="-", linewidth=2.0, label=r"$\dot z$")
    ax.plot(t, y_nn, color="#FF0000", linestyle="--", linewidth=2.0, label=r"$\hat{\dot z}_{\mathrm{NN}}$")
    ax.plot(t, y_cf, color="#0000FF", linestyle="-.", linewidth=2.0, label=r"$y_r$")
    style_axes(ax, "时间 (s)", "输出响应")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.set_xlim(0.0, float(t[-1]))
    ax.set_ylim(-1.1, 1.1)
    leg = ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="0.35")
    leg.get_frame().set_linewidth(0.8)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.11, top=0.985)
    save_figure(fig, output_dir / "ch3_filter_nn_output_compare")


def plot_estimation_error(
    output_dir: Path,
    t: np.ndarray,
    err_nn: np.ndarray,
    err_cf: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=SIM_FIGSIZE)
    ax.plot(t, err_nn, color="#FF0000", linestyle="--", linewidth=2.0, label=r"$\epsilon_{\mathrm{NN}}$")
    ax.plot(t, err_cf, color="#0000FF", linestyle="-.", linewidth=2.0, label=r"$\epsilon_f$")
    style_axes(ax, "时间 (s)", "估计误差")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.set_xlim(0.0, float(t[-1]))
    ax.set_ylim(-0.30, 0.30)
    leg = ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="0.35")
    leg.get_frame().set_linewidth(0.8)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.11, top=0.985)
    save_figure(fig, output_dir / "ch3_filter_nn_error_compare")


def build_figures(output_dir: Path, duration: float = 10.0, dt: float = 0.001) -> None:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(0.0, duration + dt, dt)
    z, z_dot = benchmark_signal(t)
    _, _, y_cf = simulate_second_order_command_filter(t, z)
    _, _, y_nn = simulate_nn_estimator(t, z, z_dot)

    err_cf = z_dot - y_cf
    err_nn = z_dot - y_nn
    cf_metrics = error_metrics(z_dot, y_cf)
    nn_metrics = error_metrics(z_dot, y_nn)

    plot_output_response(output_dir, t, z_dot, y_nn, y_cf)
    plot_estimation_error(output_dir, t, err_nn, err_cf)
    write_metrics_csv(output_dir, nn_metrics, cf_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chapter 3 command-filter vs NN comparison figures.")
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args()
    build_figures(args.output_dir, args.duration, args.dt)


if __name__ == "__main__":
    main()
