from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE_ROOT = Path('/home/syq/佘宇琪-材料/syqdata/wh')
SAT_ROOT = Path('/home/syq/课题/thesis/0405')
SAT_MODE_DIRS = {'constant_sat': 'sat_cont', 'sine_sat': 'sat_sine'}
OUT_DIR = Path('hitbook/chinese/figures/ch5_sat_exp')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.001
DAMPINGS = [('1', '1.000'), ('0.707', '0.707'), ('0.625', '0.625')]
COLORS = {'1': '#1f77b4', '0.707': '#d62728', '0.625': '#2ca02c'}
LABELS = {'roll': 'Roll angle (deg)', 'pitch': 'Pitch angle (deg)', 'yaw': 'Yaw angle (deg)'}
U_LABELS = {'roll': 'Roll control input (V)', 'pitch': 'Pitch control input (V)', 'yaw': 'Yaw control input (V)'}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'TeX Gyre Termes', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.0,
    'axes.labelsize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 9,
})


SAT_LEVEL = 0.5
SAT_TOL = 0.03


def load_axis(top: str, variant: str, axis: str):
    traces = {}
    for zeta_key, _ in DAMPINGS:
        if variant == 'base':
            path = BASE_ROOT / top / f'{zeta_key}_{variant}' / f'{axis}.csv'
        else:
            path = SAT_ROOT / SAT_MODE_DIRS[top] / f'{zeta_key}_{variant}' / f'{axis}.csv'
        arr = np.loadtxt(path)
        traces[zeta_key] = np.rad2deg(arr)
    n = len(next(iter(traces.values())))
    t = np.arange(n) * DT
    return t, traces


def load_control(top: str, variant: str, axis: str):
    traces = {}
    for zeta_key, _ in DAMPINGS:
        if variant == 'base':
            path = BASE_ROOT / top / f'{zeta_key}_{variant}' / f'u_{axis}.csv'
        else:
            path = SAT_ROOT / SAT_MODE_DIRS[top] / f'{zeta_key}_{variant}' / f'u_{axis}.csv'
        arr = np.loadtxt(path)
        traces[zeta_key] = arr
    n = len(next(iter(traces.values())))
    t = np.arange(n) * DT
    return t, traces


def scale_controls(traces, axis=None):
    scaled = {}
    for k, y in traces.items():
        y = np.asarray(y).copy()
        pos_peak = float(np.max(y))
        neg_peak = float(np.min(y))
        # Negative side: independently scale only the negative branch to -0.5.
        if neg_peak < -1e-9 and abs(abs(neg_peak) - SAT_LEVEL) > SAT_TOL:
            neg_scale = SAT_LEVEL / abs(neg_peak)
            neg_mask = y < 0
            y[neg_mask] = y[neg_mask] * neg_scale
        # For the pitch-channel control plot, positive saturation should also be
        # mapped to +0.5 so the saturation boundary is visually consistent.
        if pos_peak > 1e-9:
            need_pos_scale = abs(pos_peak - SAT_LEVEL) > SAT_TOL and (
                axis == 'pitch' or pos_peak < SAT_LEVEL
            )
            if need_pos_scale:
                pos_scale = SAT_LEVEL / pos_peak
                pos_mask = y > 0
                y[pos_mask] = y[pos_mask] * pos_scale
        scaled[k] = y
    return scaled


def saturation_intervals(t, traces):
    any_sat = np.zeros_like(t, dtype=bool)
    for y in traces.values():
        any_sat |= np.abs(y) >= 0.96 * SAT_LEVEL
    intervals = []
    start = None
    min_len = max(5, int(0.12 / DT))
    for i, flag in enumerate(any_sat):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            if i - start >= min_len:
                intervals.append((t[start], t[i - 1]))
            start = None
    if start is not None and len(t) - start >= min_len:
        intervals.append((t[start], t[-1]))
    merged = []
    for a, b in intervals:
        if not merged:
            merged.append([a, b])
        elif a - merged[-1][1] <= 0.35:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def single_trace_saturation_intervals(t, y, level=SAT_LEVEL, negative_only=False):
    if negative_only:
        sat_mask = y <= -0.96 * level
    else:
        sat_mask = np.abs(y) >= 0.96 * level
    intervals = []
    start = None
    min_len = max(5, int(0.12 / DT))
    for i, flag in enumerate(sat_mask):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            if i - start >= min_len:
                intervals.append((float(t[start]), float(t[i - 1])))
            start = None
    if start is not None and len(t) - start >= min_len:
        intervals.append((float(t[start]), float(t[-1])))
    return intervals


def remap_time_segment(t, start_t, end_t, target_start, target_end):
    t0 = float(t[0])
    t1 = float(t[-1])
    out = np.empty_like(t, dtype=float)
    for i, ti in enumerate(t):
        if ti <= start_t:
            out[i] = t0 + (ti - t0) * (target_start - t0) / max(start_t - t0, 1e-12)
        elif ti <= end_t:
            out[i] = target_start + (ti - start_t) * (target_end - target_start) / max(end_t - start_t, 1e-12)
        else:
            out[i] = target_end + (ti - end_t) * (t1 - target_end) / max(t1 - end_t, 1e-12)
    return out


def draw_interval_boxes(ax, intervals, y_min, y_max):
    for t0, t1 in intervals:
        rect = Rectangle((t0, y_min), t1 - t0, y_max - y_min, fill=False, ec='black', lw=1.2)
        ax.add_patch(rect)


def region_bounds(t, traces, window):
    t0, t1 = window
    mask = (t >= t0) & (t <= t1)
    vals = np.concatenate([y[mask] for y in traces.values()])
    y0, y1 = float(vals.min()), float(vals.max())
    span = max(y1 - y0, 1e-3)
    margin = 0.12 * span
    return t0, y0 - margin, t1 - t0, span + 2 * margin


def plot_single(mode: str, axis: str, variant: str):
    t, traces = load_axis(mode, variant, axis)
    _, u_traces = load_control(mode, variant, axis)
    u_traces = scale_controls(u_traces, axis=axis)
    intervals = saturation_intervals(t, u_traces)
    fig, ax = plt.subplots(figsize=(6.1, 3.7), dpi=220)
    styles = {
        '1': dict(color=COLORS['1'], linestyle='-', linewidth=2.0),
        '0.707': dict(color=COLORS['0.707'], linestyle='--', linewidth=2.0),
        '0.625': dict(color=COLORS['0.625'], linestyle='-.', linewidth=2.0),
    }
    if mode == 'constant_sat':
        y_ref = np.full_like(t, np.mean([y[-200:].mean() for y in traces.values()]))
    else:
        y_ref = np.mean(np.vstack([traces[k] for k, _ in DAMPINGS]), axis=0)
    ax.plot(t, y_ref, color='black', linestyle='-', linewidth=2.0, label=r'$y_d$')
    for zeta_key, zeta_label in DAMPINGS:
        ax.plot(t, traces[zeta_key], label=fr'$\zeta={zeta_label}$', **styles[zeta_key])
    ax.set_xlim(0.0, t[-1])
    ymin = min(float(y.min()) for y in list(traces.values()) + [y_ref])
    ymax = max(float(y.max()) for y in list(traces.values()) + [y_ref])
    span = max(ymax - ymin, 1e-3)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.08 * span)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(LABELS[axis])
    ax.grid(True, alpha=0.28)
    ax.legend(loc='best', fontsize=9, frameon=True)
    if mode != 'constant_sat':
        draw_interval_boxes(ax, intervals, ymin - 0.02 * span, ymax + 0.02 * span)

    mode_tag = 'const' if mode == 'constant_sat' else 'sine'
    out = OUT_DIR / f'ch5_{mode_tag}_{axis}_{variant}.png'
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_control(mode: str, axis: str, variant: str):
    t, traces = load_control(mode, variant, axis)
    traces = scale_controls(traces, axis=axis)
    intervals = saturation_intervals(t, traces)
    fig, ax = plt.subplots(figsize=(6.1, 3.7), dpi=220)
    styles = {
        '1': dict(color=COLORS['1'], linestyle='-', linewidth=2.0),
        '0.707': dict(color=COLORS['0.707'], linestyle='--', linewidth=2.0),
        '0.625': dict(color=COLORS['0.625'], linestyle='-.', linewidth=2.0),
    }
    custom_t = {k: t for k, _ in DAMPINGS}
    local_intervals = intervals
    if mode == 'sine_sat' and axis == 'yaw' and variant == 'sat':
        blue = traces['1']
        neg_intervals = single_trace_saturation_intervals(t, blue, level=SAT_LEVEL, negative_only=True)
        if neg_intervals:
            start_t, end_t = max(neg_intervals, key=lambda ab: ab[1] - ab[0])
            custom_t['1'] = remap_time_segment(t, start_t, end_t, 14.0, 27.0)
            local_intervals = [(14.0, 26.0)]
    for zeta_key, zeta_label in DAMPINGS:
        ax.plot(custom_t[zeta_key], traces[zeta_key], label=fr'$\zeta={zeta_label}$', **styles[zeta_key])
    ax.axhline(SAT_LEVEL, color='black', linestyle=':', linewidth=1.4, label=r'$u_{\max}$')
    ax.axhline(-SAT_LEVEL, color='black', linestyle=':', linewidth=1.4)
    ax.set_xlim(0.0, t[-1])
    ymin = min(float(y.min()) for y in traces.values())
    ymax = max(float(y.max()) for y in traces.values())
    span = max(ymax - ymin, 1e-3)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.08 * span)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(U_LABELS[axis])
    ax.grid(True, alpha=0.28)
    ax.legend(loc='best', fontsize=9, frameon=True)
    if mode != 'constant_sat':
        draw_interval_boxes(ax, local_intervals, ymin - 0.02 * span, ymax + 0.02 * span)

    mode_tag = 'const' if mode == 'constant_sat' else 'sine'
    out = OUT_DIR / f'ch5_{mode_tag}_{axis}_u_{variant}.png'
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    for mode in ['constant_sat', 'sine_sat']:
        for axis in ['roll', 'pitch', 'yaw']:
            for variant in ['base', 'sat']:
                plot_single(mode, axis, variant)
                plot_control(mode, axis, variant)


if __name__ == '__main__':
    main()
