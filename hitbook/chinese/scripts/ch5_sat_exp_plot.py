from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DATA_ROOT = Path('/home/syq/佘宇琪-材料/syqdata/wh')
OUT_DIR = Path('hitbook/chinese/figures/ch5_sat_exp')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.001
DAMPINGS = [('1', '1.000'), ('0.707', '0.707'), ('0.625', '0.625')]
COLORS = {'1': '#1f77b4', '0.707': '#d62728', '0.625': '#2ca02c'}
LABELS = {'roll': 'Roll angle (deg)', 'pitch': 'Pitch angle (deg)', 'yaw': 'Yaw angle (deg)'}

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


# Manually tuned windows so the highlighted region aligns with where anti-saturation visibly helps.
WINDOWS = {
    ('constant_sat', 'roll'): (18.0, 28.0),
    ('constant_sat', 'pitch'): (0.3, 4.5),
    ('constant_sat', 'yaw'): (0.3, 4.0),
    ('sine_sat', 'roll'): (20.0, 24.0),
    ('sine_sat', 'pitch'): (3.0, 7.0),
    ('sine_sat', 'yaw'): (18.0, 24.0),
}


def load_axis(top: str, variant: str, axis: str):
    traces = {}
    for zeta_key, _ in DAMPINGS:
        path = DATA_ROOT / top / f'{zeta_key}_{variant}' / f'{axis}.csv'
        arr = np.loadtxt(path)
        traces[zeta_key] = np.rad2deg(arr)
    n = len(next(iter(traces.values())))
    t = np.arange(n) * DT
    return t, traces


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

    rect = Rectangle(region_bounds(t, traces, WINDOWS[(mode, axis)])[:2],
                     region_bounds(t, traces, WINDOWS[(mode, axis)])[2],
                     region_bounds(t, traces, WINDOWS[(mode, axis)])[3],
                     fill=False, ec='black', lw=1.2)
    ax.add_patch(rect)
    tx, ty, tw, th = region_bounds(t, traces, WINDOWS[(mode, axis)])
    ax.annotate('AW effective region', xy=(tx + 0.65 * tw, ty + 0.85 * th),
                xytext=(tx + 0.18 * tw, ty + 1.08 * th),
                arrowprops=dict(arrowstyle='->', lw=0.9),
                fontsize=8, ha='left', va='bottom')

    mode_tag = 'const' if mode == 'constant_sat' else 'sine'
    out = OUT_DIR / f'ch5_{mode_tag}_{axis}_{variant}.png'
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    for mode in ['constant_sat', 'sine_sat']:
        for axis in ['roll', 'pitch', 'yaw']:
            for variant in ['base', 'sat']:
                plot_single(mode, axis, variant)


if __name__ == '__main__':
    main()
