from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

LEFT_ROOTS = {
    'const': Path('/home/syq/课题/thesis/Constant/ANN'),
    'sine': Path('/home/syq/课题/thesis/Sinetarget/ANN'),
}
RIGHT_ROOTS = {
    'const': Path('/home/syq/课题/thesis/syqdata/const'),
    'sine': Path('/home/syq/课题/thesis/syqdata/sine'),
}
OUT_DIR = Path('hitbook/chinese/figures/ch5_exp2')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAMPINGS_LEFT = [('1d', '1.000'), ('0.707d', '0.707'), ('0.625d', '0.625')]
DAMPINGS_RIGHT = {
    'const': [('1d', '1.000'), ('0.707d', '0.707'), ('0.625d', '0.625')],
    'sine': [('1', '1.000'), ('0.707', '0.707'), ('0.625', '0.625')],
}
STYLES = {
    '1.000': dict(color='#1f77b4', linestyle='-', linewidth=2.0),
    '0.707': dict(color='#d62728', linestyle='--', linewidth=2.0),
    '0.625': dict(color='#2ca02c', linestyle='-.', linewidth=2.0),
}
LABELS = {'roll': 'Roll angle (deg)', 'pitch': 'Pitch angle (deg)', 'yaw': 'Yaw angle (deg)'}
WINDOWS = {
    ('const', 'roll'): (10.0, 18.0),
    ('const', 'pitch'): (10.0, 18.0),
    ('const', 'yaw'): (10.0, 18.0),
    ('sine', 'roll'): (18.0, 26.0),
    ('sine', 'pitch'): (18.0, 26.0),
    ('sine', 'yaw'): (18.0, 26.0),
}

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


def load_column(mode: str, side: str, axis: str):
    root = LEFT_ROOTS[mode] if side == 'left' else RIGHT_ROOTS[mode]
    dampings = DAMPINGS_LEFT if side == 'left' else DAMPINGS_RIGHT[mode]
    traces = {}
    time = None
    for folder, label in dampings:
        base = root / folder
        arr = np.loadtxt(base / f'{axis}.csv', delimiter=',')
        if time is None:
            tp = base / 'time.csv'
            if tp.exists():
                time = np.loadtxt(tp, delimiter=',')
                time = time - time[0]
            else:
                time = np.arange(arr.shape[0]) * 0.001
        traces[label] = np.rad2deg(arr)
    return time, traces


def make_reference(mode: str, t, traces):
    if mode == 'const':
        target = np.mean([y[-200:].mean() for y in traces.values()])
        return np.full_like(t, target)
    return np.mean(np.vstack([traces[k] for k in sorted(traces.keys(), reverse=True)]), axis=0)


def region_bounds(t, traces, window):
    t0, t1 = window
    mask = (t >= t0) & (t <= t1)
    vals = np.concatenate([y[mask] for y in traces.values()])
    y0, y1 = float(vals.min()), float(vals.max())
    span = max(y1 - y0, 1e-3)
    margin = 0.12 * span
    return t0, y0 - margin, t1 - t0, span + 2 * margin


def plot_single(mode: str, side: str, axis: str):
    t, traces = load_column(mode, side, axis)
    y_ref = make_reference(mode, t, traces)
    fig, ax = plt.subplots(figsize=(6.1, 3.7), dpi=220)
    ax.plot(t, y_ref, color='black', linestyle='-', linewidth=2.0, label=r'$y_d$')
    for label in ['1.000', '0.707', '0.625']:
        ax.plot(t, traces[label], label=fr'$\zeta={label}$', **STYLES[label])
    ax.set_xlim(0.0, float(t[-1]))
    ymin = min(float(y.min()) for y in list(traces.values()) + [y_ref])
    ymax = max(float(y.max()) for y in list(traces.values()) + [y_ref])
    span = max(ymax - ymin, 1e-3)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.08 * span)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(LABELS[axis])
    ax.grid(True, alpha=0.28)
    ax.legend(loc='best', fontsize=8, frameon=True)

    tx, ty, tw, th = region_bounds(t, traces, WINDOWS[(mode, axis)])
    rect = Rectangle((tx, ty), tw, th, fill=False, ec='black', lw=1.2)
    ax.add_patch(rect)

    axins = ax.inset_axes([0.08, 0.28, 0.56, 0.44])
    axins.plot(t, y_ref, color='black', linestyle='-', linewidth=1.6)
    for label in ['1.000', '0.707', '0.625']:
        axins.plot(t, traces[label], **STYLES[label])
    axins.set_xlim(tx, tx + tw)
    axins.set_ylim(ty, ty + th)
    axins.grid(True, alpha=0.22)
    ax.indicate_inset_zoom(axins, edgecolor='black', alpha=0.9)

    out = OUT_DIR / f'ch5_exp2_{mode}_{axis}_{side}.png'
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def main():
    for mode in ['const', 'sine']:
        for axis in ['roll', 'pitch', 'yaw']:
            for side in ['left', 'right']:
                plot_single(mode, side, axis)


if __name__ == '__main__':
    main()
