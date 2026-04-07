from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ROOT = Path('/home/syq/课题/thesis/0405/sbc_sine')
OUT_DIR = Path('hitbook/chinese/figures/ch5_sbc_sine')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAMPINGS = [('0.625', '0.625'), ('0.707', '0.707'), ('1', '1.000')]
COLORS = {'1.000': '#0000FF', '0.707': '#FF0000', '0.625': '#00CC00'}
STYLES = {
    '1.000': dict(color=COLORS['1.000'], linestyle='-', linewidth=2.0),
    '0.707': dict(color=COLORS['0.707'], linestyle='--', linewidth=2.0),
    '0.625': dict(color=COLORS['0.625'], linestyle='-.', linewidth=2.0),
}
YLABELS = {'roll': 'Roll angle (deg)', 'pitch': 'Pitch angle (deg)', 'yaw': 'Yaw angle (deg)'}
ERROR_YLABELS = {'roll': 'Roll error (deg)', 'pitch': 'Pitch error (deg)', 'yaw': 'Yaw error (deg)'}
XLIM = (0.0, 60.0)
ZOOM = {'pitch': (0.0, 3.0), 'roll': (0.0, 3.0), 'yaw': (0.0, 3.0)}

plt.rcParams.update({
    'font.family': 'Noto Serif CJK JP',
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.0,
    'axes.labelsize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 9,
})


def save_dual(fig, out_path: Path):
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')


def experimental_sine_reference(axis, t):
    if axis == 'yaw':
        return 3.0 + 3.0 * np.sin(0.1 * t)
    if axis == 'pitch':
        return 2.5 + 2.5 * np.sin(0.2 * t)
    if axis == 'roll':
        return 2.0 + 2.0 * np.sin(0.2 * t)
    raise ValueError(axis)


def reference_label(axis):
    if axis == 'yaw':
        return r'$y_d=3+3\sin(0.1t)$'
    if axis == 'pitch':
        return r'$y_d=2.5+2.5\sin(0.2t)$'
    if axis == 'roll':
        return r'$y_d=2+2\sin(0.2t)$'
    raise ValueError(axis)

for axis in ['pitch', 'roll', 'yaw']:
    traces = {}
    t = None
    for folder, label in DAMPINGS:
        base = ROOT / folder
        if t is None:
            t = np.loadtxt(base / 'time.csv', delimiter=',')
        traces[label] = np.rad2deg(np.loadtxt(base / f'{axis}.csv', delimiter=','))

    y_ref = experimental_sine_reference(axis, t)
    fig, ax = plt.subplots(figsize=(6.1, 3.7), dpi=220)
    ax.plot(t, y_ref, color='black', linestyle='-', linewidth=2.0, label=r'$y_d$')
    for label in ['1.000', '0.707', '0.625']:
        ax.plot(t, traces[label], label=fr'$\zeta={label}$', **STYLES[label])
    ax.set_xlim(*XLIM)
    vals = np.concatenate([traces[k] for k in traces] + [y_ref])
    ymin, ymax = float(vals.min()), float(vals.max())
    span = max(ymax - ymin, 1e-3)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.08 * span)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(YLABELS[axis])
    ax.grid(True, linestyle=(0, (1.0, 5.0)), color='0.7', linewidth=0.8)
    ax.tick_params(direction='in', length=6, width=1.0, top=True, right=True)
    leg = ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='0.35')
    leg.get_frame().set_linewidth(0.8)
    ax.text(0.03, 0.95, reference_label(axis), transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.20', fc='white', ec='0.35', lw=0.8), fontsize=10)

    x1, x2 = ZOOM[axis]
    maskz = (t >= x1) & (t <= x2)
    valsz = np.concatenate([traces[k][maskz] for k in traces] + [y_ref[maskz]])
    yzmin, yzmax = float(valsz.min()), float(valsz.max())
    zspan = max(yzmax - yzmin, 1e-3)
    y1, y2 = yzmin - 0.08 * zspan, yzmax + 0.08 * zspan
    axins = inset_axes(ax, width='55%', height='45%', loc='center left', borderpad=1.5)
    axins.plot(t, y_ref, color='black', linestyle='-', linewidth=1.6)
    for label in ['1.000', '0.707', '0.625']:
        axins.plot(t, traces[label], **STYLES[label])
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(True, alpha=0.25)
    axins.tick_params(direction='in', labelsize=8, top=True, right=True)
    axins.set_xlabel('Time (s)', fontsize=8)
    for spine in axins.spines.values():
        spine.set_linewidth(1.0)
    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.2', lw=1.0)
    fig.tight_layout()
    save_dual(fig, OUT_DIR / f'ch5_sbc_sine_{axis}')
    plt.close(fig)

    err = {label: traces[label] - y_ref for label in traces}
    fig, ax = plt.subplots(figsize=(6.1, 2.8), dpi=220)
    ax.axhline(0.0, color='black', linestyle='-', linewidth=1.6)
    for label in ['1.000', '0.707', '0.625']:
        ax.plot(t, err[label], label=fr'$\zeta={label}$', **STYLES[label])
    ax.set_xlim(*XLIM)
    vals = np.concatenate([err[k] for k in err])
    ymin, ymax = float(vals.min()), float(vals.max())
    span = max(ymax - ymin, 1e-3)
    ax.set_ylim(ymin - 0.10 * span, ymax + 0.10 * span)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ERROR_YLABELS[axis])
    ax.grid(True, linestyle=(0, (1.0, 5.0)), color='0.7', linewidth=0.8)
    ax.tick_params(direction='in', length=6, width=1.0, top=True, right=True)
    leg = ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='0.35')
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    save_dual(fig, OUT_DIR / f'ch5_sbc_sine_{axis}_error')
    plt.close(fig)
