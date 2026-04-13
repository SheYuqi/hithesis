from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ANN_ROOT = Path('/home/syq/课题/thesis/Sinetarget/ANN')
OUT_DIR = Path('hitbook/chinese/figures/ch5_ann_sine')
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANN_DAMPINGS = [('0.625', '0.625'), ('0.707', '0.707'), ('1', '1.000')]
COLORS = {'1.000': '#0000FF', '0.707': '#FF0000', '0.625': '#00CC00'}
ANN_STYLES = {
    '1.000': dict(color=COLORS['1.000'], linestyle='-', linewidth=2.0),
    '0.707': dict(color=COLORS['0.707'], linestyle='--', linewidth=2.0),
    '0.625': dict(color=COLORS['0.625'], linestyle='-.', linewidth=2.0),
}
YLABELS = {'roll': '滚转角 (deg)', 'pitch': '俯仰角 (deg)', 'yaw': '偏航角 (deg)'}
ERROR_YLABELS = {'roll': '滚转误差 (deg)', 'pitch': '俯仰误差 (deg)', 'yaw': '偏航误差 (deg)'}
XLIM = (0.0, 60.0)
ZOOM = {'pitch': (0.0, 3.0), 'roll': (0.0, 3.0), 'yaw': (0.0, 3.0)}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['AR PL UMing CN', 'Noto Serif CJK JP', 'Noto Serif CJK SC', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'axes.linewidth': 1.0,
    'font.size': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


def save_dual(fig, out_path: Path):
    fig.savefig(out_path.with_suffix('.pdf'))


def experimental_sine_reference(axis, t):
    if axis == 'yaw':
        return 3.0 + 3.0 * np.sin(0.1 * t)
    if axis == 'pitch':
        return 2.5 + 2.5 * np.sin(0.2 * t)
    if axis == 'roll':
        return 2.0 + 2.0 * np.sin(0.2 * t)
    raise ValueError(axis)


for axis in ['pitch', 'roll', 'yaw']:
    ann = {}
    t = None
    for folder, label in ANN_DAMPINGS:
        base = ANN_ROOT / folder
        if t is None:
            t = np.loadtxt(base / 'time.csv', delimiter=',')
        ann[label] = np.rad2deg(np.loadtxt(base / f'{axis}.csv', delimiter=','))

    y_ref = experimental_sine_reference(axis, t)
    fig, ax = plt.subplots(figsize=(6.1, 3.7), dpi=220)
    ax.plot(t, y_ref, color='black', linestyle='-', linewidth=2.0, label=r'$y_d$')
    for label in ['1.000', '0.707', '0.625']:
        ax.plot(t, ann[label], label=fr'$\zeta={label}$', **ANN_STYLES[label])

    ax.set_xlim(*XLIM)
    mask = (t >= XLIM[0]) & (t <= XLIM[1])
    vals = np.concatenate([ann[k][mask] for k in ann] + [y_ref[mask]])
    ymin, ymax = float(vals.min()), float(vals.max())
    span = max(ymax - ymin, 1e-3)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.08 * span)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel(YLABELS[axis])
    ax.grid(True, linestyle=(0, (1.0, 5.0)), color='0.7', linewidth=0.8)
    ax.tick_params(direction='in', length=6, width=1.0, top=True, right=True)
    leg = ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='0.35')
    leg.get_frame().set_linewidth(0.8)
    x1, x2 = ZOOM[axis]
    maskz = (t >= x1) & (t <= x2)
    valsz = np.concatenate([ann[k][maskz] for k in ann] + [y_ref[maskz]])
    yzmin, yzmax = float(valsz.min()), float(valsz.max())
    zspan = max(yzmax - yzmin, 1e-3)
    y1, y2 = yzmin - 0.08 * zspan, yzmax + 0.08 * zspan

    axins = inset_axes(ax, width='55%', height='45%', loc='center left', borderpad=1.5)
    axins.plot(t, y_ref, color='black', linestyle='-', linewidth=1.6)
    for label in ['1.000', '0.707', '0.625']:
        axins.plot(t, ann[label], **ANN_STYLES[label])
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(True, alpha=0.25)
    axins.tick_params(direction='in', labelsize=12, top=True, right=True)
    axins.set_xlabel('时间 (s)', fontsize=12)
    for spine in axins.spines.values():
        spine.set_linewidth(1.0)

    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.2', lw=1.0)
    fig.tight_layout()
    save_dual(fig, OUT_DIR / f'ch5_ann_sine_{axis}')
    plt.close(fig)

    err = {label: ann[label] - y_ref for label in ann}
    fig, ax = plt.subplots(figsize=(6.1, 3.7), dpi=220)
    ax.axhline(0.0, color='black', linestyle='-', linewidth=1.6)
    for label in ['1.000', '0.707', '0.625']:
        ax.plot(t, err[label], label=fr'$\zeta={label}$', **ANN_STYLES[label])
    ax.set_xlim(*XLIM)
    vals = np.concatenate([err[k][mask] for k in err])
    ymin, ymax = float(vals.min()), float(vals.max())
    span = max(ymax - ymin, 1e-3)
    ax.set_ylim(ymin - 0.10 * span, ymax + 0.10 * span)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel(ERROR_YLABELS[axis])
    ax.grid(True, linestyle=(0, (1.0, 5.0)), color='0.7', linewidth=0.8)
    ax.tick_params(direction='in', length=6, width=1.0, top=True, right=True)
    leg = ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='0.35')
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    save_dual(fig, OUT_DIR / f'ch5_ann_sine_{axis}_error')
    plt.close(fig)
