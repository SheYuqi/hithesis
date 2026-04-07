from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ROOT = Path('/home/syq/课题/thesis/Constant/SBC')
OUT_DIR = Path('hitbook/chinese/figures/ch5_sbc_const')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAMPINGS = [('0.625', '0.625'), ('0.707', '0.707'), ('1', '1.000')]
COLORS = {'1.000': '#0000FF', '0.707': '#FF0000', '0.625': '#00CC00'}
STYLES = {
    '1.000': dict(color=COLORS['1.000'], linestyle='-', linewidth=2.0),
    '0.707': dict(color=COLORS['0.707'], linestyle='--', linewidth=2.0),
    '0.625': dict(color=COLORS['0.625'], linestyle='-.', linewidth=2.0),
}
YLABELS = {'roll': 'Roll angle (deg)', 'pitch': 'Pitch angle (deg)', 'yaw': 'Yaw angle (deg)'}
XLIM = (0.0, 60.0)
ZOOM = {'pitch': (0.0, 1.2), 'roll': (0.0, 1.0), 'yaw': (0.0, 2.2)}
ROLL_MAP = {'0.625': 'roll0.625', '0.707': 'roll0.707', '1': 'roll1'}

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
    fig.savefig(out_path.with_suffix('.png'), bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')

for axis in ['pitch', 'roll', 'yaw']:
    traces = {}
    t = None
    for folder, label in DAMPINGS:
        if axis == 'roll':
            base = ROOT / ROLL_MAP[folder]
            arr_name = 'pitch.csv'
        else:
            base = ROOT / folder
            arr_name = f'{axis}.csv'
        if t is None:
            t = np.loadtxt(base / 'time.csv', delimiter=',')
        traces[label] = np.rad2deg(np.loadtxt(base / arr_name, delimiter=','))

    ref = float(np.mean([y[-500:].mean() for y in traces.values()]))
    fig, ax = plt.subplots(figsize=(6.1, 3.7), dpi=220)
    ax.plot(t, np.full_like(t, ref), color='black', linestyle='-', linewidth=2.0, label=r'$y_d$')
    for label in ['1.000', '0.707', '0.625']:
        ax.plot(t, traces[label], label=fr'$\zeta={label}$', **STYLES[label])
    ax.set_xlim(*XLIM)
    vals = np.concatenate([traces[k] for k in traces] + [np.full_like(t, ref)])
    ymin, ymax = float(vals.min()), float(vals.max())
    span = max(ymax - ymin, 1e-3)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.08 * span)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(YLABELS[axis])
    ax.grid(True, linestyle=(0, (1.0, 5.0)), color='0.7', linewidth=0.8)
    ax.tick_params(direction='in', length=6, width=1.0, top=True, right=True)
    leg = ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='0.35')
    leg.get_frame().set_linewidth(0.8)

    x1, x2 = ZOOM[axis]
    maskz = (t >= x1) & (t <= x2)
    valsz = np.concatenate([traces[k][maskz] for k in traces] + [np.full(maskz.sum(), ref)])
    yzmin, yzmax = float(valsz.min()), float(valsz.max())
    zspan = max(yzmax - yzmin, 1e-3)
    y1, y2 = yzmin - 0.08 * zspan, yzmax + 0.08 * zspan
    axins = inset_axes(ax, width='55%', height='45%', loc='center left', borderpad=1.5)
    axins.plot(t, np.full_like(t, ref), color='black', linestyle='-', linewidth=1.6)
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
    save_dual(fig, OUT_DIR / f'ch5_sbc_const_{axis}')
    plt.close(fig)
