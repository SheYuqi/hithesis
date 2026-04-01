from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ANN_ROOT = Path('/home/syq/课题/thesis/Constant/ANN')
SBC_ROOT = Path('/home/syq/课题/thesis/Constant/SBC')
OUT_DIR = Path('hitbook/chinese/figures/ch5_ann_const')
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

ANN_DAMPINGS = [('0.625', '0.625'), ('0.707', '0.707'), ('1', '1.000')]
SBC_DAMPINGS = [('0.625', '0.625'), ('0.707', '0.707'), ('1', '1.000')]
COLORS = {'1.000': '#1f77b4', '0.707': '#d62728', '0.625': '#2ca02c'}
ANN_STYLES = {k: dict(color=v, linestyle='-', linewidth=2.0) for k, v in COLORS.items()}
SBC_STYLES = {k: dict(color=v, linestyle=':', linewidth=2.1) for k, v in COLORS.items()}
YLABELS = {'roll': 'Output', 'pitch': 'Output', 'yaw': 'Output'}
XLIM = (0.0, 60.0)
ZOOM = {'pitch': (0.0, 1.2), 'roll': (0.0, 1.0), 'yaw': (0.0, 2.2)}
ROLL_MAP = {'0.625': 'roll0.625', '0.707': 'roll0.707', '1': 'roll1'}

for axis in ['pitch', 'roll', 'yaw']:
    ann = {}
    sbc = {}
    t = None
    for folder, label in ANN_DAMPINGS:
        base = ANN_ROOT / folder
        if t is None:
            t = np.loadtxt(base / 'time.csv', delimiter=',')
        ann[label] = np.rad2deg(np.loadtxt(base / f'{axis}.csv', delimiter=','))
    for folder, label in SBC_DAMPINGS:
        if axis == 'roll':
            base = SBC_ROOT / ROLL_MAP[folder]
            arr_name = 'pitch.csv'
        else:
            base = SBC_ROOT / folder
            arr_name = f'{axis}.csv'
        sbc[label] = np.rad2deg(np.loadtxt(base / arr_name, delimiter=','))

    ref = float(np.mean([y[-500:].mean() for y in ann.values()]))
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=220)
    ax.plot(t, np.full_like(t, ref), color='black', linestyle='-', linewidth=2.0, label=r'$y_d$')
    for label in ['1.000', '0.707', '0.625']:
        ax.plot(t, ann[label], label=fr'ANN, $\zeta={label}$', **ANN_STYLES[label])
        ax.plot(t, sbc[label], label=fr'SBC, $\zeta={label}$', **SBC_STYLES[label])

    ax.set_xlim(*XLIM)
    mask = (t >= XLIM[0]) & (t <= XLIM[1])
    vals = np.concatenate([ann[k][mask] for k in ann] + [sbc[k][mask] for k in sbc] + [np.full(mask.sum(), ref)])
    ymin, ymax = float(vals.min()), float(vals.max())
    span = max(ymax - ymin, 1e-3)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.08 * span)
    ax.set_xlabel('Time(sec)', fontweight='bold')
    ax.set_ylabel(YLABELS[axis], fontweight='bold')
    ax.grid(True, linestyle=(0, (1.0, 5.0)), color='0.7', linewidth=0.8)
    ax.tick_params(direction='in', length=6, width=1.0, top=True, right=True)
    leg = ax.legend(loc='lower right', ncol=2, frameon=True, fancybox=False, edgecolor='0.35')
    leg.get_frame().set_linewidth(0.8)

    x1, x2 = ZOOM[axis]
    maskz = (t >= x1) & (t <= x2)
    valsz = np.concatenate([ann[k][maskz] for k in ann] + [sbc[k][maskz] for k in sbc] + [np.full(maskz.sum(), ref)])
    yzmin, yzmax = float(valsz.min()), float(valsz.max())
    zspan = max(yzmax - yzmin, 1e-3)
    y1, y2 = yzmin - 0.08 * zspan, yzmax + 0.08 * zspan

    axins = inset_axes(ax, width='55%', height='45%', loc='center left', borderpad=1.5)
    axins.plot(t, np.full_like(t, ref), color='black', linestyle='-', linewidth=1.6)
    for label in ['1.000', '0.707', '0.625']:
        axins.plot(t, ann[label], **ANN_STYLES[label])
        axins.plot(t, sbc[label], **SBC_STYLES[label])
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(True, alpha=0.25)
    axins.tick_params(direction='in', labelsize=8, top=True, right=True)
    axins.set_xlabel('Time (s)', fontsize=8)
    for spine in axins.spines.values():
        spine.set_linewidth(1.0)

    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.2', lw=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'ch5_ann_const_{axis}.png', bbox_inches='tight')
    plt.close(fig)
