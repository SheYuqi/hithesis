from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ROOT = Path('/home/syq/课题/thesis/Constant/SBC/0.707')
OUT = Path('hitbook/chinese/figures/ch5_sbc_example')
OUT.mkdir(parents=True, exist_ok=True)

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
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')

t = np.loadtxt(ROOT / 'time.csv', delimiter=',')
y = np.rad2deg(np.loadtxt(ROOT / 'pitch.csv', delimiter=','))
ref = float(np.mean(y[-500:].mean())) if np.ndim(y[-500:].mean()) else float(np.mean(y[-500:]))
yd = np.full_like(t, np.mean(y[-500:]))

# pick a steady-state jitter window after the main transient
start = np.searchsorted(t, 8.0)
window = max(300, int(3.0 / max(t[1]-t[0], 1e-6)))
best_i = start
best_std = -1.0
for i in range(start, len(t) - window, max(1, window // 20)):
    s = float(np.std(y[i:i+window]))
    if s > best_std:
        best_std = s
        best_i = i

x1 = float(t[best_i])
x2 = float(t[min(best_i + window, len(t)-1)])
maskz = (t >= x1) & (t <= x2)
ymin, ymax = float(y[maskz].min()), float(y[maskz].max())
span = max(ymax - ymin, 1e-3)
y1, y2 = ymin - 0.25 * span, ymax + 0.25 * span

fig, ax = plt.subplots(figsize=(6.1, 3.7), dpi=220)
ax.plot(t, yd, color='black', linestyle='-', linewidth=2.0, label=r'$y_d$')
ax.plot(t, y, color='#d62728', linestyle='--', linewidth=2.0, label=r'SBC+$\zeta$, $\zeta=0.707$')
ax.set_xlim(0.0, 60.0)
vals = np.concatenate([y, yd])
Ymin, Ymax = float(vals.min()), float(vals.max())
Span = max(Ymax - Ymin, 1e-3)
ax.set_ylim(Ymin - 0.08 * Span, Ymax + 0.08 * Span)
ax.set_xlabel('时间 (s)')
ax.set_ylabel('俯仰角 (deg)')
ax.grid(True, linestyle=(0, (1.0, 5.0)), color='0.7', linewidth=0.8)
ax.tick_params(direction='in', length=6, width=1.0, top=True, right=True)
leg = ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='0.35')
leg.get_frame().set_linewidth(0.8)

axins = inset_axes(ax, width='55%', height='45%', loc='center left', borderpad=1.5)
axins.plot(t, yd, color='black', linestyle='-', linewidth=1.6)
axins.plot(t, y, color='#d62728', linestyle='--', linewidth=1.8)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.grid(True, alpha=0.25)
axins.tick_params(direction='in', labelsize=12, top=True, right=True)
axins.set_xlabel('时间 (s)', fontsize=12)
for spine in axins.spines.values():
    spine.set_linewidth(1.0)
mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.2', lw=1.0)
fig.tight_layout()
save_dual(fig, OUT / 'ch5_sbc_pitch_example')
plt.close(fig)
