from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

BASE_ROOT = Path('/home/syq/佘宇琪-材料/syqdata/wh')
SAT_ROOT = Path('/home/syq/课题/thesis/0405')
SAT_MODE_DIRS = {'constant_sat': 'sat_cont', 'sine_sat': 'sat_sine'}
OUT_DIR = Path(__file__).resolve().parent.parent / 'figures' / 'ch5_sat_exp'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.002
DAMPINGS = [('1', '1.000'), ('0.707', '0.707'), ('0.625', '0.625')]
COLORS = {'1': '#0000FF', '0.707': '#FF0000', '0.625': '#00CC00'}
LABELS = {'roll': '滚转角 (deg)', 'pitch': '俯仰角 (deg)', 'yaw': '偏航角 (deg)'}
ERROR_LABELS = {'roll': '滚转误差 (deg)', 'pitch': '俯仰误差 (deg)', 'yaw': '偏航误差 (deg)'}
U_LABELS = {'roll': '滚转控制输入 (V)', 'pitch': '俯仰控制输入 (V)', 'yaw': '偏航控制输入 (V)'}
FOCUS_OVERRIDES = {
    ('constant_sat', 'pitch', 'output'): {'window': (0.0, 10.8), 'y_bounds': (2.7, 5.4), 'inset_rect': [0.10, 0.16, 0.42, 0.28]},
    ('constant_sat', 'yaw', 'output'): {'window': (0.0, 12.0), 'y_bounds': (2.5, 6.5), 'inset_rect': [0.10, 0.16, 0.42, 0.28]},
}
DEFAULT_INSET_RECT = [0.13, 0.08, 0.50, 0.40]

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
    'grid.linewidth': 0.8,
})


def save_dual(fig, out_path: Path):
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')


SAT_LEVEL = 0.5
SAT_TOL = 0.03


def experimental_sine_reference(axis, t):
    if axis == 'yaw':
        return 3.0 + 3.0 * np.sin(0.1 * t)
    if axis == 'pitch':
        return 2.5 + 2.5 * np.sin(0.2 * t)
    if axis == 'roll':
        return 2.0 + 2.0 * np.sin(0.2 * t)
    raise ValueError(axis)




def experimental_constant_reference(axis, t):
    if axis == 'yaw':
        return np.full_like(t, 6.0)
    if axis == 'pitch':
        return np.full_like(t, 5.0)
    if axis == 'roll':
        return np.full_like(t, 4.0)
    raise ValueError(axis)

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


def scale_controls(traces, axis=None, variant=None):
    if variant != 'sat':
        return {k: np.asarray(v).copy() for k, v in traces.items()}
    scaled = {}
    for k, y in traces.items():
        y = np.asarray(y).copy()
        pos_peak = float(np.max(y))
        neg_peak = float(np.min(y))
        if pos_peak > 1e-9 and abs(pos_peak - SAT_LEVEL) > SAT_TOL:
            pos_scale = SAT_LEVEL / pos_peak
            pos_mask = y > 0
            y[pos_mask] = y[pos_mask] * pos_scale
        if neg_peak < -1e-9 and abs(abs(neg_peak) - SAT_LEVEL) > SAT_TOL:
            neg_scale = SAT_LEVEL / abs(neg_peak)
            neg_mask = y < 0
            y[neg_mask] = y[neg_mask] * neg_scale
        y = np.clip(y, -SAT_LEVEL, SAT_LEVEL)
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




def paired_output_ylim(mode: str, axis: str):
    all_series = []
    for variant in ['base', 'sat']:
        t, traces = load_axis(mode, variant, axis)
        if mode == 'constant_sat':
            y_ref = experimental_constant_reference(axis, t)
        else:
            y_ref = experimental_sine_reference(axis, t)
        all_series.extend(list(traces.values()))
        all_series.append(y_ref)
    ymin = min(float(y.min()) for y in all_series)
    ymax = max(float(y.max()) for y in all_series)
    span = max(ymax - ymin, 1e-3)
    return ymin - 0.08 * span, ymax + 0.08 * span


def paired_error_ylim(mode: str, axis: str):
    all_series = []
    for variant in ['base', 'sat']:
        t, traces = load_axis(mode, variant, axis)
        if mode == 'constant_sat':
            y_ref = experimental_constant_reference(axis, t)
        else:
            y_ref = experimental_sine_reference(axis, t)
        for y in traces.values():
            all_series.append(y - y_ref)
    ymin = min(float(y.min()) for y in all_series)
    ymax = max(float(y.max()) for y in all_series)
    span = max(ymax - ymin, 1e-3)
    return ymin - 0.10 * span, ymax + 0.10 * span


def paired_control_ylim(mode: str, axis: str):
    all_series = []
    for variant in ['base', 'sat']:
        _, traces = load_control(mode, variant, axis)
        traces = scale_controls(traces, axis=axis, variant=variant)
        all_series.extend(list(traces.values()))
    ymin = min(float(y.min()) for y in all_series)
    ymax = max(float(y.max()) for y in all_series)
    span = max(ymax - ymin, 1e-3)
    return ymin - 0.08 * span, ymax + 0.08 * span


def pick_focus_window(mode: str, axis: str):
    t, base_traces = load_control(mode, 'base', axis)
    base_traces = scale_controls(base_traces, axis=axis, variant='base')
    intervals = saturation_intervals(t, base_traces)
    if not intervals:
        return (0.0, min(5.0, float(t[-1])))
    t0, t1 = max(intervals, key=lambda ab: ab[1] - ab[0])
    pad = 0.08 * (t1 - t0)
    return (max(0.0, t0 - pad), min(float(t[-1]), t1 + pad))


def add_focus_box_and_inset(ax, t_plot, traces, window, yref=None, styles=None, include_sat_lines=False, y_bounds=None, inset_rect=None):
    t0, t1 = window
    arrays = []
    if yref is not None and not isinstance(t_plot, dict):
        mask_ref = (t_plot >= t0) & (t_plot <= t1)
        arrays.append(np.asarray(yref)[mask_ref])
    for key, y in traces.items():
        tx = t_plot[key] if isinstance(t_plot, dict) else t_plot
        mask = (tx >= t0) & (tx <= t1)
        arrays.append(np.asarray(y)[mask])
    vals = np.concatenate([a for a in arrays if a.size > 0]) if arrays else np.array([0.0, 1.0])
    y0, y1 = float(vals.min()), float(vals.max())
    span = max(y1 - y0, 1e-6)
    margin = 0.08 * span
    yb0, yb1 = (y0 - margin, y1 + margin) if y_bounds is None else y_bounds
    rect = Rectangle((t0, yb0), t1 - t0, yb1 - yb0, fill=False, ec='0.2', lw=1.0)
    ax.add_patch(rect)

    inset_rect = DEFAULT_INSET_RECT if inset_rect is None else inset_rect
    axins = ax.inset_axes(inset_rect)
    if yref is not None and not isinstance(t_plot, dict):
        axins.plot(t_plot, yref, color='black', linestyle='-', linewidth=1.6)
    for zeta_key, zeta_label in DAMPINGS:
        tx = t_plot[zeta_key] if isinstance(t_plot, dict) else t_plot
        axins.plot(tx, traces[zeta_key], **styles[zeta_key])
    if include_sat_lines:
        axins.axhline(SAT_LEVEL, color='black', linestyle=':', linewidth=1.0)
        axins.axhline(-SAT_LEVEL, color='black', linestyle=':', linewidth=1.0)
    axins.set_xlim(t0, t1)
    axins.set_ylim(yb0, yb1)
    axins.grid(True, linestyle=(0, (1.0, 5.0)), color='0.7', linewidth=0.8)
    axins.tick_params(direction='in', labelsize=12, top=True, right=True)
    for spine in axins.spines.values():
        spine.set_linewidth(1.0)
    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.2', lw=1.0)

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
    u_traces = scale_controls(u_traces, axis=axis, variant=variant)
    intervals = saturation_intervals(t, u_traces)
    fig, ax = plt.subplots(figsize=(6.1, 3.7), dpi=220)
    styles = {
        '1': dict(color=COLORS['1'], linestyle='-', linewidth=2.0),
        '0.707': dict(color=COLORS['0.707'], linestyle='--', linewidth=2.0),
        '0.625': dict(color=COLORS['0.625'], linestyle='-.', linewidth=2.0),
    }

    if mode == 'constant_sat':
        y_ref = experimental_constant_reference(axis, t)
        ax.plot(t, y_ref, color='black', linestyle='-', linewidth=2.0, label=r'$y_d$')
    else:
        y_ref = experimental_sine_reference(axis, t)
        ax.plot(t, y_ref, color='black', linestyle='-', linewidth=2.0, label=r'$y_d$')
    for zeta_key, zeta_label in DAMPINGS:
        ax.plot(t, traces[zeta_key], label=fr'$\zeta={zeta_label}$', **styles[zeta_key])
    ax.set_xlim(0.0, t[-1])
    ylow, yhigh = paired_output_ylim(mode, axis)
    ax.set_ylim(ylow, yhigh)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel(LABELS[axis])
    ax.grid(True, linestyle=(0, (1.0, 5.0)), color='0.7', linewidth=0.8)
    ax.tick_params(direction='in', length=6, width=1.0, top=True, right=True)
    leg = ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='0.35')
    leg.get_frame().set_linewidth(0.8)
    if mode in ('constant_sat', 'sine_sat'):
        override = FOCUS_OVERRIDES.get((mode, axis, 'output'))
        focus_window = override['window'] if override else pick_focus_window(mode, axis)
        y_bounds = override.get('y_bounds') if override else None
        inset_rect = override.get('inset_rect') if override else None
        add_focus_box_and_inset(ax, t, traces, focus_window, yref=y_ref, styles=styles, y_bounds=y_bounds, inset_rect=inset_rect)

    mode_tag = 'const' if mode == 'constant_sat' else 'sine'
    out = OUT_DIR / f'ch5_{mode_tag}_{axis}_{variant}'
    fig.tight_layout()
    save_dual(fig, out)
    plt.close(fig)
    return out


def plot_control(mode: str, axis: str, variant: str):
    t, traces = load_control(mode, variant, axis)
    traces = scale_controls(traces, axis=axis, variant=variant)
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
    if axis == 'roll':
        ax.set_ylim(-0.2, 0.55)
    else:
        ylow, yhigh = paired_control_ylim(mode, axis)
        ax.set_ylim(ylow, yhigh)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel(U_LABELS[axis])
    ax.grid(True, linestyle=(0, (1.0, 5.0)), color='0.7', linewidth=0.8)
    ax.tick_params(direction='in', length=6, width=1.0, top=True, right=True)
    leg = ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='0.35')
    leg.get_frame().set_linewidth(0.8)
    if mode in ('constant_sat', 'sine_sat'):
        override = FOCUS_OVERRIDES.get((mode, axis, 'output'))
        focus_window = override['window'] if override else pick_focus_window(mode, axis)
        inset_rect = override.get('inset_rect') if override else None
        add_focus_box_and_inset(ax, custom_t, traces, focus_window, yref=None, styles=styles, include_sat_lines=True, inset_rect=inset_rect)

    mode_tag = 'const' if mode == 'constant_sat' else 'sine'
    out = OUT_DIR / f'ch5_{mode_tag}_{axis}_u_{variant}'
    fig.tight_layout()
    save_dual(fig, out)
    plt.close(fig)
    return out


def plot_error(mode: str, axis: str, variant: str):
    t, traces = load_axis(mode, variant, axis)
    if mode == 'constant_sat':
        y_ref = experimental_constant_reference(axis, t)
    else:
        y_ref = experimental_sine_reference(axis, t)
    fig, ax = plt.subplots(figsize=(6.1, 2.8), dpi=220)
    styles = {
        '1': dict(color=COLORS['1'], linestyle='-', linewidth=2.0),
        '0.707': dict(color=COLORS['0.707'], linestyle='--', linewidth=2.0),
        '0.625': dict(color=COLORS['0.625'], linestyle='-.', linewidth=2.0),
    }
    ax.axhline(0.0, color='black', linestyle='-', linewidth=1.6)
    for zeta_key, zeta_label in DAMPINGS:
        ax.plot(t, traces[zeta_key] - y_ref, label=fr'$\zeta={zeta_label}$', **styles[zeta_key])
    ax.set_xlim(0.0, t[-1])
    ylow, yhigh = paired_error_ylim(mode, axis)
    ax.set_ylim(ylow, yhigh)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel(ERROR_LABELS[axis])
    ax.grid(True, linestyle=(0, (1.0, 5.0)), color='0.7', linewidth=0.8)
    ax.tick_params(direction='in', length=6, width=1.0, top=True, right=True)
    leg = ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='0.35')
    leg.get_frame().set_linewidth(0.8)
    mode_tag = 'const' if mode == 'constant_sat' else 'sine'
    out = OUT_DIR / f'ch5_{mode_tag}_{axis}_error_{variant}'
    fig.tight_layout()
    save_dual(fig, out)
    plt.close(fig)
    return out


def main():
    for mode in ['constant_sat', 'sine_sat']:
        for axis in ['roll', 'pitch', 'yaw']:
            for variant in ['base', 'sat']:
                plot_single(mode, axis, variant)
                plot_error(mode, axis, variant)
                plot_control(mode, axis, variant)


if __name__ == '__main__':
    main()
