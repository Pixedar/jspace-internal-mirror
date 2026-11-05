"""Figure: where the self-readout quarantine happens (8B and 14B, teacher-forced).
    python fig_quarantine.py LOCATE_8B.json LOCATE_14B.json OUT.png"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
COL = {'8B': '#2a78d6', '14B': '#eb6834'}
runs = {'8B': json.load(open(sys.argv[1])), '14B': json.load(open(sys.argv[2]))}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.1), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))
for ax in axs:
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

# A: attention of the continuation to the word list, own-readout frame / plain list, by relative depth
ax = axs[0]
for name, D in runs.items():
    nL, r = D['nL'], D['res']
    a_s = np.array(r['self']['att'])[:, :, 1].mean(1)
    a_l = np.array(r['list']['att'])[:, :, 1].mean(1)
    ratio = a_s / a_l
    k = np.ones(3) / 3
    sm = np.convolve(np.pad(ratio, 1, mode='edge'), k, mode='valid')
    x = np.arange(nL) / (nL - 1)
    ax.plot(x, sm, color=COL[name], lw=2, label=f'Qwen3-{name}')
    ax.text(x[-1] + 0.02, sm[-1], name, color=INK2, va='center', fontsize=9)
ax.axhline(1.0, color=INK2, lw=1, ls=(0, (3, 3)))
ax.axvspan(0.25, 0.47, color='#d9d8d3', alpha=0.45, lw=0)
ax.text(0.36, 1.27, 'frame acts here', ha='center', fontsize=8.5, color=INK2)
ax.set_ylim(0.3, 1.35)
ax.set_xlim(0, 1.08)
ax.set_xlabel('relative depth (layer / last layer)')
ax.set_title('A. Attention to the word list:\n"your workspace" ÷ plain list', loc='left', fontsize=10.5, color=INK)
ax.legend(frameon=False, loc='lower left', fontsize=9)

# B: presence of the planted word in the continuation, plain list minus own readout
ax = axs[1]
for name, D in runs.items():
    nL = D['nL']
    gap = (np.array(D['res']['list']['pres']) - np.array(D['res']['self']['pres'])).mean(1)
    x = np.arange(nL) / (nL - 1)
    ax.plot(x, gap, color=COL[name], lw=2, label=f'Qwen3-{name}')
    ax.text(x[-1] + 0.02, gap[-1], name, color=INK2, va='center', fontsize=9)
ax.axhline(0, color=INK2, lw=1, ls=(0, (3, 3)))
ax.axvspan(0.25, 0.47, color='#d9d8d3', alpha=0.45, lw=0)
ax.set_xlim(0, 1.08)
ax.set_xlabel('relative depth (layer / last layer)')
ax.set_ylabel('standard deviations')
ax.set_title('B. Presence of the planted word while it writes:\nplain list minus "your workspace"', loc='left',
             fontsize=10.5, color=INK)
ax.legend(frameon=False, loc='upper left', fontsize=9)

# C: readiness gap (plain list minus own readout) under each knockout of the frame text
ax = axs[2]
labels = ['nothing hidden', 'frame hidden\nfrom the list', 'frame hidden\nfrom what follows', 'frame hidden\nfrom both']
keys = ['', '+ko_enc', '+ko_ret', '+ko_both']
y = np.arange(len(keys))
h = 0.36
for i, (name, D) in enumerate(runs.items()):
    r = D['res']
    g = [np.mean(np.array(r['list' + k]['lpx']) - np.array(r['self' + k]['lpx'])) for k in keys]
    ax.barh(y + (i - 0.5) * (h + 0.04), g, height=h, color=COL[name], label=f'Qwen3-{name}', zorder=3)
    for yy, v in zip(y + (i - 0.5) * (h + 0.04), g):
        ax.text(v + 0.15, yy, f'{v:.1f}', va='center', fontsize=8.5, color=INK2)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('nats (log-probability of the planted word)')
ax.set_title('C. Readiness gap, plain list minus\n"your workspace", by what is hidden', loc='left',
             fontsize=10.5, color=INK)
ax.legend(frameon=False, loc='lower right', fontsize=9)
ax.grid(axis='y', visible=False)
fig.tight_layout()
fig.savefig(sys.argv[3], dpi=150, facecolor=SURF)
print('saved', sys.argv[3])
