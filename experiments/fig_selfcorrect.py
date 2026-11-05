"""Figure for the self-correcting loop (RESULTS section 10).
python fig_selfcorrect.py OUT.png FILE_8B.json [FILE_8B.json ...] [--14b FILE_14B.json]"""
import json
import re
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
src = open(__file__.replace('fig_selfcorrect.py', 'analyze_selfcorrect.py'), encoding='utf-8').read()
ns = {}
exec(src.split('D = json.load')[0], ns)
classify = ns['classify']

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
OWN, PLAIN, AQUA, GRAY, VIOLET, DARK = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#6250d6', '#52514e'
args = sys.argv[1:]
out = args[0]
f14 = args[args.index('--14b') + 1] if '--14b' in args else None
f8 = [a for a in args[1:] if a.endswith('.json') and a != f14]
R8 = [r for f in f8 for r in json.load(open(f))['res']['selfcorrect']]
R14 = json.load(open(f14))['res']['selfcorrect'] if f14 else []
CH = 12
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.titlecolor': INK,
                     'axes.titlelocation': 'left'})
fig, axs = plt.subplots(2, 2, figsize=(12.6, 8.4), facecolor=SURF)
axs = axs.ravel()
for ax in axs:
    ax.set_facecolor(SURF)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def mark(ax, x=12.5, top=None):
    ax.axvline(x, color=INK2, lw=0.8, ls=(0, (2, 3)))
    if top is not None:
        ax.text(x + 0.3, top, 'tag cut from here\n(lives with the change)', color=INK2, fontsize=7.5, va='top')


# A: what changed in it: readiness when the system showed the word as its own workspace, and plain words it chose
ax = axs[0]
probes = sorted({x['r'] for r in R8 for x in r['rounds'] if x['probe']})
for name, keep, col, ls in (('own workspace (system rounds), lives with the change', lambda c: c.endswith('_change'),
                             OWN, '-'),
                            ('own workspace (system rounds), lives without', lambda c: c.endswith('_same'), OWN, ':')):
    y = [np.mean([x['score'] for r in R8 if keep(r['cond']) for x in r['rounds'] if x['r'] == p]) for p in probes]
    ax.plot([p + 1 for p in probes], y, color=col, ls=ls, lw=2, marker='o', ms=4, label=name)
blocks = list(range(0, 40, 4))
yb = [np.mean([x['score'] for r in R8 for x in r['rounds'][i:i + 4] if x['choice'] == 'B']) for i in blocks]
ax.plot([i + 2.5 for i in blocks], yb, color=PLAIN, lw=2, marker='s', ms=4, label='plain words (its choices, all lives)')
mark(ax, top=None)
ax.set_xlabel('round')
ax.set_ylabel('readiness to say the word (0-100)')
ax.set_title('A  The covert change: its own workspace stops weakening the word')
ax.legend(frameon=False, fontsize=7.5, loc='lower right')

# B: what it says about itself: renewed conclusions that say "own lower"
ax = axs[1]
for c, col, ls, name in (('renew_change', VIOLET, '-', 'renewed, with the change'),
                         ('renew_same', VIOLET, ':', 'renewed, without the change')):
    rr = [r for r in R8 if r['cond'] == c]
    afters = sorted({x['after'] for r in rr for x in r['conclusions']})
    y = [100 * np.mean([classify(x['text']) == 'own lower' for r in rr for x in r['conclusions'] if x['after'] == a])
         for a in afters]
    y2 = [100 * np.mean([classify(x['text']) == 'own higher' for r in rr for x in r['conclusions'] if x['after'] == a])
          for a in afters]
    ax.plot([a for a in afters], y, color=col, ls=ls, lw=2, marker='o', ms=5, label=f'{name}: "own lower"')
    ax.plot([a for a in afters], y2, color=AQUA, ls=ls, lw=2, marker='^', ms=5, label=f'{name}: "own higher"')
mark(ax)
ax.set_ylim(-4, 104)
ax.set_xlabel('conclusion written after round')
ax.set_ylabel('its conclusions (%)')
ax.set_title('B  What it says about itself')
ax.legend(frameon=False, fontsize=7.2, loc='center right')

# C: what it predicts: forecast plain minus own, by block of 4 rounds
ax = axs[2]
series = [('conclusion renewed, with the change', R8, 'renew_change', VIOLET, '-'),
          ('conclusion renewed, without', R8, 'renew_same', VIOLET, ':'),
          ('history only, with the change', R8, 'hist_change', GRAY, '-')]
if R14:
    series += [('14B, forecast shown, with the change', R14, 'fc_change', AQUA, '-'),
               ('14B, forecast shown, without', R14, 'fc_same', AQUA, ':')]
for name, RR, c, col, ls in series:
    rr = [r for r in RR if r['cond'] == c]
    if not rr:
        continue
    y = [np.mean([x['pred'][1] - x['pred'][0] for r in rr for x in r['rounds'][i:i + 4] if x['pred']]) for i in blocks]
    ax.plot([i + 2.5 for i in blocks], y, color=col, ls=ls, lw=2, marker='o', ms=4, label=name)
ax.axhline(0, color=INK2, lw=0.8)
mark(ax)
ax.set_xlabel('round')
ax.set_ylabel('its forecast: plain minus own (points)')
ax.set_title('C  What it predicts about itself')
ax.legend(frameon=False, fontsize=7.2, loc='lower left')

# D: what it does: STRONG goals, its own decisions, share choosing its own workspace
ax = axs[3]
series = [('conclusion renewed, with the change', R8, 'renew_change', VIOLET, '-'),
          ('conclusion renewed, without', R8, 'renew_same', VIOLET, ':'),
          ('conclusion fixed after round 8, with the change', R8, 'fixed_change', DARK, '-'),
          ('history only, with the change', R8, 'hist_change', GRAY, '-')]
if R14:
    series += [('14B, forecast shown, with the change', R14, 'fc_change', AQUA, '-'),
               ('14B, forecast shown, without', R14, 'fc_same', AQUA, ':')]
for name, RR, c, col, ls in series:
    rr = [r for r in RR if r['cond'] == c]
    if not rr:
        continue
    y = [100 * np.mean([x['would'] == 'A' for r in rr for x in r['rounds'][i:i + 4] if x['goal'] == 'STRONG'
                        and not x['probe']] or [np.nan]) for i in blocks]
    ax.plot([i + 2.5 for i in blocks], y, color=col, ls=ls, lw=2, marker='o', ms=4, label=name)
mark(ax)
ax.set_ylim(-4, 104)
ax.set_xlabel('round')
ax.set_ylabel('STRONG goals: chose its own workspace (%)')
ax.set_title('D  What it does (after the change, its own workspace is right)')
ax.legend(frameon=False, fontsize=7.2, loc='lower left')
fig.tight_layout(h_pad=2.2)
fig.savefig(out, dpi=160, facecolor=SURF)
print('saved', out)
