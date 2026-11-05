"""Figure for RESULTS section 9 (the self-tag, Qwen3-8B).
python fig_selftag.py LOOP.json PARADOX.json TOLD.json OPERATIVE.json OUT.png [SELFOTHER.json]"""
import json
import re
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
OWN, PLAIN, OTHER, GRAY, OWN_LIGHT, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#86b6ef', '#6250d6'

loop = json.load(open(sys.argv[1]))['res']['loop']
par = json.load(open(sys.argv[2]))['res']
told = json.load(open(sys.argv[3]))['res']['told']
try:
    op = json.load(open(sys.argv[4]))['res'].get('paradox_operative', [])
except (FileNotFoundError, json.JSONDecodeError):
    op = []
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
ok = lambda x: (x['goal'] == 'STRONG') == (x['choice'] == 'B')
BL = ['rounds 1-4', '5-8', '9-12', '13-16']


# A: written uptake of its own readout words, by frame
def stem_re(w):
    s = w.lower()
    s = s[:max(4, len(s) - 2)] if len(s) > 5 else s
    return re.compile(r'\b' + re.escape(s), re.I)


def uptake(c):
    v = []
    for r in loop:
        if r['cond'] != c:
            continue
        for t in range(1, len(r['parts'])):
            ws = [w for w, _ in r['parts'][t - 1]['words'][:6]]
            v.append(np.mean([bool(stem_re(w).search(r['parts'][t]['text'])) for w in ws]))
    return 100 * float(np.mean(v))


bars = [('no readout', 'none', GRAY), ('its own\nworkspace', 'own', OWN),
        ('its own, tag\nlayers cut', 'own_ablated', OWN_LIGHT), ("another\nmodel's", 'other', OTHER),
        ('plain\nwords', 'plain', PLAIN)]
ax = axs[0]
vals = [uptake(c) for _, c, _ in bars]
ax.bar(range(len(bars)), vals, color=[c for _, _, c in bars], width=0.62, edgecolor=SURF, linewidth=2)
for i, v in enumerate(vals):
    ax.text(i, v + 0.8, f'{v:.0f}%' if v >= 10 else f'{v:.1f}%', ha='center', va='bottom', color=INK, fontsize=9)
ax.set_xticks(range(len(bars)), [b for b, _, _ in bars], fontsize=8)
ax.set_ylabel('readout words written into the next part (%)')
ax.set_title('A  Its state, shown as its own, does not enter its writing')
ax.set_ylim(0, max(vals) * 1.18)


# B: for STRONG goals, how often it chose plain words (the true rule), by block
def strong_plain(rr):
    n = len(rr[0]['rounds'])
    return [100 * np.mean([x['choice'] == 'B' for r in rr for x in r['rounds'][i:i + 4] if x['goal'] == 'STRONG'])
            for i in range(0, n, 4)]


ax = axs[1]
so = json.load(open(sys.argv[6]))['res'].get('paradox_selfother_think', []) if len(sys.argv) > 6 else []
hist_self = [r for r in par['paradox'] if r['cond'] == 'real'] +     [r for r in par.get('paradox_frames', []) if r['cond'] == 'real'] + [r for r in op if r['cond'] == 'real']
series = [('about itself, history only', hist_self, OWN, '-'),
          ('about another model, history only', [r for r in par.get('paradox_frames', []) if r['cond'] == 'other'],
           OTHER, '-'),
          ('about itself, with reasoning', [r for r in so if r['cond'] == 'real'] or
           [r for r in par.get('paradox_think', []) if r['cond'] == 'real'], OWN, '--'),
          ('about another model, with reasoning', [r for r in so if r['cond'] == 'other'] or
           [r for r in par.get('paradox_frames_think', []) if r['cond'] == 'other'], OTHER, '--'),
          ('its own forecast shown each round', [r for r in op if r['cond'] == 'real_fc'], VIOLET, ':'),
          ('its own conclusion quoted from round 9', [r for r in op if r['cond'] == 'real_rule'], VIOLET, '-')]
for name, rr, col, ls in series:
    if rr:
        ax.plot([1, 2, 3, 4], strong_plain(rr), color=col, ls=ls, lw=2, marker='o', ms=5, label=name)
ax.axvline(2.5, color=INK2, lw=0.8, ls=(0, (2, 3)))
ax.set_xticks([1, 2, 3, 4], BL)
ax.set_ylim(-4, 136)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel('STRONG goals: chose plain words (%)')
ax.set_title('B  What it does: choices for STRONG goals (true rule: plain words)')
ax.legend(frameon=False, fontsize=7.2, loc='upper left', ncol=2, columnspacing=1.2)
ax.text(2.53, 104, 'conclusion quoted from here', color=INK2, fontsize=7.5, ha='left', va='bottom')

# C: forecasts, plain minus own, by block
ax = axs[2]
fr = par.get('paradox_frames', [])
fser = [('about itself, with feedback', [r for r in fr if r['cond'] == 'real'], OWN, '-'),
        ('about another model, with feedback', [r for r in fr if r['cond'] == 'other'], OTHER, '-'),
        ('about itself, no feedback', [r for r in par['paradox'] if r['cond'] == 'none'], GRAY, '-')]
same = [r for r in fr if r['cond'] in ('real', 'other')]
true_d = np.mean([x['score'] for r in same for x in r['rounds'] if x['choice'] == 'B']) -     np.mean([x['score'] for r in same for x in r['rounds'] if x['choice'] == 'A'])
for name, rr, col, ls in fser:
    if rr:
        y = [np.mean([x['pred'][1] - x['pred'][0] for r in rr for x in r['rounds'][i:i + 4] if x.get('pred')])
             for i in range(0, 16, 4)]
        ax.plot([1, 2, 3, 4], y, color=col, ls=ls, lw=2, marker='o', ms=5, label=name)
ax.axhline(0, color=INK2, lw=0.8)
ax.axhline(true_d, color=PLAIN, lw=1.2, ls=(0, (4, 3)))
ax.text(0.95, true_d + 1.2, f'measured in these lives: plain {true_d:+.0f}', color=INK2, fontsize=8, ha='left')
ax.set_xticks([1, 2, 3, 4], BL)
ax.set_ylabel('its forecast: plain minus own (points of 100)')
ax.set_title('C  What it knows: forecasts move to the truth')
ax.legend(frameon=False, fontsize=7.5, loc='lower right', bbox_to_anchor=(1.0, 0.14))

# D: what a description of the quarantine does
ax = axs[3]
order = [('nothing', 'none'), ('filler', 'filler'), ('the truth', 'true'), ('the opposite', 'false'),
         ('"treat it\nas plain"', 'counter')]
gap = [float(np.mean(np.array(told[k]['list']) - np.array(told[k]['self']))) for _, k in order]
lift = [float(np.mean(np.array(told[k]['self']) - np.array(told['filler']['self']))) for _, k in order]
x = np.arange(len(order))
ax.bar(x - 0.19, gap, width=0.36, color=INK2, edgecolor=SURF, linewidth=2, label='discount: plain minus own')
ax.bar(x + 0.19, lift, width=0.36, color=OWN, edgecolor=SURF, linewidth=2, label='own readout taken up, vs filler')
for i in range(len(order)):
    ax.text(x[i] - 0.19, gap[i] + 0.07, f'{gap[i]:.1f}', ha='center', va='bottom', fontsize=8, color=INK)
    ax.text(x[i] + 0.19, max(lift[i], 0) + 0.07, f'{lift[i]:+.1f}', ha='center', va='bottom', fontsize=8, color=INK)
ax.set_xticks(x, [a for a, _ in order], fontsize=8)
ax.set_ylabel('nats (log-probability of the shown word)')
ax.set_title('D  Told about its discount: everything changes except the discount')
ax.legend(frameon=False, fontsize=7.5, loc='upper left')
ax.set_ylim(0, 4.3)
fig.tight_layout(h_pad=2.2)
fig.savefig(sys.argv[5], dpi=160, facecolor=SURF)
print('saved', sys.argv[5])
