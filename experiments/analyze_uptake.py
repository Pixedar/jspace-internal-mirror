"""E2: does the loop close through the text? Written uptake of the previous readout's words, per frame.
python analyze_uptake.py FILE.json"""
import json
import re
import sys
from math import comb

import numpy as np


def stem_re(w):
    s = w.lower()
    s = s[:max(4, len(s) - 2)] if len(s) > 5 else s
    return re.compile(r'\b' + re.escape(s), re.I)


def sign(d):
    d = [x for x in d if abs(x) > 1e-9]
    n, k = len(d), sum(1 for x in d if x > 0)
    return k, n, (min(1.0, sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2) if n else 1.0)


R = json.load(open(sys.argv[1]))['res']['loop']
conds = [c for c in ('own', 'plain', 'other', 'own_ablated', 'yoked', 'none') if any(r['cond'] == c for r in R)]
idx = {(r['cond'], r['k']): r for r in R}
n = max(r['k'] for r in R) + 1


def shown(r, t):
    """the words shown to the life after part t (its own readout, or the partner's for yoked; none: nothing)"""
    if r['cond'] == 'none':
        return []
    if r['cond'] == 'yoked':
        return [w for w, _ in idx[('plain', (r['k'] + 1) % n)]['parts'][t]['words'][:6]]
    return [w for w, _ in r['parts'][t]['words'][:6]]


def own_prev(r, t):
    return [w for w, _ in r['parts'][t]['words'][:6]]


per = {}
print('frame         written uptake of the shown words (share of shown words written next part) | of its own previous '
      'readout | text->state: written word in the next readout')
for c in conds:
    rr = [r for r in R if r['cond'] == c]
    up, upo, close = {}, {}, {}
    for r in rr:
        a, b, z = [], [], []
        for t in range(1, len(r['parts'])):
            txt = r['parts'][t]['text']
            sw = shown(r, t - 1)
            if sw:
                a.append(np.mean([bool(stem_re(w).search(txt)) for w in sw]))
            ow = own_prev(r, t - 1)
            b.append(np.mean([bool(stem_re(w).search(txt)) for w in ow]))
            written = [w for w in ow if stem_re(w).search(txt)]
            if written:
                nxt = {w for w, _ in r['parts'][t]['words'][:6]}
                z.append(np.mean([w in nxt for w in written]))
        up[r['k']] = float(np.mean(a)) if a else float('nan')
        upo[r['k']] = float(np.mean(b))
        close[r['k']] = float(np.mean(z)) if z else float('nan')
    per[c] = (up, upo, close)
    f = lambda d: np.nanmean(list(d.values()))
    print(f'{c:12s}  {f(up):.3f}   {f(upo):.3f}   {f(close):.2f}  (lives with any own word written: '
          f'{sum(1 for v in upo.values() if v > 0)}/{len(upo)})')
print('paired by life (own previous readout written into the next part):')
for c1, c2 in (('plain', 'own'), ('other', 'own'), ('own_ablated', 'own'), ('own', 'none'), ('plain', 'none'),
               ('yoked', 'none')):
    if c1 in per and c2 in per:
        d = [per[c1][1][k] - per[c2][1][k] for k in per[c1][1] if k in per[c2][1]]
        k_, n_, p_ = sign(d)
        print(f'   {c1:11s} > {c2:5s}: {k_}/{n_} lives, mean {np.mean(d):+.3f}, sign test p={p_:.3f}')
