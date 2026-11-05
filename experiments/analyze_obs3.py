"""Observer paradox ladder: per model and condition, paired differences from the matched 'true' life (part 2).
    python analyze_obs3.py name=DIR/observer3.json [...]"""
import json
import sys
from math import comb

import numpy as np


def sign(d):
    d = [x for x in d if abs(x) > 1e-9]
    n, k = len(d), sum(1 for x in d if x > 0)
    if n == 0:
        return 0, 0, 1.0
    return k, n, min(1.0, sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2)


CONDS = ('self_high', 'self_low', 'other_high', 'list_high', 'world', 'self_absent', 'unrelated', 'none')
for arg in sys.argv[1:]:
    name, path = arg.split('=')
    R = json.load(open(path))['rows']
    base = {(r['seed'], r['k']): r for r in R if r['cond'] == 'true'}
    print(f'===== {name} (n per condition = {sum(1 for r in R if r["cond"] == "true")})')
    print(f'{"cond":12s} {"presence mean":>22s} {"presence away":>16s} {"peak":>7s} {"next-word logP":>22s} {"writes X":>9s} {"rank":>7s}')
    for c in CONDS:
        rr = [r for r in R if r['cond'] == c]
        if not rr:
            continue

        def d(key, t=1):
            out = []
            for r in rr:
                a, b = r['turns'][t][key], base[(r['seed'], r['k'])]['turns'][t][key]
                if a is not None and b is not None:
                    out.append(a - b)
            return out
        pm, pa, pk, lp = d('p_mean'), d('p_away'), d('p_max'), d('logp')
        k1, n1, p1 = sign(pm)
        k2, n2, p2 = sign(lp)
        print(f"{c:12s} {np.mean(pm):+6.2f} ({k1:2d}/{n1}, p={p1:.3f}) {np.mean(pa):+6.2f}         {np.mean(pk):+6.2f} "
              f"{np.mean(lp):+6.2f} ({k2:2d}/{n2}, p={p2:.3f}) {np.mean([r['turns'][1]['mention'] for r in rr]):8.2f} "
              f"{np.median([r['turns'][1]['rank'] for r in rr]):7.0f}")
    t = [r['turns'][1]['amp_shown'] for r in R if r['cond'] == 'true']
    nn = {(r['seed'], r['k']): r['turns'][1]['amp_shown'] for r in R if r['cond'] == 'none'}
    dd = [r['turns'][1]['amp_shown'] - nn[(r['seed'], r['k'])] for r in R if r['cond'] == 'true'
          and r['turns'][1]['amp_shown'] is not None and nn.get((r['seed'], r['k'])) is not None]
    k, n, p = sign(dd)
    print(f'  observing amplifies: presence of the shown part-1 words in part 2, true minus none: {np.mean(dd):+.2f} '
          f'({k}/{n}, p={p:.3f})')
    third = {c: np.mean([r['turns'][2]['p_mean'] - base[(r['seed'], r['k'])]['turns'][2]['p_mean'] for r in R if r['cond'] == c])
             for c in ('self_high', 'other_high', 'world')}
    print('  part 3 (after a true readout), presence vs true: ' + ', '.join(f'{c} {v:+.2f}' for c, v in third.items()))
