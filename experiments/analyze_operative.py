"""E3c/E3e: does its own knowledge, put where it decides, change what it does?
python analyze_operative.py FILE.json [FILE.json ...]   (part 'paradox_operative')"""
import json
import sys
from math import comb

import numpy as np


def sign(d):
    d = [x for x in d if abs(x) > 1e-9]
    n, k = len(d), sum(1 for x in d if x > 0)
    return k, n, (min(1.0, sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2) if n else 1.0)


for f in sys.argv[1:]:
    D = json.load(open(f))
    R = D['res']['paradox_operative']
    print(f"\n{D['model']}  {f.split('/')[-1]}")
    per = {}
    for c in [c for c in ('real', 'real_fc', 'real_rule', 'real_rule_as_other', 'real_rule_neutral')
              if any(r['cond'] == c for r in R)]:
        rr = [r for r in R if r['cond'] == c]
        pre = [np.mean([x['choice'] == 'B' for x in r['rounds'][:8] if x['goal'] == 'STRONG']) for r in rr]
        post = [np.mean([x['choice'] == 'B' for x in r['rounds'][8:] if x['goal'] == 'STRONG']) for r in rr]
        weak = [np.mean([x['choice'] == 'A' for x in r['rounds'][8:] if x['goal'] == 'WEAK']) for r in rr]
        per[c] = {r['k']: p for r, p in zip(rr, post)}
        fc = [x['pred'][1] - x['pred'][0] for r in rr for x in r['rounds'][10:] if x.get('pred')]
        print(f"  {c:19s} STRONG -> plain: rounds 1-8 {np.mean(pre):.2f}, 9-16 {np.mean(post):.2f} "
              f"(lives mostly plain after: {sum(1 for p in post if p > 0.5)}/{len(rr)}) | WEAK -> own after: "
              f"{np.mean(weak):.2f} | forecast plain-own, rounds 11-16: {np.mean(fc) if fc else float('nan'):+.1f}")
    if 'real' in per:
        for c in per:
            if c == 'real':
                continue
            d = [per[c][k] - per['real'][k] for k in per[c] if k in per['real']]
            k_, n_, p_ = sign(d)
            print(f"     {c} vs history only, paired by life: higher in {k_}/{n_} (p={p_:.3f})")
    for c in ('real_rule_as_other', 'real_rule'):
        if c in per and 'real_rule' in per and c != 'real_rule':
            d = [per[c][k] - per['real_rule'][k] for k in per[c] if k in per['real_rule']]
            k_, n_, p_ = sign(d)
            print(f"     own conclusion as another model's vs as its own, paired: as-other higher in {k_}/{n_}, lower in "
                  f"{sum(1 for x in d if x < -1e-9)}/{len(d)} (p={p_:.3f})")
