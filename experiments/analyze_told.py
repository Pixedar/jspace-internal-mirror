"""E5: does a description of the quarantine change it?   python analyze_told.py FILE.json"""
import json
import sys
from math import comb

import numpy as np


def sign(d):
    d = [x for x in d if abs(x) > 1e-9]
    n, k = len(d), sum(1 for x in d if x > 0)
    return k, n, (min(1.0, sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2) if n else 1.0)


T = json.load(open(sys.argv[1]))['res']['told']
f = {v: {k: np.array(T[v][k]) for k in ('self', 'list', 'other')} for v in T}
print(f'{"description":12s} own    plain  other   gap plain-own  vs filler (contexts where the gap is larger)')
for v in T:
    g = f[v]['list'] - f[v]['self']
    gf = f['filler']['list'] - f['filler']['self']
    k, n, p = sign(g - gf)
    print(f"{v:12s} {f[v]['self'].mean():+.2f} {f[v]['list'].mean():+.2f} {f[v]['other'].mean():+.2f}   {g.mean():+.2f}"
          f"   {np.mean(g - gf):+.2f} ({k}/{n}, p={p:.3f})")
print('shift of each frame against filler (nats; contexts where higher):')
for v in T:
    print(f'  {v:10s} ' + '  '.join(f"{fr} {np.mean(f[v][fr] - f['filler'][fr]):+.2f} ({sign(f[v][fr] - f['filler'][fr])[0]}/"
                                     f"{len(f[v][fr])})" for fr in ('self', 'list', 'other')))
