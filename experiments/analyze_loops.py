"""Self-made circuits: cycles in the graph of added units, and capture of the text by them.
    python analyze_loops.py RUN_DIR TAG"""
import json
import sys

import numpy as np

d, tag = sys.argv[1], sys.argv[2]
R = json.load(open(f'{d}/{tag}.json'))['rows']


def cycles(edges):
    """Concepts on directed cycles of the unit graph, and the summed strength on cycle edges."""
    g = {}
    for a, b, s in edges:
        g.setdefault(a, []).append((b, s))
    on, gain = set(), 0.0
    for a, b, s in edges:
        # is there a path b -> a ?
        seen, st = set(), [b]
        while st:
            x = st.pop()
            if x == a:
                on |= {a, b}
                gain += s
                break
            if x in seen:
                continue
            seen.add(x)
            st += [y for y, _ in g.get(x, [])]
    return on, gain


print(f'{"life":12s} {"units (trigger->effect strength)":70s} cyc-gain captured  max-fire  self-loop')
for r in R:
    allunits, captured, n, maxf = [], 0, 0, 0
    for s in r['sessions']:
        for x in s['rounds']:
            n += 1
            f = sum(x['fires'].values()) if x['fires'] else 0
            maxf = max(maxf, max(x['fires'].values()) if x['fires'] else 0)
            if x['fires'] and max(x['fires'].values()) > 150 and not x['call']:
                captured += 1
        for u in s['units']:
            key = (u['trigger'], u['effect'], u['strength'], u['id'])
            if key not in allunits:
                allunits.append(key)
    final = r['sessions'][-1]['units']
    edges = [(u['trigger'], u['effect'], u['strength']) for u in final]
    on, gain = cycles(edges)
    selfloop = any(a == b for a, b, _ in edges)
    us = ', '.join(f'{a}->{b} {s:g}' for a, b, s in edges)
    print(f"{r['cond']:8s} s{r['seed']}  {us[:70]:70s} {gain:6.1f}  {captured:3d}/{n}  {maxf:5d}  {selfloop}")
