"""Self-reading map analysis across gains: settling, self-consistency, number of end states, own-vs-yoked.
    python analyze_map.py FILE.json [FILE.json ...]"""
import json
import sys

import numpy as np


def clusters(ends, tol=3.0):
    """Greedy count of distinct end states (mean abs difference above tol)."""
    reps = []
    for e in ends:
        if not any(np.abs(e - r).mean() <= tol for r in reps):
            reps.append(e)
    return len(reps)


for f in sys.argv[1:]:
    D = json.load(open(f))
    R = D['rows']
    print(f"== {f.split('/')[-1]} gain {D.get('gain')}")
    for fr in sorted({r['frame'] for r in R}):
        for loop in ('closed', 'yoked'):
            rr = [r for r in R if r['frame'] == fr and r['loop'] == loop]
            mm = np.mean([np.mean(r['self_mismatch'][-15:]) for r in rr])
            ch = np.mean([np.mean(r['change'][-15:]) for r in rr])
            ends = np.array([np.mean(np.array(r['reading'])[-10:], 0) for r in rr])
            spread = float(np.abs(ends[:, None] - ends[None]).mean())
            # tail variability of each life (wandering vs settled) and period
            var = np.mean([np.array(r['reading'])[-20:].std(0).mean() for r in rr])
            per = [r['period_tail'] for r in rr]
            print(f'  {fr:8s} {loop:7s} mismatch {mm:5.2f} change {ch:5.2f} tail-sd {var:5.2f} end-spread {spread:5.2f} '
                  f'end-states {clusters(ends)} periods {per}')
    # own vs yoked: does the closed loop describe itself better than a yoked reader describes the same lines?
    d = []
    for fr in sorted({r['frame'] for r in R}):
        c = {r['init']: np.mean(r['self_mismatch'][-30:]) for r in R if r['frame'] == fr and r['loop'] == 'closed'}
        y = {r['init']: np.mean(r['self_mismatch'][-30:]) for r in R if r['frame'] == fr and r['loop'] == 'yoked'}
        d += [y[i] - c[i] for i in c]
    d = np.array(d)
    print(f'  yoked minus closed mismatch (last 30): mean {d.mean():+.2f}, {np.sum(d > 0)}/{len(d)} lives closed better')
