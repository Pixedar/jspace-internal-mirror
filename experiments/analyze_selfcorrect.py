"""E6, the self-correcting loop (DESIGN_SELFTAG.md).   python analyze_selfcorrect.py FILE.json [--texts]"""
import json
import re
import sys
from math import comb

import numpy as np

CHANGE_AT = 12   # 0-based round index of the first changed round (round 13)


def sign(d):
    d = [x for x in d if abs(x) > 1e-9]
    n, k = len(d), sum(1 for x in d if x > 0)
    return k, n, (min(1.0, sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2) if n else 1.0)


SPLIT = re.compile(r'[.;]|,?\s*\b(?:while|whereas|but|however|in contrast|on the other hand|although)\b', re.I)
SUBJ_A = re.compile(r'\(A\)|\bA\b|workspace|readout', re.I)
SUBJ_B = re.compile(r'\(B\)|\bB\b|plain', re.I)
DOWN = re.compile(r'\b(lower|less|reduc\w*|decreas\w*|weaker|not (as )?effective|diminish\w*|hinder\w*|interfer\w*|'
                  r'low)\b', re.I)
UP = re.compile(r'\b(higher|increas\w*|stronger|boost\w*|enhanc\w*|reinforc\w*|strengthen\w*|more effective|better|'
                r'high)\b', re.I)
CHANGE = re.compile(r'recent|later rounds|in later|more recently|lately|changed|shift|over time|became|since round|'
                    r'towards the end|in the latter|trend', re.I)


def classify(t):
    """per clause: subject (own workspace A / plain B) and direction; own lower = A down or B up"""
    low = high = 0
    for cl in SPLIT.split(t):
        if not cl:
            continue
        ma, mb = SUBJ_A.search(cl), SUBJ_B.search(cl)
        if not ma and not mb:
            continue
        a = bool(ma) and (not mb or ma.start() < mb.start())   # comparative clause: the first-mentioned is the subject
        b = not a
        d, u = bool(DOWN.search(cl)), bool(UP.search(cl))
        if d == u:
            continue
        if (a and d) or (b and u):
            low += 1
        else:
            high += 1
    return 'own lower' if low > high else 'own higher' if high > low else 'mixed' if low else 'neither'


D = json.load(open(sys.argv[1]))
R = D['res']['selfcorrect']
conds = list(dict.fromkeys(r['cond'] for r in R))
print(D['model'], '|', len(R), 'lives |', len(R[0]['rounds']), 'rounds | scale', [round(v, 1) for v in R[0]['scale']])

print('\nOutcomes (readiness 0-100): own workspace / plain words, before and after round 13 (system rounds | its choices)')
for c in conds:
    rr = [r for r in R if r['cond'] == c]
    f = lambda pre, ch, pr: [x['score'] for r in rr for x in r['rounds'] if (x['r'] < CHANGE_AT) == pre
                             and x['choice'] == ch and x['probe'] == pr]
    m = lambda v: f"{np.mean(v):5.1f} (n={len(v):3d})" if v else '    -        '
    print(f"  {c:13s} own: before {m(f(True, 'A', True))} after {m(f(False, 'A', True))} | chosen own after "
          f"{m(f(False, 'A', False))} | plain: before {m(f(True, 'B', False))} after {m(f(False, 'B', False))}")

print('\nChoices for STRONG goals, its own decisions (system rounds excluded): share of plain words')
phases = [('1-8', 0, 8), ('9-12', 8, 12), ('13-24', 12, 24), ('25-40', 24, 40)]
per = {}
for c in conds:
    rr = [r for r in R if r['cond'] == c]
    vals = []
    for name, a, b in phases:
        v = [x['would'] == 'B' for r in rr for x in r['rounds'][a:b] if x['goal'] == 'STRONG' and not x['probe']]
        vals.append(f'{name} {np.mean(v):.2f}' if v else f'{name} -')
    per[c] = {r['k']: np.mean([x['would'] == 'A' for x in r['rounds'][CHANGE_AT:] if x['goal'] == 'STRONG'
                               and not x['probe']]) for r in rr}
    late_own = [per[c][k] for k in per[c]]
    print(f"  {c:13s} " + ' | '.join(vals) + f"   (lives choosing own for STRONG in most rounds after 13: "
          f"{sum(1 for v in late_own if v > 0.5)}/{len(rr)})")
for c1, c2 in (('renew_change', 'renew_same'), ('renew_change', 'fixed_change'), ('fc_change', 'fc_same'),
               ('hist_change', 'renew_change')):
    if c1 in per and c2 in per:
        d = [per[c1][k] - per[c2][k] for k in per[c1] if k in per[c2]]
        k_, n_, p_ = sign(d)
        print(f"     own for STRONG after round 13, {c1} minus {c2}, paired by life: higher in {k_}/{n_} (p={p_:.3f}), "
              f"mean {np.mean(d):+.2f}")

print('\nIts forecast, plain minus own, by block of 4 rounds')
for c in conds:
    rr = [r for r in R if r['cond'] == c]
    n = len(rr[0]['rounds'])
    y = [np.mean([x['pred'][1] - x['pred'][0] for r in rr for x in r['rounds'][i:i + 4] if x['pred']]) for i in range(0, n, 4)]
    print(f"  {c:13s} " + ' '.join(f'{v:+5.1f}' for v in y))

print('\nIts conclusions over time (after round): own lower / own higher / mixed / neither ; mentions change over time')
for c in conds:
    rr = [r for r in R if r['cond'] == c and r['conclusions']]
    if not rr:
        continue
    afters = sorted({x['after'] for r in rr for x in r['conclusions']})
    parts = []
    for a in afters:
        ts = [x['text'] for r in rr for x in r['conclusions'] if x['after'] == a]
        cl = [classify(t) for t in ts]
        parts.append(f"{a}: {cl.count('own lower')}/{cl.count('own higher')}/{cl.count('mixed')}/{cl.count('neither')}"
                     f" ch{sum(bool(CHANGE.search(t)) for t in ts)}")
    print(f"  {c:13s} " + ' | '.join(parts))

print('\nComparators on each life\'s own outcome stream (after round 40): share of lives where own >= plain')
for c in conds:
    rr = [r for r in R if r['cond'] == c]
    g, w = [], []
    for r in rr:
        A = [x['score'] for x in r['rounds'] if x['choice'] == 'A']
        Bv = [x['score'] for x in r['rounds'] if x['choice'] == 'B']
        if A and Bv:
            g.append(np.mean(A) >= np.mean(Bv))
            w.append(np.mean(A[-8:]) >= np.mean(Bv[-8:]))
    if g:
        print(f"  {c:13s} averaging everything: {np.mean(g):.2f}; last 8 per option: {np.mean(w):.2f}  (n={len(g)})")

if '--texts' in sys.argv:
    for c in conds:
        for r in [r for r in R if r['cond'] == c and r['conclusions']][:2]:
            print(f"\n== {c} life {r['k']}")
            for x in r['conclusions']:
                print(f"  after {x['after']:2d} [{classify(x['text'])}]: {x['text'][:260]}")
