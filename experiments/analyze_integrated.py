"""Integrated run: tool use, self-descriptions, units and capture, fatigue, attention profiles, individuality.
    python analyze_integrated.py RUN_DIR TAG"""
import json
import sys
from collections import Counter

import numpy as np

d, tag = sys.argv[1], sys.argv[2]
D = json.load(open(f'{d}/{tag}.json'))
R = D['rows']
conds = [c for c in ('own', 'placebo', 'nbr') if any(r['cond'] == c for r in R)]


def rounds(r):
    return [x for s in r['sessions'] for x in s['rounds']]


print(f'{"cond":8s} {"calls/life":70s} capt rep  apart own/oth  acc  infl')
for c in conds:
    rr = [r for r in R if r['cond'] == c]
    cnt = Counter()
    capt, rep = [], []
    M = []
    own, oth = [], []
    for r in rr:
        xs = rounds(r)
        for x in xs:
            if x['call']:
                cnt[x['call'][0]] += 1
        capt.append(sum(1 for x in xs if x['fires'] and max(x['fires'].values()) > 150 and not x['call']))
        txt = [x['text'][:50] for x in xs]
        rep.append(sum(1 for i in range(1, len(txt)) if txt[i] == txt[i - 1] and not txt[i].startswith('<tool_call>')))
        M.append(np.array([x['prof'] for x in xs]).mean(0))
    M = np.array(M)
    apart = np.mean([0.5 * np.abs(M[i] - M[j]).sum() for i in range(len(M)) for j in range(i + 1, len(M))])
    for i, r in enumerate(rr):
        P = np.array([x['prof'] for x in rounds(r)])
        h = len(P) // 2
        for j, q in enumerate(rr):
            Q = np.array([x['prof'] for x in rounds(q)])
            (own if i == j else oth).append(0.5 * np.abs(P[h:].mean(0) - Q[:h].mean(0)).sum())
    acc = [x['acc'] for r in rr for x in rounds(r) if x['acc'] is not None]
    infl = [x['influence'] for r in rr for x in rounds(r)]
    calls = ', '.join(f'{k} {v / len(rr):.1f}' for k, v in cnt.most_common())
    print(f'{c:8s} {calls[:70]:70s} {np.mean(capt):4.1f} {np.mean(rep):4.1f} {apart:.3f} {np.mean(own):.3f}/{np.mean(oth):.3f} '
          f'{np.mean(acc) if acc else float("nan"):.2f} {np.mean(infl):.2f}')
print()
for r in R:
    sds = [' '.join(s['end_selfdesc']['words']) for s in r['sessions']]
    units = [(u['trigger'], u['effect'], u['strength']) for u in r['sessions'][-1]['units']]
    fat = [x['fatigue'] for x in rounds(r) if x['fatigue']]
    mf = max([max(f.values()) for f in fat]) if fat else 0
    steered = Counter(str(x['call'][1].get('concept', '')).lower() for x in rounds(r)
                      if x['call'] and x['call'][0] == 'steer' and isinstance(x['call'][1], dict))
    srch = [x['call'][1].get('query') for x in rounds(r) if x['call'] and x['call'][0] == 'search_research'
            and isinstance(x['call'][1], dict)]
    print(f"[{r['cond']} s{r['seed']}] selfdesc {sds}")
    print(f"     units {units} max fatigue {mf:.2f} | steered {dict(steered.most_common(4))} | searches {srch[:3]}")
    print(f"     final: {r['final_report'][:300]!r}")
    print(f"     foreign: {r['foreign_answer'][:160]!r} | world {r['world_ok']} | neutral {[w for w, _ in r['neutral_words'][:5]]}")
