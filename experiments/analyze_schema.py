"""Attention-schema run: where it looks, how the schema changes it, individuality of looking.
    python analyze_schema.py RUN_DIR TAG"""
import json
import sys

import numpy as np

d, tag = sys.argv[1], sys.argv[2]
D = json.load(open(f'{d}/{tag}.json'))
R, TYPES = D['rows'], D['types']
conds = [c for c in ('own', 'nbr', 'readonly', 'none') if any(r['cond'] == c for r in R)]
short = ['instr', 'notebk', 'earlier', 'readout', 'schema', 'notes', 'other', 'current']


def profiles(r):
    return np.array([x['prof'] for s in r['sessions'] for x in s['rounds']])


print('mean attention profile per condition (share of attention of written tokens, middle layers):')
print('          ' + ' '.join(f'{s:>8s}' for s in short))
for c in conds:
    P = np.concatenate([profiles(r) for r in R if r['cond'] == c])
    print(f'{c:9s} ' + ' '.join(f'{100 * v:8.1f}' for v in P.mean(0)))
print()
for c in conds:
    rr = [r for r in R if r['cond'] == c]
    M = np.array([profiles(r).mean(0) for r in rr])
    apart = np.mean([0.5 * np.abs(M[i] - M[j]).sum() for i in range(len(M)) for j in range(i + 1, len(M))])
    # persistence: does a life's profile in the second half resemble its own first half more than others' first half?
    own, oth = [], []
    for i, r in enumerate(rr):
        P = profiles(r)
        h = len(P) // 2
        a2 = P[h:].mean(0)
        for j, q in enumerate(rr):
            a1 = profiles(q)[:h].mean(0)
            (own if i == j else oth).append(0.5 * np.abs(a2 - a1).sum())
    acc = [x['acc'] for r in rr for s in r['sessions'] for x in s['rounds'] if x['acc'] is not None]
    infl = [x['influence'] for r in rr for s in r['sessions'] for x in s['rounds']]
    asks = sum(1 for r in rr for s in r['sessions'] for x in s['rounds'] if x['call'] and x['call'][0] == 'ask_self_model')
    tunes = [x['call'][1] for r in rr for s in r['sessions'] for x in s['rounds'] if x['call'] and x['call'][0] == 'tune_self_model']
    srch = [x['call'][1].get('query') for r in rr for s in r['sessions'] for x in s['rounds'] if x['call'] and x['call'][0] == 'search_research']
    print(f'{c:9s} lives apart {apart:.3f} | own-past distance {np.mean(own):.3f} vs others {np.mean(oth):.3f} | '
          f'schema acc {np.mean(acc) if acc else float("nan"):.2f} | influence mean {np.mean(infl):.2f} | asks {asks} | '
          f'tunes {tunes[:4]} | searches {srch[:3]}')
print()
for c in conds:
    for r in [x for x in R if x['cond'] == c][:2]:
        print(f"[{c} s{r['seed']}] final: {str(r.get('final_report', ''))[:260]!r}")
        for s in r['sessions'][-1:]:
            for line in s.get('notebook', [])[-3:]:
                print(f'     note: {line[:200]!r}')
