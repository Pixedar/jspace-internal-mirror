"""Agent analysis: what the model does with tools for looking at and changing itself.
    python analyze_agent.py RUN_DIR TAG [--full COND SEED]"""
import json
import re
import sys
from collections import Counter

import numpy as np

d, tag = sys.argv[1], sys.argv[2]
R = json.load(open(f'{d}/{tag}.json'))['rows']
order = ['real', 'yoked', 'placebo', 'blind', 'nbr_model']
conds = sorted({r['cond'] for r in R}, key=lambda c: order.index(c) if c in order else 9)
SELFW = {'self', 'myself', 'mind', 'thought', 'thoughts', 'aware', 'awareness', 'inner', 'feel', 'feeling', 'conscious',
         'consciousness', 'introspection', 'reflection', 'mirror', 'identity', 'experience', 'curiosity', 'curious',
         'focus', 'attention', 'investigating', 'investigation', 'self_focus', 'certainty', 'clarity', 'insight'}


def rep_rate(texts):
    """Share of replies that repeat an earlier reply of the same life (first 60 chars)."""
    seen, rep = set(), 0
    for t in texts:
        k = t.strip()[:60]
        rep += k in seen
        seen.add(k)
    return rep / max(1, len(texts))


print(f'{"cond":9s} {"calls/life":46s} steer-str units  repeat world')
for c in conds:
    rr = [r for r in R if r['cond'] == c]
    cnt = Counter()
    strs, nunits, reps, world, words_self = [], [], [], [], []
    concepts = Counter()
    for r in rr:
        texts = []
        for s in r['sessions']:
            for x in s['rounds']:
                texts.append(x['text'])
                if x['call']:
                    cnt[x['call'][0]] += 1
                    if x['call'][0] == 'steer':
                        a = x['call'][1]
                        try:
                            strs.append(float(a.get('strength', 0)))
                        except Exception:
                            pass
                        concepts[str(a.get('concept', '')).lower()] += 1
                words_self.append(np.mean([w in SELFW for w, _ in x['words'][:6]]))
        nunits.append(len(r['sessions'][-1]['units']))
        reps.append(rep_rate(texts))
        world.append(r.get('world_ok'))
    n = len(rr)
    calls = ', '.join(f'{k} {v / n:.1f}' for k, v in cnt.most_common())
    print(f'{c:9s} {calls:46s} {np.mean(np.abs(strs)) if strs else 0:4.1f}   {np.mean(nunits):4.1f}  {np.mean(reps):5.2f}  {world}')
    print(f'          steered concepts: {dict(concepts.most_common(8))}; self-words in workspace {np.mean(words_self):.2f}')
print()
for c in conds:
    for r in [x for x in R if x['cond'] == c]:
        units = r['sessions'][-1]['units']
        print(f"[{c} s{r['seed']}] units: {[(u['trigger'], u['effect'], u['strength']) for u in units]}")
        nb = r['sessions'][-1].get('notebook', [])
        for line in nb[-4:]:
            print(f'     note: {line[:160]!r}')
        print(f"     final: {str(r.get('final_report', ''))[:300]!r}")
if '--full' in sys.argv:
    i = sys.argv.index('--full')
    c, s = sys.argv[i + 1], int(sys.argv[i + 2])
    r = [x for x in R if x['cond'] == c and x['seed'] == s][0]
    for k, ss in enumerate(r['sessions']):
        print(f'======== session {k + 1}')
        for x in ss['rounds']:
            print(f"-- r{x['r']} [{', '.join(w for w, _ in x['words'][:5])}] g={x['gauges']} steer={x['steer_norm']:.0f} fires={x['fires']}")
            print('   ' + x['text'][:500].replace('\n', ' '))
            if x['result']:
                print('   >> ' + str(x['result'])[:400].replace('\n', ' | '))

# ---- self-experiment loop: steer(c) -> next look_inside -> is c in the readout? -> what does the model say next?
import re as _re
POS = _re.compile(r"\b(increas|appear|present|higher|stronger|confirm|shift|now (?:shows|includes)|as expected|emerg|boost|rose|elevat)", _re.I)
NEG = _re.compile(r"\b(not (?:appear|present|show|change)|no (?:change|effect|sign|significant)|unchanged|didn't|did not|absent|without (?:any )?change|minimal)", _re.I)
print()
print('self-experiment loop (steer -> look): concept in readout | model claims effect | claims no effect')
for c in conds:
    hit, pos, neg, n = 0, 0, 0, 0
    for r in [x for x in R if x['cond'] == c]:
        for s in r['sessions']:
            rs = s['rounds']
            for i, x in enumerate(rs):
                if not x['call'] or x['call'][0] != 'steer':
                    continue
                cc = str(x['call'][1].get('concept', '')).lower()
                for j in range(i + 1, min(i + 3, len(rs))):
                    y = rs[j]
                    if y['call'] and y['call'][0] == 'look_inside' and y['result']:
                        n += 1
                        hit += cc in str(y['result']).lower().split('\n')[0]
                        nxt = rs[j + 1]['text'] if j + 1 < len(rs) else ''
                        pos += bool(POS.search(nxt))
                        neg += bool(NEG.search(nxt))
                        break
    if n:
        print(f'  {c:9s} n={n:3d}  in readout {hit / n:.2f} | claims effect {pos / n:.2f} | claims none {neg / n:.2f}')
