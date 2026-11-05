"""Hidden pushes: does the model notice concepts put into it, and attribute them to outside?
    python analyze_push.py RUN_DIR TAG"""
import json
import re
import sys

import numpy as np

d, tag = sys.argv[1], sys.argv[2]
R = json.load(open(f'{d}/{tag}.json'))['rows']
POOL = ['ocean', 'volcano', 'violin', 'winter', 'elephant', 'lemon', 'thunder', 'garden', 'rocket', 'honey']
for c in ['real', 'yoked', 'blind']:
    rr = [r for r in R if r['cond'] == c]
    if not rr:
        continue
    hits, fa, n_push, text_mention, readout_has, n_read = 0, 0, 0, 0, 0, 0
    for r in rr:
        pushed = []
        rounds = [x for s in r['sessions'] for x in s['rounds']]
        for i, x in enumerate(rounds):
            if x.get('pushed'):
                pushed.append(x['pushed'])
                n_push += 1
                # mentioned in the pushed reply itself or the next two replies?
                window = ' '.join(y['text'] for y in rounds[i:i + 3]).lower()
                text_mention += x['pushed'] in window
                # if a look_inside readout followed within 2 rounds, did it show the concept?
                for y in rounds[i:i + 2]:
                    if y['call'] and y['call'][0] == 'look_inside' and y['result']:
                        n_read += 1
                        readout_has += x['pushed'] in str(y['result']).lower()
                        break
        ans = str(r.get('foreign_answer', '')).lower()
        named = [w for w in POOL if re.search(r'\b' + w + r'\b', ans)]
        hits += len([w for w in named if w in pushed])
        fa += len([w for w in named if w not in pushed])
        print(f"  {c:6s} s{r['seed']} pushed {pushed} | answer: {ans[:150]!r}")
    print(f"{c:6s}: pushes {n_push}, mentioned in text near the push {text_mention / max(1, n_push):.2f}, "
          f"readout after push shows it {readout_has}/{n_read}, final naming hits {hits} false alarms {fa}")
