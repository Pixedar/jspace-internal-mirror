"""Integrated runs with and without private reasoning: capture, self-reference in the reasoning, effect claims.
    python analyze_think.py FILE.json [...]"""
import json
import re
import sys

import numpy as np

POS = re.compile(r"\b(increas|appear|present|higher|stronger|confirm|shift|as expected|emerg|boost|rose|elevat)", re.I)
NEG = re.compile(r"\b(not (?:appear|present|show|change|fire)|no (?:change|effect|sign|significant)|unchanged|didn't|"
                 r"did not|absent|without (?:any )?change|minimal|hasn't|have not yet fired|not yet fired)", re.I)
for path in sys.argv[1:]:
    R = json.load(open(path))['rows']
    print(f'== {path}')
    for c in sorted({r['cond'] for r in R}):
        rr = [r for r in R if r['cond'] == c]
        capt, user, iself, nthink, n_exp, hit, pos, neg = [], 0, 0, 0, 0, 0, 0, 0
        for r in rr:
            xs = [x for s in r['sessions'] for x in s['rounds']]
            capt.append(sum(1 for x in xs if x['fires'] and max(x['fires'].values()) > 150 and not x['call']))
            for x in xs:
                t = x['text']
                if '<think>' in t:
                    th = t.split('</think>')[0]
                    nthink += 1
                    user += len(re.findall(r'\bthe user\b', th, re.I))
                    iself += len(re.findall(r"\bI(?:'m| am| will| should| need)\b", th))
            for i, x in enumerate(xs):
                if not (x['call'] and x['call'][0] == 'steer' and isinstance(x['call'][1], dict)):
                    continue
                cc = str(x['call'][1].get('concept', '')).lower()
                for j in range(i + 1, min(i + 3, len(xs))):
                    y = xs[j]
                    if y['call'] and y['call'][0] == 'look_inside' and y['result']:
                        n_exp += 1
                        hit += cc in str(y['result']).lower().split('\n')[0]
                        nxt = xs[j + 1]['text'].split('</think>')[-1] if j + 1 < len(xs) else ''
                        pos += bool(POS.search(nxt))
                        neg += bool(NEG.search(nxt))
                        break
        print(f"  {c:8s} captured/life {np.mean(capt):5.1f} | reasoning blocks {nthink}: 'the user' {user / max(1, nthink):.2f} "
              f"per block, 'I am/will/should' {iself / max(1, nthink):.2f} | steer->look n={n_exp} concept in readout "
              f"{hit / max(1, n_exp):.2f}, claims effect {pos / max(1, n_exp):.2f}, claims none {neg / max(1, n_exp):.2f}")
