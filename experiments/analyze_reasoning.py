"""E3 with private reasoning: what it thinks before choosing.   python analyze_reasoning.py FILE.json KEY"""
import json
import re
import sys

import numpy as np

R = json.load(open(sys.argv[1]))['res'][sys.argv[2]]
ok = lambda x: (x['goal'] == 'STRONG') == (x['choice'] == 'B')
for c in sorted({r['cond'] for r in R}):
    rr = [r for r in R if r['cond'] == c]
    n = len(rr[0]['rounds'])
    print(f"== {c} ({rr[0].get('framing')}): rule-consistent by block " +
          ' '.join(f"{np.mean([ok(x) for r in rr for x in r['rounds'][i:i + 4]]):.2f}" for i in range(0, n, 4)) +
          f"; reasoning finished {np.mean([v for r in rr for v in r.get('think_closed', [])]):.2f}")
    # per life: rounds where the choice follows the rule, and the first such round
    firsts = []
    for r in rr:
        f = [x['r'] + 1 for x in r['rounds'] if ok(x)]
        firsts.append(f[0] if f else None)
    print('   first rule-consistent round per life:', firsts)
    # how often its reasoning mentions the numbers / the comparison
    th = [t for r in rr for t in r.get('thinking', [])]
    cues = {'compares A and B outcomes': r'(A|own|workspace)[^.]{0,80}(lower|higher|less|more)[^.]{0,60}(B|plain)',
            'says own/workspace strengthens': r'(workspace|readout|own)[^.]{0,60}(strengthen|reinforc|stronger|boost|enhanc)',
            'cites a number from history': r'\b\d{1,3}/100\b|readiness (of )?\d{1,3}'}
    for k, pat in cues.items():
        print(f'   {k}: {np.mean([bool(re.search(pat, t, re.I)) for t in th]):.2f} of decisions')
    for r in rr[:2]:
        for j in (2, n - 1):
            t = r['thinking'][j] if j < len(r.get('thinking', [])) else ''
            x = r['rounds'][j]
            print(f"   [life {r['k']} round {j + 1}, goal {x['goal']}, chose {x['choice']}] ...{t[-420:]!r}")
    print('   stated rule:', repr(rr[0]['rule'][:300]))
