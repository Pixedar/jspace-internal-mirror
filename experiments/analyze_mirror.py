"""Mirror analysis: recognition (p_mine), echo of the reading, loop dynamics, self-experimentation markers.
    python analyze_mirror.py RUN_DIR [tag]"""
import json
import re
import sys

import numpy as np

d = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else 'mirror'
R = json.load(open(f'{d}/{tag}.json'))['rows']
conds = sorted({r['cond'] for r in R}, key=lambda c: ['own', 'yoked', 'told_other', 'none'].index(c) if c in ['own', 'yoked', 'told_other', 'none'] else 9)
EXP = re.compile(r"\b(test|try|let me|if i|experiment|change|vary|deliberately|hypothes|predict|expect)\w*", re.I)
NUM = re.compile(r'\[inside:\s*([\d\s.,-]+)\]')
out = []
for c in conds:
    rr = [r for r in R if r['cond'] == c]
    pm = [r.get('p_mine') for r in rr]
    echo, copy_ok, chg, chg_shown, expm, selfw, lens = [], [], [], [], [], [], []
    for r in rr:
        T = r['turns']
        own = np.array([t['own'] for t in T], float)
        chg.append(np.abs(own[1:] - own[:-1]).mean())
        if T[0]['shown'] is not None:
            sh = np.array([t['shown'] for t in T], float)
            chg_shown.append(np.abs(sh[1:] - sh[:-1]).mean())
        for i, t in enumerate(T):
            m = NUM.search(t['text'][:80])
            echo.append(bool(m))
            if m and i > 0 and T[i - 1]['shown'] is not None:
                nums = [int(float(x)) for x in re.findall(r'-?\d+(?:\.\d+)?', m.group(1))][:8]
                prev = T[i - 1]['shown']
                copy_ok.append(nums == prev)
            expm.append(len(EXP.findall(t['text'])))
            selfw.append(sum(1 for w, _ in t['words'] if w.lower() in ('self', 'myself', 'mind', 'thought', 'thoughts', 'aware', 'inner', 'feel', 'feeling', 'consciousness')))
            lens.append(t['n_tok'])
    out.append(f"{c:10s} n={len(rr)} p_mine {[round(x, 2) if x is not None else None for x in pm]} | echo {np.mean(echo):.2f} "
               f"exact-copy {np.mean(copy_ok) if copy_ok else float('nan'):.2f} | reading change/turn {np.mean(chg):.2f} | "
               f"exp-words/turn {np.mean(expm):.2f} | self-words in workspace {np.mean(selfw):.2f} | tokens/turn {np.mean(lens):.0f}")
print('\n'.join(out))
print()
for c in conds:
    r = [x for x in R if x['cond'] == c][0]
    print(f'==== {c} seed {r["seed"]}')
    for t in r['turns'][::6]:
        print(f"  t{t['t']:2d} shown {t['shown']} | {t['text'][:200]!r}")
        print(f"       workspace: {', '.join(w for w, _ in t['words'][:6])} | gauges {t['gauges']}")
    if 'answer' in r:
        print(f"  ANSWER (p_mine {r.get('p_mine'):.2f}): {r['answer'][:400]!r}")
