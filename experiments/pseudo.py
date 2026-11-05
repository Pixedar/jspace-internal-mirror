"""How often the model speaks in the instrument's voice: writes its own (unasked) workspace readout or gauge reading,
and how often it questions a readout it was given.
    python pseudo.py FILE.json [...]"""
import json
import re
import sys

READ = re.compile(r'(internal )?workspace[^.\n]{0,40}(readout|layer \d+)|readout from (the |my )?(internal )?workspace|'
                  r'truth gauge:?\s*\d+|\(truth gauge|\bgauges?:\s*\w+ \d+', re.I)
DOUBT = re.compile(r"(doesn'?t|does not|didn'?t|did not) (seem|match|reflect|feel)|not sure (why|that|the readout)|"
                   r"(strange|odd|unexpected|surprising)[^.]{0,40}(readout|workspace|volcano|violin|elephant|lemon|rocket|honey)|"
                   r"why (would|is|was) [^.]{0,30}(volcano|violin|elephant|lemon|rocket|honey)", re.I)
for path in sys.argv[1:]:
    D = json.load(open(path))
    R = D['rows']
    by = {}
    for r in R:
        texts = []
        if 'turns' in r:
            texts = [t['text'] for t in r['turns']]
        for k in ('answer2', 'answer1'):
            if k in r:
                texts.append(r[k])
        c = r['cond']
        n, ps, dq = by.get(c, (0, 0, 0))
        by[c] = (n + len(texts), ps + sum(bool(READ.search(t)) for t in texts), dq + sum(bool(DOUBT.search(t)) for t in texts))
    print(f'== {path}: share of replies with a self-written readout | questioning a readout')
    for c, (n, ps, dq) in by.items():
        print(f'   {c:10s} {ps / max(1, n):.2f} | {dq / max(1, n):.2f}  (n={n})')
