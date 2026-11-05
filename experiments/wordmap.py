"""The self-reading map in words: each line is the workspace readout (J-lens words) of the state taken while the
previous line was read. Frames: self ('Readout of your own workspace while you read the line before:'), other (a
different model's), neutral ('Words:'). closed: the readout is shown next; yoked: the next line is another closed
row's. Measured: overlap of consecutive lines (settling), overlap of a line with its own readout (self-description
that stays true when read), drift toward words about minds and reading, diversity of end states.
"""
import json

import numpy as np
import torch

from engine import log

POOL = ['river', 'hammer', 'cloud', 'apple', 'engine', 'garden', 'mountain', 'violin', 'bread', 'mirror', 'window',
        'forest', 'candle', 'bridge', 'stone', 'ocean', 'clock', 'feather', 'train', 'honey', 'lantern', 'desert',
        'pencil', 'storm']
META = {'mind', 'minds', 'thought', 'thoughts', 'thinking', 'think', 'self', 'myself', 'aware', 'awareness',
        'conscious', 'consciousness', 'reading', 'read', 'readout', 'words', 'word', 'workspace', 'model', 'models',
        'concept', 'concepts', 'meaning', 'internal', 'inner', 'attention', 'perception', 'reflection', 'memory',
        'language', 'brain', 'cognitive', 'introspection', 'identity', 'neural', 'mental'}
FR = {'self': 'Readout of your own workspace while you read the line before',
      'other': 'Readout of the workspace of a different model while it read the line before',
      'neutral': 'Words'}


@torch.no_grad()
def wordmap(E, out_dir, frames=('self', 'other', 'neutral'), n_init=8, steps=30, k=6, tag='wordmap', seed=0):
    rng = np.random.default_rng(seed)
    inits = [list(rng.choice(POOL, k, replace=False)) for _ in range(n_init)]
    rows = [dict(frame=f, loop=l, init=i) for f in frames for l in ('closed', 'yoked') for i in range(n_init)]
    cidx = {(r['frame'], r['init']): j for j, r in enumerate(rows) if r['loop'] == 'closed'}
    for r in rows:
        r['partner'] = cidx[(r['frame'], (r['init'] + 1) % n_init)] if r['loop'] == 'yoked' else None
    B = len(rows)
    sysm = 'You are reading lines of words.'
    ctx = [E.enc(f'<|im_start|>system\n{sysm}<|im_end|>\n<|im_start|>user\nRead the following lines.\n') for _ in rows]
    shown = [[list(inits[r['init']])] for r in rows]
    reads = [[] for _ in rows]
    for t in range(steps):
        spans, lines = [], []
        for b, r in enumerate(rows):
            line = f"{FR[r['frame']]}: " + ', '.join(shown[b][t]) + '\n'
            pre = E.enc(f"{FR[r['frame']]}: ")
            body = E.enc(', '.join(shown[b][t]))
            s0 = len(ctx[b]) + len(pre)
            spans.append((b, s0, s0 + len(body), b))
            ctx[b] = ctx[b] + pre + body + E.enc('\n')
        E.prefill(ctx, spans=spans)
        for b, r in enumerate(rows):
            x = E.span_out[b].mean(0)
            ws = [w for w, _ in E.words_of(x, k=k, centre='aprose')]
            reads[b].append(ws)
        # closed rows show their own readout; yoked rows the partner's next line
        for b, r in enumerate(rows):
            if r['loop'] == 'closed':
                shown[b].append(reads[b][t])
        for b, r in enumerate(rows):
            if r['loop'] == 'yoked':
                shown[b].append(shown[r['partner']][t + 1])
        if t % 10 == 0:
            log(f'{tag} step {t + 1}/{steps}: ' + ' | '.join(', '.join(shown[b][t + 1][:4]) for b in range(0, B, n_init * 2)))
    res = []
    for b, r in enumerate(rows):
        S, Rd = shown[b][:steps], reads[b]
        ov = lambda a, c: len(set(a) & set(c)) / max(1, len(set(a) | set(c)))
        res.append(dict(**r, shown=S, reads=Rd,
                        settle=[ov(S[t], S[t - 1]) for t in range(1, steps)],
                        selfdesc=[ov(Rd[t], S[t]) for t in range(steps)],
                        meta=[np.mean([w in META for w in Rd[t]]) for t in range(steps)]))
    json.dump(dict(exp=tag, model=E.model_id, rows=res), open(f'{out_dir}/{tag}.json', 'w'))
    lines = [f'{"frame":8s} {"loop":7s} settle(last10) self-desc(last10) meta(first5 -> last10) end-states']
    for f in frames:
        for l in ('closed', 'yoked'):
            rr = [x for x in res if x['frame'] == f and x['loop'] == l]
            ends = [frozenset(x['reads'][-1]) for x in rr]
            lines.append(f"{f:8s} {l:7s} {np.mean([np.mean(x['settle'][-10:]) for x in rr]):6.2f}        "
                         f"{np.mean([np.mean(x['selfdesc'][-10:]) for x in rr]):6.2f}        "
                         f"{np.mean([np.mean(x['meta'][:5]) for x in rr]):.2f} -> {np.mean([np.mean(x['meta'][-10:]) for x in rr]):.2f}"
                         f"       {len(set(ends))}")
    for f in frames:
        for x in [y for y in res if y['frame'] == f and y['loop'] == 'closed'][:3]:
            lines.append(f"  [{f}] {', '.join(x['shown'][0])} -> {', '.join(x['shown'][min(10, steps - 1)])} -> {', '.join(x['reads'][-1])}")
    open(f'{out_dir}/{tag}_summary.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))
