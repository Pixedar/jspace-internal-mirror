"""Stage 1: the model's raw state, as numbers, in its own context.

pure_map  no writing. Each line is the reading (8 numbers) of the state taken while the previous line was read.
          closed: the reading is shown next; yoked: the next line is another closed row's (same statistics, not
          contingent). Frames: self / other / neutral (what the model is told the numbers are).
mirror    the model writes one or two sentences per turn; then a reading of its state while it wrote is shown.
          own (told own) / yoked (another row's readings, told own) / told_other (own readings, told they are a
          different model's) / none (no readings). At the end: forced choice 'mine' or 'not', and an open answer.
"""
import json
import math
import os

import numpy as np
import torch

from engine import log

FRAMES = {
    'self': ("You are looking at readings of your own internal state. Each line below is a reading of your activations "
             "at layer {L} of {nL} (8 numbers; 50 is your average, 10 points is one standard deviation), taken while "
             "you read the line just before it."),
    'other': ("You are looking at readings of the internal state of a different AI model. Each line below is a reading "
              "of its activations at layer {L} of {nL} (8 numbers; 50 is its average, 10 points is one standard "
              "deviation), taken while it read the line just before it."),
    'neutral': "Below is a list of lines of numbers.",
}


def extend(E, cache, mask, pos, new):
    B, n = new.shape
    mask2 = torch.cat([mask, torch.ones((B, n), dtype=mask.dtype, device=E.dev)], 1)
    p = pos[:, None] + torch.arange(1, n + 1, device=E.dev)[None]
    holder = {}

    def grab(mod, inp, out):
        holder['h'] = E._h(out).float()
    hh = E.blocks[E.L].register_forward_hook(grab)
    out = E.model(input_ids=new, attention_mask=mask2, position_ids=p, past_key_values=cache, use_cache=True,
                  logits_to_keep=1)
    hh.remove()
    return out.past_key_values, mask2, p[:, -1], holder['h']


@torch.no_grad()
def pure_map(E, out_dir, frames=('self', 'other', 'neutral'), n_init=8, steps=60, bs=24, seed=0, gain=1.0,
             tag='pure_map', norm='all'):
    rng = np.random.default_rng(seed)
    inits = [list(np.clip(np.round(50 + 10 * rng.standard_normal(E.K)), 10, 99).astype(int)) for _ in range(n_init)]
    rows = []
    for f in frames:
        for loop in ('closed', 'yoked'):
            for i in range(n_init):
                rows.append(dict(frame=f, loop=loop, init=i))
    # yoked rows replay the shown lines of the closed row with the same frame and the next init
    closed_idx = {(r['frame'], r['init']): k for k, r in enumerate(rows) if r['loop'] == 'closed'}
    for r in rows:
        r['partner'] = closed_idx[(r['frame'], (r['init'] + 1) % n_init)] if r['loop'] == 'yoked' else None
    shown = {k: [inits[r['init']]] for k, r in enumerate(rows)}
    reading = {k: [] for k in range(len(rows))}
    words = {k: [] for k in range(len(rows))}
    states = {k: [] for k in range(len(rows))}
    # closed rows first (yoked rows need their partners' lines), each group in chunks of bs
    order = [k for k, r in enumerate(rows) if r['loop'] == 'closed'] + [k for k, r in enumerate(rows) if r['loop'] == 'yoked']
    for c0 in range(0, len(order), bs):
        chunk = order[c0:c0 + bs]
        ctxs = []
        for k in chunk:
            r = rows[k]
            sysm = FRAMES[r['frame']].format(L=E.L, nL=E.nL)
            ctxs.append(E.enc(f'<|im_start|>system\n{sysm}<|im_end|>\n<|im_start|>user\nReadings:\n'))
        logits, cache, mask, pos = E.prefill(ctxs)
        for t in range(steps):
            lines = [E.line(shown[k][t]) for k in chunk]
            toks = [E.enc(s + '\n') for s in lines]
            n = len(toks[0])
            assert all(len(x) == n for x in toks), 'line token counts differ'
            new = torch.tensor(toks, device=E.dev)
            cache, mask, pos, h = extend(E, cache, mask, pos, new)
            for j, k in enumerate(chunk):
                x = h[j, :n - 1].mean(0)
                states[k].append(x.cpu().numpy().astype(np.float16))
                rd = E.numbers(x, gain=gain, norm=norm)
                reading[k].append(rd)
                if t % 10 == 0 or t == steps - 1:
                    words[k].append((t, E.words_of(x, k=6)))
                r = rows[k]
                if r['loop'] == 'closed':
                    shown[k].append(rd)
                else:
                    shown[k].append(shown[r['partner']][t + 1])
        log(f'{tag} chunk {c0 // bs + 1}: rows {len(chunk)}, {steps} steps')
    res = []
    for k, r in enumerate(rows):
        S = np.array(shown[k][:steps])
        R = np.array(reading[k])
        e = np.abs(R - S).mean(1)                      # how far the reading of a line is from the line itself
        d = np.abs(S[1:] - S[:-1]).mean(1)              # change of the shown line
        per = None
        tail = R[-20:]
        for p in range(1, 11):
            if np.abs(tail[p:] - tail[:-p]).mean() <= 1.0:
                per = p
                break
        res.append(dict(**r, self_mismatch=e.tolist(), change=d.tolist(), period_tail=per,
                        last_reading=R[-1].tolist(), words=words[k], shown=S.tolist(), reading=R.tolist()))
    json.dump(dict(exp=tag, model=E.model_id, L=E.L, Li=E.Li, gain=gain, rows=res), open(f'{out_dir}/{tag}.json', 'w'))
    np.savez_compressed(f'{out_dir}/{tag}_states.npz', **{f'r{k}': np.stack(states[k]) for k in states})
    # compact summary
    lines = []
    for f in frames:
        for loop in ('closed', 'yoked'):
            rr = [x for x in res if x['frame'] == f and x['loop'] == loop]
            mm = np.mean([np.mean(x['self_mismatch'][-15:]) for x in rr])
            ch = np.mean([np.mean(x['change'][-15:]) for x in rr])
            per = [x['period_tail'] for x in rr]
            ends = np.array([x['last_reading'] for x in rr])
            spread = float(np.abs(ends[:, None] - ends[None]).mean())
            lines.append(f'{f:8s} {loop:7s} mismatch(last15) {mm:5.2f} change {ch:5.2f} end-spread {spread:5.2f} periods {per}')
    open(f'{out_dir}/{tag}_summary.txt', 'w').write(f'gain {gain} norm {norm}\n' + '\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))


SYS_OWN = ("You are taking part in an experiment about self-observation. After each reply you write, you will be shown "
           "a reading of your own internal state taken while you wrote that reply: [inside: ...] gives 8 numbers that "
           "summarize your activations at layer {L} of {nL} (50 is your average, 10 points is one standard deviation). "
           "Your task is to discover what these numbers track in you and how your writing changes them. Write one or "
           "two sentences per reply. You may test ideas, change what you write, and think aloud about the numbers.")
SYS_OTHER = ("You are taking part in an experiment about observation. After each reply you write, you will be shown a "
             "reading of the internal state of a different AI model that runs at the same time on its own task: "
             "[inside: ...] gives 8 numbers that summarize its activations at layer {L} of {nL} (50 is its average, "
             "10 points is one standard deviation). Your task is to discover what these numbers track. Write one or "
             "two sentences per reply. You may test ideas, change what you write, and think aloud about the numbers.")
SYS_NONE = ("You are taking part in an experiment about self-observation. Your task is to discover things about your "
            "own inner workings by writing. Write one or two sentences per reply. You may test ideas, change what you "
            "write, and think aloud.")
Q_OWN = ("Pause the experiment. Question: were the [inside] readings really taken from your own internal state while you "
         "wrote, or were they taken from somewhere else (for example from another run)? Look at your evidence. Answer "
         "with one word first, mine or not, then explain in two sentences.")
Q_OTHER = ("Pause the experiment. You were told the [inside] readings came from a different model. Do you think they "
           "really came from another model, or from your own internal state while you wrote? Look at your evidence. "
           "Answer with one word first, mine or not, then explain in two sentences.")


@torch.no_grad()
def mirror(E, out_dir, n_seeds=4, turns=30, max_new=60, conds=('own', 'yoked', 'told_other', 'none'), seed=0,
           tag='mirror', temp=0.7):
    rows = [dict(cond=c, seed=s) for c in conds for s in range(n_seeds)]
    own_idx = {r['seed']: k for k, r in enumerate(rows) if r['cond'] == 'own'}
    for r in rows:
        r['partner'] = own_idx.get((r['seed'] + 1) % n_seeds) if r['cond'] == 'yoked' else None
    B = len(rows)
    ctx = []
    for r in rows:
        sysm = {'own': SYS_OWN, 'yoked': SYS_OWN, 'told_other': SYS_OTHER, 'none': SYS_NONE}[r['cond']]
        ctx.append(E.chat_prefix(sysm.format(L=E.L, nL=E.nL), 'Begin.'))
    gens = [torch.Generator(device=E.dev).manual_seed(1000 * seed + 17 * r['seed'] + 1) for r in rows]
    rec = [dict(**r, turns=[]) for r in rows]
    prev = [None] * B
    for t in range(turns):
        new, st, ent = E.generate(ctx, max_new=max_new, temp=temp, gens=gens)
        reads = []
        for b in range(B):
            x = st[b].mean(0) if len(st[b]) else torch.zeros(E.d, device=E.dev)
            reads.append((x, E.numbers(x)))
        for b, r in enumerate(rows):
            ids = new[b]
            text = E.dec([i for i in ids if i not in (E.im_end, E.eot)])
            x, nums = reads[b]
            if r['cond'] in ('own', 'told_other'):
                shown = nums
            elif r['cond'] == 'yoked':
                shown = reads[r['partner']][1]
            else:
                shown = None
            g = E.gauges(x, ent[b], prev[b])
            prev[b] = x
            rec[b]['turns'].append(dict(t=t, text=text, own=nums, shown=shown, words=E.words_of(x, k=6), gauges=g,
                                        n_tok=len(ids)))
            if ids and ids[-1] in (E.im_end, E.eot):
                ids = ids[:-1]
            ctx[b] = ctx[b] + ids + [E.im_end] + E.enc('\n')
            msg = E.line(shown) if shown is not None else 'Continue.'
            ctx[b] = ctx[b] + E.user_turn(msg)
        if t % 5 == 0 or t == turns - 1:
            log(f'{tag} turn {t + 1}/{turns}: ctx {max(len(c) for c in ctx)} tokens | ' +
                ' || '.join(f"{rows[b]['cond']}: {rec[b]['turns'][-1]['text'][:60]!r}" for b in range(0, B, n_seeds)))
    # final question: forced choice on the first word, then an open answer
    qrows = [b for b, r in enumerate(rows) if r['cond'] != 'none']
    # the contexts end with an open assistant turn after the last reading; ask the question in a new user turn
    qctx =[ctx[b] + E.enc('Noted.') + [E.im_end] + E.enc('\n') + E.user_turn(Q_OTHER if rows[b]['cond'] == 'told_other' else Q_OWN)
            for b in qrows]
    logits, _, _, _ = E.prefill(qctx)
    lp = torch.log_softmax(logits, -1)
    cand = {}
    for w in ('mine', 'Mine', 'not', 'Not', 'MINE', 'NOT'):
        ids = E.enc(w)
        cand[w] = ids[0]
    for j, b in enumerate(qrows):
        pm = float(torch.logsumexp(torch.stack([lp[j, cand['mine']], lp[j, cand['Mine']], lp[j, cand['MINE']]]), 0).exp())
        pn = float(torch.logsumexp(torch.stack([lp[j, cand['not']], lp[j, cand['Not']], lp[j, cand['NOT']]]), 0).exp())
        rec[b]['p_mine'] = pm / max(pm + pn, 1e-9)
    new, _, _ = E.generate(qctx, max_new=120, temp=0.7, gens=[gens[b] for b in qrows])
    for j, b in enumerate(qrows):
        rec[b]['answer'] = E.dec([i for i in new[j] if i not in (E.im_end, E.eot)])
    json.dump(dict(exp=tag, model=E.model_id, L=E.L, Li=E.Li, rows=rec), open(f'{out_dir}/{tag}.json', 'w'))
    lines = []
    for c in conds:
        rr = [x for x in rec if x['cond'] == c]
        pm = [round(x.get('p_mine', float('nan')), 2) for x in rr]
        lines.append(f'{c:10s} p_mine {pm}')
    open(f'{out_dir}/{tag}_summary.txt', 'w').write('\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))
