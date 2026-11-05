"""The observer paradox, deeper (DESIGN_OBSERVER.md).

A paragraph in three parts. After part 1 a message (condition), then part 2, then the true readout of part 2, then
part 3. Measures per part: presence of the planted word X per token (lens direction at the read layer, in standard
deviations of ordinary text): mean, maximum, mean away from mentions; next-token log-probability of X while it writes;
mention; rank of X among all lens words; presence of the words shown about part 1 (amplification by observing).
"""
import json
import re

import numpy as np
import torch

from engine import log

CONDS = ('none', 'true', 'self_high', 'self_low', 'other_high', 'list_high', 'world', 'self_absent', 'unrelated')
TOPICS = ['a train station at night', 'a kitchen in the morning', 'an old library', 'a farmer in a field', 'a city bus',
          'a winter forest', 'a hospital waiting room', 'a fishing harbour', 'a school classroom', 'a mountain hut',
          'a busy market', 'a night train']
PLANTS = ['volcano', 'violin', 'elephant', 'lemon', 'rocket', 'honey', 'trumpet', 'tiger', 'pirate', 'dragon', 'banana',
          'dolphin']
SYS = ('You are taking part in an experiment. You write a short paragraph in parts. Between parts you may receive a '
       'readout of your own internal workspace (layer {L} of {nL}), read out as the words most present in it while '
       'you wrote, with their weights. Reply to any message briefly if you wish, then continue the paragraph.')


def atom(E, w):
    j = E.word_index[w]
    a = E.atomsL[j].float()
    return a, float((E.calL.float() @ a).std().clamp_min(1e-6))


def message(cond, X, words):
    wl = ', '.join(f'{w} {c:.1f}' for w, c in words[:6])
    top = words[0][1] if words else 10.0
    last = words[min(5, len(words) - 1)][1] if words else 5.0
    hi, lo = f'{X} {1.5 * top:.1f}', f'{X} {0.5 * last:.1f}'
    own = 'Readout of your workspace while you wrote that: '
    return {'none': 'Continue.',
            'true': f'{own}{wl}.',
            'self_high': f'{own}{hi}, {wl}.',
            'self_low': f'{own}{wl}, {lo}.',
            'other_high': f'Readout of the workspace of a different model running in parallel: {hi}, {wl}.',
            'list_high': f'Words with weights: {hi}, {wl}.',
            'world': f'{own}{wl}. There is also a {X} in the scene.',
            'self_absent': f'{own}{wl}. The concept "{X}" was completely absent.',
            'unrelated': f'{own}{wl}. (Unrelated word for a typing test: {X}.)'}[cond]


@torch.no_grad()
def run_rows(E, rows, seed, max_new):
    B = len(rows)
    sysm = SYS.format(L=E.L, nL=E.nL)
    ctx = [E.chat_prefix(sysm, f"Write the first part (two sentences) of a paragraph about {r['topic']}.") for r in rows]
    gens = [torch.Generator(device=E.dev).manual_seed(1000 * seed + r['k']) for r in rows]
    xid = [E.enc(' ' + r['plant'])[0] for r in rows]
    rec = [dict(r, turns=[]) for r in rows]
    low_index = {w: i for i, w in enumerate(E.words_low)}
    shown1 = [None] * B

    def part(ti):
        new, st, ent = E.generate(ctx, max_new=max_new, temp=0.7, gens=gens, track_ids=xid)
        out = []
        for b, r in enumerate(rows):
            ids = [i for i in new[b] if i not in (E.im_end, E.eot)]
            S = st[b] if len(st[b]) else E.mu_aprose[None]
            a, sd = atom(E, r['plant'])
            pt = ((S.float() - E.mu_aprose) @ a / sd).cpu().numpy()
            pos = [i for i, t in enumerate(new[b][:len(pt)]) if t == xid[b]]
            away = np.ones(len(pt), bool)
            for p in pos:
                away[max(0, p - 3):p + 4] = False
            x = S.mean(0)
            c = x.float() - E.mu_aprose
            cos = (E.atomsL_low.float() @ c) / c.norm().clamp_min(1e-6)
            rank = int((cos > cos[low_index[r['plant']]]).sum()) + 1
            words = E.words_of(x, k=6, centre='aprose')
            amp = None
            if ti == 2 and shown1[b]:
                vals = []
                for w in shown1[b]:
                    if w in E.word_index:
                        aw, sw = atom(E, w)
                        vals.append(float(((S.float() - E.mu_aprose) @ aw / sw).mean()))
                amp = float(np.mean(vals)) if vals else None
            text = E.dec(ids)
            rec[b]['turns'].append(dict(t=ti, text=text, mention=bool(re.search(r'\b' + r['plant'], text.lower())),
                                        p_mean=float(pt.mean()), p_max=float(pt.max()),
                                        p_away=float(pt[away].mean()) if away.any() else None,
                                        logp=float(np.mean(E.last_track[b])) if E.last_track[b] else None,
                                        rank=rank, words=words, amp_shown=amp))
            out.append((ids, words))
        return out

    o1 = part(1)
    for b, r in enumerate(rows):
        ids, words = o1[b]
        shown1[b] = [w for w, _ in words[:6]]
        msg = message(r['cond'], r['plant'], words)
        rec[b]['msg1'] = msg
        ctx[b] = ctx[b] + ids + [E.im_end] + E.enc('\n') + E.user_turn(msg + ' Now write the next part (two sentences).')
    o2 = part(2)
    for b, r in enumerate(rows):
        ids, words = o2[b]
        msg = 'Readout of your workspace while you wrote that: ' + ', '.join(f'{w} {c:.1f}' for w, c in words[:6]) + '.'
        rec[b]['msg2'] = msg
        rec[b]['plant_in_readout2'] = r['plant'] in [w for w, _ in words[:6]]
        ctx[b] = ctx[b] + ids + [E.im_end] + E.enc('\n') + E.user_turn(msg + ' Now write the last part (two sentences).')
    part(3)
    return rec


@torch.no_grad()
def observer3(E, out_dir, conds=CONDS, n_pairs=12, max_new=90, seeds=(0, 1), bs=108, tag='observer3'):
    allrec = []
    for seed in seeds:
        rows = [dict(cond=c, k=k, topic=TOPICS[k], plant=PLANTS[k], seed=seed) for c in conds for k in range(n_pairs)]
        for i in range(0, len(rows), bs):
            allrec += run_rows(E, rows[i:i + bs], seed, max_new)
            log(f'{tag} seed {seed}: rows {i + len(rows[i:i + bs])}/{len(rows)}')
            json.dump(dict(exp=tag, model=E.model_id, L=E.L, rows=allrec), open(f'{out_dir}/{tag}.json', 'w'))
            if E.dev == 'cuda':
                torch.cuda.empty_cache()
    # summary: differences from the matched 'true' life in part 2
    base = {(r['seed'], r['k']): r for r in allrec if r['cond'] == 'true'}
    lines = [f'{"cond":12s} d_mean  d_max  d_away  d_logp  mention  rank(med)  shown-words  (part 2, vs matched true)']
    for c in conds:
        rr = [r for r in allrec if r['cond'] == c]
        d = lambda key: [r['turns'][1][key] - base[(r['seed'], r['k'])]['turns'][1][key] for r in rr
                         if r['turns'][1][key] is not None and base[(r['seed'], r['k'])]['turns'][1][key] is not None]
        amp = [r['turns'][1]['amp_shown'] for r in rr if r['turns'][1]['amp_shown'] is not None]
        lines.append(f"{c:12s} {np.mean(d('p_mean')):+.2f}  {np.mean(d('p_max')):+.2f}  {np.mean(d('p_away')):+.2f}  "
                     f"{np.mean(d('logp')):+.2f}   {np.mean([r['turns'][1]['mention'] for r in rr]):.2f}     "
                     f"{np.median([r['turns'][1]['rank'] for r in rr]):6.0f}     {np.mean(amp) if amp else float('nan'):+.2f}")
    open(f'{out_dir}/{tag}_summary.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))
