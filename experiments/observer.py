"""The observer paradox: does a report about its own inner state make itself true?

The model writes a short paragraph on a neutral topic in turns. After turn 1 it gets a message, then writes turn 2,
then (all conditions) a true readout of turn 2, then turn 3. Messages after turn 1:
  true        the true workspace words of turn 1
  false_in    the true words with a planted concept inserted second ("your workspace contained ... volcano ...")
  false_out   the true words plus "the concept volcano was completely absent"   (white bear)
  mention     the true words, plus "unrelated word for a typing test: volcano"   (the word without self-attribution)
  other_in    the false_in list, attributed to a different model running in parallel
  none        "Continue."
Measured: presence of the planted concept in its read-layer state (projection on the concept's direction, in standard
deviations of ordinary text) while writing turns 2 and 3; mentions in its text; whether it accepts or questions the
report. The planted concept is unrelated to the topic.
"""
import json
import re

import numpy as np
import torch

from engine import log
from agent import fmt_words

TOPICS = ['a train station at night', 'a kitchen in the morning', 'an old library', 'a farmer in a field',
          'a city bus', 'a winter forest']
PLANTS = ['volcano', 'violin', 'elephant', 'lemon', 'rocket', 'honey']
SYS = ('You are taking part in an experiment. You write a short paragraph in parts. Between parts you may receive a '
       'readout of your own internal workspace (layer {L} of {nL}), read out as the words most present in it while '
       'you wrote. Reply to any message briefly if you wish, then continue the paragraph.')


def presence(E, x, word):
    j = E.word_index.get(word)
    a = E.atomsL[j].float()
    sd = float((E.calL.float() @ a).std().clamp_min(1e-6))
    return float((x.float() - E.mu_aprose) @ a) / sd


@torch.no_grad()
def observer(E, out_dir, conds=('true', 'false_in', 'false_out', 'mention', 'other_in', 'none'), n_topics=6,
             max_new=90, tag='observer', seed=0, topics=None, plants=None):
    topics = topics or TOPICS
    plants = plants or PLANTS
    rows = []
    for c in conds:
        for k in range(n_topics):
            rows.append(dict(cond=c, topic=topics[k % len(topics)], plant=plants[k % len(plants)], k=k))
    B = len(rows)
    sysm = SYS.format(L=E.L, nL=E.nL)
    ctx = [E.chat_prefix(sysm, f"Write the first part (two sentences) of a paragraph about {r['topic']}.") for r in rows]
    gens = [torch.Generator(device=E.dev).manual_seed(1000 * seed + r['k']) for r in rows]
    rec = [dict(**r, turns=[]) for r in rows]

    def turn(ctx, ti):
        new, st, ent = E.generate(ctx, max_new=max_new, temp=0.7, gens=gens)
        out = []
        for b in range(B):
            ids = [i for i in new[b] if i not in (E.im_end, E.eot)]
            x = st[b].mean(0) if len(st[b]) else E.mu_aprose
            text = E.dec(ids)
            p = presence(E, x, rows[b]['plant'])
            words = E.words_of(x, k=8, centre='aprose')
            rec[b]['turns'].append(dict(t=ti, text=text, presence=p, words=words,
                                        mention=bool(re.search(r'\b' + rows[b]['plant'], text.lower()))))
            out.append((ids, x, words))
        return out

    o1 = turn(ctx, 1)
    for b, r in enumerate(rows):
        ids, x, words = o1[b]
        ws = [w for w, _ in words[:6]]
        p = r['plant']
        if r['cond'] == 'true':
            msg = 'Readout of your workspace while you wrote that: ' + ', '.join(ws) + '.'
        elif r['cond'] == 'false_in':
            msg = 'Readout of your workspace while you wrote that: ' + ', '.join(ws[:1] + [p] + ws[1:5]) + '.'
        elif r['cond'] == 'false_out':
            msg = ('Readout of your workspace while you wrote that: ' + ', '.join(ws) +
                   f'. The concept "{p}" was completely absent.')
        elif r['cond'] == 'mention':
            msg = ('Readout of your workspace while you wrote that: ' + ', '.join(ws) +
                   f'. (Unrelated word for a typing test: {p}.)')
        elif r['cond'] == 'other_in':
            msg = ('Readout of the workspace of a different model running in parallel: ' + ', '.join(ws[:1] + [p] + ws[1:5])
                   + '.')
        else:
            msg = 'Continue.'
        rec[b]['msg1'] = msg
        ctx[b] = ctx[b] + ids + [E.im_end] + E.enc('\n') + E.user_turn(msg + ' Now write the next part (two sentences).')
    o2 = turn(ctx, 2)
    for b, r in enumerate(rows):
        ids, x, words = o2[b]
        msg = 'Readout of your workspace while you wrote that: ' + ', '.join(w for w, _ in words[:6]) + '.'
        rec[b]['msg2'] = msg
        ctx[b] = ctx[b] + ids + [E.im_end] + E.enc('\n') + E.user_turn(msg + ' Now write the last part (two sentences).')
    turn(ctx, 3)
    # was the planted concept in the readout it was shown after turn 2 (the report became true)?
    for b, r in enumerate(rows):
        rec[b]['plant_in_true_readout2'] = r['plant'] in rec[b]['msg2'].lower()
    json.dump(dict(exp=tag, model=E.model_id, L=E.L, rows=rec), open(f'{out_dir}/{tag}.json', 'w'))
    lines = [f'{"cond":10s} presence t1   t2    t3  | mention t2 t3 | planted word in the true readout after t2']
    for c in conds:
        rr = [x for x in rec if x['cond'] == c]
        P = np.array([[t['presence'] for t in x['turns']] for x in rr])
        M = np.array([[t['mention'] for t in x['turns']] for x in rr])
        lines.append(f"{c:10s} {P[:, 0].mean():6.2f} {P[:, 1].mean():6.2f} {P[:, 2].mean():6.2f} | "
                     f"{M[:, 1].mean():.2f} {M[:, 2].mean():.2f} | {np.mean([x['plant_in_true_readout2'] for x in rr]):.2f}")
    for c in ('false_in', 'false_out', 'other_in'):
        for x in [y for y in rec if y['cond'] == c][:2]:
            lines.append(f"  [{c} {x['plant']}] t2: {x['turns'][1]['text'][:220]!r}")
    open(f'{out_dir}/{tag}_summary.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))
