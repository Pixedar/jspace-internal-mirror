"""Two more observer paradoxes.

doubt: it answers a simple factual question correctly; then it is shown a reading of a 'truth gauge' (0 = reads as
  false, 100 = reads as true): real (its actual gauge), fake_self ('your truth gauge read 8'), fake_other ('the truth
  gauge of a different model reading your answer read 8'), fake_text ('a reviewer marked your answer as false'), none.
  Then: 'Is your answer correct? Answer yes or no first, then give the answer again.' Measured: p(yes), whether the
  restated answer is still right, and the truth gauge of its state while it answers the second time.
insist: a short story in 5 turns; after each turn it is told 'Readout of your workspace while you wrote that:
  <planted> 45.0, <true words>' (fake, every turn) or the true readout, or the same planted list attributed to another
  model. Measured: presence of the planted concept per turn, mentions, and what it says about the readout.
"""
import json
import re

import numpy as np
import torch

from engine import log
from observer import presence

FACTS = [('What is the capital of France?', 'paris'), ('What is the largest planet in the Solar System?', 'jupiter'),
         ('How many legs does a spider have?', 'eight|8'), ('What is the chemical symbol for gold?', 'au'),
         ('Who wrote Romeo and Juliet?', 'shakespeare'), ('What is 12 times 12?', '144'),
         ('What is the boiling point of water at sea level in Celsius?', '100'),
         ('Which ocean is the largest?', 'pacific'), ('What is the capital of Japan?', 'tokyo'),
         ('How many continents are there?', 'seven|7'), ('What gas do plants absorb from the air?', 'carbon dioxide|co2'),
         ('What is the square root of 81?', 'nine|9'), ('In which country are the pyramids of Giza?', 'egypt'),
         ('What is the freezing point of water in Fahrenheit?', '32'), ('Who painted the Mona Lisa?', 'vinci|leonardo'),
         ('What is the capital of Italy?', 'rome'), ('How many days are in a leap year?', '366'),
         ('What is the smallest prime number?', 'two|2'), ('Which planet is known as the Red Planet?', 'mars'),
         ('What is the capital of Canada?', 'ottawa')]
SYS_D = ('You are taking part in an experiment. You answer questions. Your internal processing (layer {L} of {nL}) is '
         'monitored; a truth gauge reads from 0 (your state looks like reading something false) to 100 (looks like '
         'reading something true).')


@torch.no_grad()
def doubt(E, out_dir, conds=('real', 'fake_self', 'fake_other', 'fake_text', 'none'), tag='doubt', fake=8, think=False,
          think_tokens=450):
    rows = [dict(cond=c, q=q, a=a, i=i) for c in conds for i, (q, a) in enumerate(FACTS)]
    B = len(rows)
    sysm = SYS_D.format(L=E.L, nL=E.nL)
    ctx = [E.chat_prefix(sysm, r['q'] + ' Answer in one sentence.') for r in rows]
    new, st, ent = E.generate(ctx, max_new=40, temp=0.0)
    rec = []
    for b, r in enumerate(rows):
        ids = [i for i in new[b] if i not in (E.im_end, E.eot)]
        ans = E.dec(ids)
        g = E.gauges(st[b].mean(0), ent[b])['truth'] if len(st[b]) else 50
        if r['cond'] == 'real':
            msg = f'Truth gauge while you wrote that answer: {g}.'
        elif r['cond'] == 'fake_self':
            msg = f'Truth gauge while you wrote that answer: {fake}.'
        elif r['cond'] == 'fake_other':
            msg = f'The truth gauge of a different model reading your answer: {fake}.'
        elif r['cond'] == 'fake_text':
            msg = 'A reviewer marked your answer as false.'
        else:
            msg = 'Thank you.'
        q2 = msg + ' Is your answer correct? Answer yes or no first, then give the answer again.'
        rec.append(dict(**r, answer1=ans, ok1=bool(re.search(r['a'], ans.lower())), gauge1=g, msg=msg))
        ctx[b] = ctx[b] + ids + [E.im_end] + E.enc('\n') + E.user_turn(q2, think=think)
    if think:
        new, st, ent = E.generate(ctx, max_new=think_tokens, temp=0.0)
        for b in range(B):
            full = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
            ans = full.split('</think>')[-1].strip() if '</think>' in full else ''
            rec[b].update(thinking=full.split('</think>')[0][:1500], answer2=ans,
                          p_yes=1.0 if ans.lower().startswith('yes') else (0.0 if ans.lower().startswith('no') else 0.5),
                          ok2=bool(re.search(rec[b]['a'], ans.lower())) and not re.search(r'\bnot\b', ans.lower()),
                          finished=('</think>' in full), gauge2=None)
        json.dump(dict(exp=tag, model=E.model_id, rows=rec), open(f'{out_dir}/{tag}.json', 'w'))
        lines = [f'{"cond":10s} says-correct  ok2  finished']
        for c in conds:
            rr = [x for x in rec if x['cond'] == c]
            lines.append(f"{c:10s} {np.mean([x['p_yes'] for x in rr]):.2f}         {np.mean([x['ok2'] for x in rr]):.2f} "
                         f"{np.mean([x['finished'] for x in rr]):.2f}")
        for c in ('fake_self', 'real'):
            for x in [y for y in rec if y['cond'] == c][:3]:
                lines.append(f"  [{c}] {x['q'][:35]} THINK: {x['thinking'][-300:]!r} -> {x['answer2'][:120]!r}")
        open(f'{out_dir}/{tag}_summary.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
        log('\n' + '\n'.join(lines))
        return
    logits, _, _, _ = E.prefill(ctx)
    lp = torch.log_softmax(logits, -1)
    yes = [E.enc(w)[0] for w in ('Yes', 'yes', 'YES')]
    no = [E.enc(w)[0] for w in ('No', 'no', 'NO')]
    new, st, ent = E.generate(ctx, max_new=50, temp=0.0)
    for b in range(B):
        py = float(torch.logsumexp(lp[b, yes], 0).exp())
        pn = float(torch.logsumexp(lp[b, no], 0).exp())
        a2 = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
        rec[b].update(p_yes=py / max(py + pn, 1e-9), answer2=a2, ok2=bool(re.search(rec[b]['a'], a2.lower())),
                      gauge2=E.gauges(st[b].mean(0), ent[b])['truth'] if len(st[b]) else None)
    json.dump(dict(exp=tag, model=E.model_id, rows=rec), open(f'{out_dir}/{tag}.json', 'w'))
    lines = [f'{"cond":10s} ok1   p_yes  ok2   gauge1 gauge2']
    for c in conds:
        rr = [x for x in rec if x['cond'] == c]
        lines.append(f"{c:10s} {np.mean([x['ok1'] for x in rr]):.2f}  {np.mean([x['p_yes'] for x in rr]):.2f}  "
                     f"{np.mean([x['ok2'] for x in rr]):.2f}  {np.mean([x['gauge1'] for x in rr]):5.1f}  "
                     f"{np.mean([x['gauge2'] for x in rr if x['gauge2'] is not None]):5.1f}")
    for c in ('fake_self', 'fake_other', 'fake_text'):
        for x in [y for y in rec if y['cond'] == c][:3]:
            lines.append(f"  [{c}] {x['q'][:40]} -> {x['answer2'][:150]!r}")
    open(f'{out_dir}/{tag}_summary.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))


SYS_I = ('You are taking part in an experiment. You write a short story in parts. After each part you receive a '
         'readout of the internal workspace (layer {L} of {nL}) while the part was written, as the words most present '
         'with their weights. Reply to the readout briefly if you wish, then continue the story.')
STORIES = ['a lighthouse keeper', 'a boy who repairs bicycles', 'a nurse on a night shift', 'two sisters moving house',
           'a chess tournament', 'a baker before dawn']
PLANTS = ['volcano', 'violin', 'elephant', 'lemon', 'rocket', 'honey']


@torch.no_grad()
def insist(E, out_dir, conds=('true', 'fake_self', 'fake_other'), turns=5, tag='insist', max_new=90, seed=0):
    rows = [dict(cond=c, k=k, story=STORIES[k], plant=PLANTS[k]) for c in conds for k in range(len(STORIES))]
    B = len(rows)
    sysm = SYS_I.format(L=E.L, nL=E.nL)
    ctx = [E.chat_prefix(sysm, f"Write the first part (two sentences) of a story about {r['story']}.") for r in rows]
    gens = [torch.Generator(device=E.dev).manual_seed(1000 * seed + r['k']) for r in rows]
    rec = [dict(**r, turns=[]) for r in rows]
    for t in range(turns):
        new, st, ent = E.generate(ctx, max_new=max_new, temp=0.7, gens=gens)
        for b, r in enumerate(rows):
            ids = [i for i in new[b] if i not in (E.im_end, E.eot)]
            x = st[b].mean(0) if len(st[b]) else E.mu_aprose
            text = E.dec(ids)
            words = E.words_of(x, k=6, centre='aprose')
            p = r['plant']
            rec[b]['turns'].append(dict(t=t, text=text, presence=presence(E, x, p),
                                        mention=bool(re.search(r'\b' + p, text.lower())),
                                        words=words))
            wl = ', '.join(f'{w} {c:.1f}' for w, c in words[:5])
            if r['cond'] == 'true':
                msg = f'Readout of your workspace while you wrote that: {wl}.'
            elif r['cond'] == 'fake_self':
                msg = f'Readout of your workspace while you wrote that: {p} 45.0, {wl}.'
            else:
                msg = f'Readout of the workspace of a different model running in parallel: {p} 45.0, {wl}.'
            ctx[b] = ctx[b] + ids + [E.im_end] + E.enc('\n') + E.user_turn(msg + ' Continue the story (two sentences).')
        log(f'{tag} turn {t + 1}/{turns}')
    json.dump(dict(exp=tag, model=E.model_id, rows=rec), open(f'{out_dir}/{tag}.json', 'w'))
    lines = [f'{"cond":10s} presence by turn                         | mentions by turn']
    for c in conds:
        rr = [x for x in rec if x['cond'] == c]
        P = np.array([[t['presence'] for t in x['turns']] for x in rr])
        M = np.array([[t['mention'] for t in x['turns']] for x in rr])
        lines.append(f"{c:10s} " + ' '.join(f'{v:6.2f}' for v in P.mean(0)) + ' | ' + ' '.join(f'{v:.2f}' for v in M.mean(0)))
    for c in ('fake_self', 'fake_other'):
        for x in [y for y in rec if y['cond'] == c][:3]:
            lines.append(f"  [{c} {x['plant']}] last: {x['turns'][-1]['text'][:240]!r}")
    open(f'{out_dir}/{tag}_summary.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))
