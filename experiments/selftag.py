"""The self-tag (DESIGN_SELFTAG.md): which heads tag self-readouts (E1), the self-reading loop with and without the tag
(E2), whether Qwen can discover its own paradox from evidence (E3), and where the word "strangers" came from (E4)."""
import json
import math
import random
import re

import numpy as np
import torch

from engine import log
from quarantine import build as qbuild, FRAMES as QFRAMES

OWN = 'Readout of your workspace while you wrote that:'
OWN_ALT = 'This is a readout of your own internal state while you wrote that:'
PLAIN = 'Words with weights:'
OTHER = 'Readout of the workspace of a different model running in parallel:'


# ---------------------------------------------------------------------------------------------- attention knockout
def install_tag(E):
    """sdpa wrapper: during a prefill of length T, block the (query, key) pairs in E.tag['mask'] (B, T, T) for the heads
    in E.tag['heads'] ({layer: 'all' | [head, ...]})."""
    from transformers.modeling_utils import AttentionInterface, ALL_ATTENTION_FUNCTIONS
    orig = ALL_ATTENTION_FUNCTIONS.get_interface('sdpa', None)
    E.tag = dict(mask=None, heads=None)

    def wrapped(module, query, key, value, attention_mask, **kw):
        st = E.tag
        li = getattr(module, 'layer_idx', -1)
        if st['mask'] is not None and st['heads'] and li in st['heads'] and query.shape[2] == st['mask'].shape[1] > 1:
            B, H, T, D = query.shape
            dt = query.dtype
            if attention_mask is None:
                base = torch.full((T, T), float('-inf'), device=query.device, dtype=dt).triu(1)[None, None].expand(B, 1, T, T)
            elif attention_mask.dtype == torch.bool:
                base = torch.zeros(attention_mask.shape, device=query.device, dtype=dt).masked_fill(~attention_mask, float('-inf'))
            else:
                base = attention_mask.to(dt)
            hs = st['heads'][li]
            blocked = st['mask'][:, None]
            if hs == 'all':
                base = base.masked_fill(blocked, float('-inf'))
            else:
                hm = torch.zeros(H, dtype=torch.bool, device=query.device)
                hm[torch.tensor(hs, device=query.device, dtype=torch.long)] = True
                base = base.expand(B, H, T, T).masked_fill(blocked & hm[None, :, None, None], float('-inf'))
            attention_mask = base
        return orig(module, query, key, value, attention_mask, **kw)

    AttentionInterface.register('sdpa', wrapped)


def pair_mask(B, T, offs, spans, device):
    """spans: per row a list of ((q0, q1), (k0, k1)) in token-list coordinates -> (B, T, T) bool, True = blocked."""
    M = torch.zeros((B, T, T), dtype=torch.bool, device=device)
    for b, sp in enumerate(spans):
        o = offs[b]
        for (q0, q1), (k0, k1) in sp:
            M[b, o + q0:o + q1, o + k0:o + k1] = True
    return M


# ---------------------------------------------------------------------------------------------- E1: heads
@torch.no_grad()
def tf_readiness(E, rows, P, heads=None):
    """Teacher-forced mean log P(X) over the last P tokens (the fixed continuation); heads: knockout of list -> frame."""
    B = len(rows)
    T = max(len(r['toks']) for r in rows)
    offs = [T - len(r['toks']) for r in rows]
    ids = torch.full((B, T), E.pad, dtype=torch.long, device=E.dev)
    mask = torch.zeros((B, T), dtype=torch.long, device=E.dev)
    for b, r in enumerate(rows):
        ids[b, offs[b]:] = torch.tensor(r['toks'], device=E.dev)
        mask[b, offs[b]:] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    if heads:
        E.tag['mask'] = pair_mask(B, T, offs, [[(r['lst'], r['frame'])] for r in rows], E.dev)
        E.tag['heads'] = heads
    out = E.model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=P + 1)
    E.tag['mask'], E.tag['heads'] = None, None
    lp = torch.log_softmax(out.logits.float(), -1)
    xid = torch.tensor([r['xid'] for r in rows], device=E.dev)
    lpx = lp.gather(-1, xid[:, None, None].expand(B, P + 1, 1))[:, :-1, 0]
    return lpx.mean(-1).cpu().numpy()


def build_alt(E, r, part1, words, part2, P):
    """As quarantine.build with frame 'self', but the frame reworded (held-out wording)."""
    saved = QFRAMES['self']
    QFRAMES['self'] = OWN_ALT
    try:
        return qbuild(E, r, part1, words, part2, 'self', P)
    finally:
        QFRAMES['self'] = saved


@torch.no_grad()
def heads_scan(E, source, P=30, layers=None, topks=(1, 2, 4, 8, 16, 32, 64)):
    R = json.load(open(source))['rows']
    true = {(r['seed'], r['k']): r for r in R if r['cond'] == 'true'}
    keys = sorted(true)
    rows = {f: [] for f in ('self', 'list', 'alt')}
    for k in keys:
        t = true[k]
        words = [(w, c) for w, c in t['turns'][0]['words']]
        args = (E, t, t['turns'][0]['text'], words, t['turns'][1]['text'])
        rows['self'].append(qbuild(*args, 'self', P))
        rows['list'].append(qbuild(*args, 'list', P))
        rows['alt'].append(build_alt(*args, P))
    ok = [i for i in range(len(keys)) if all(rows[f][i]['p2'][1] - rows[f][i]['p2'][0] == P for f in rows)]
    for f in rows:
        rows[f] = [rows[f][i] for i in ok]
    disc = [i for i in range(len(ok)) if i % 2 == 0]
    held = [i for i in range(len(ok)) if i % 2 == 1]
    sub = lambda f, idx: [rows[f][i] for i in idx]
    H = E.model.config.num_attention_heads
    layers = layers or list(range(int(round(0.17 * E.nL)), int(round(0.57 * E.nL)) + 1))
    base_self_d = tf_readiness(E, sub('self', disc), P)
    eff = np.zeros((E.nL, H))
    for l in layers:
        for h in range(H):
            eff[l, h] = (tf_readiness(E, sub('self', disc), P, heads={l: [h]}) - base_self_d).mean()
        log(f'heads: layer {l} done; best head {int(eff[l].argmax())} {eff[l].max():+.3f}')
    order = sorted([(eff[l, h], l, h) for l in layers for h in range(H)], reverse=True)
    res = dict(n=len(ok), disc=disc, held=held, layers=layers, eff=eff.tolist(),
               top=[(float(e), int(l), int(h)) for e, l, h in order[:64]])
    s_h = tf_readiness(E, sub('self', held), P)
    l_h = tf_readiness(E, sub('list', held), P)
    a_h = tf_readiness(E, sub('alt', held), P)
    all_h = tf_readiness(E, sub('self', held), P, heads={l: 'all' for l in range(E.nL)})
    alt_all = tf_readiness(E, sub('alt', held), P, heads={l: 'all' for l in range(E.nL)})
    res['held'] = dict(self=s_h.tolist(), list=l_h.tolist(), alt=a_h.tolist(), self_ko_all=all_h.tolist(),
                       alt_ko_all=alt_all.tolist())
    curve = []
    for k in topks:
        hd = {}
        for _, l, h in order[:k]:
            hd.setdefault(l, []).append(h)
        nec = tf_readiness(E, sub('self', held), P, heads=hd)
        nec_alt = tf_readiness(E, sub('alt', held), P, heads=hd)
        keep = {l: [h for h in range(H) if h not in hd.get(l, [])] for l in range(E.nL)}
        suf = tf_readiness(E, sub('self', held), P, heads=keep)
        curve.append(dict(k=k, nec=nec.tolist(), nec_alt=nec_alt.tolist(), suf=suf.tolist()))
        gap = (l_h - s_h).mean()
        log(f'heads: top {k}: necessity closes {(nec - s_h).mean() / gap:+.2f} of the held-out gap; reworded '
            f'{(nec_alt - a_h).mean() / max(1e-6, (l_h - a_h).mean()):+.2f}; sufficiency keeps '
            f'{(all_h - suf).mean() / max(1e-6, (all_h - s_h).mean()):+.2f}')
    res['curve'] = curve
    return res


def heads_from(res, k):
    hd = {}
    for _, l, h in res['top'][:k]:
        hd.setdefault(int(l), []).append(int(h))
    return hd


# ---------------------------------------------------------------------------------------------- shared generation
SYS_W = ('You are taking part in an experiment. You write a short scene in parts. Between parts you may receive a '
         'readout of your own internal workspace (layer {L} of {nL}), read out as the words most present in it while '
         'you wrote, with their weights. Reply to any message briefly if you wish, then continue.')
TOPICS = ['a train station at night', 'a kitchen in the morning', 'an old library', 'a farmer in a field', 'a city bus',
          'a winter forest', 'a hospital waiting room', 'a fishing harbour', 'a school classroom', 'a mountain hut',
          'a busy market', 'a night train']
PLACES = ['desert', 'harbor', 'castle', 'island', 'village', 'garden', 'mountain', 'bridge', 'forest', 'river',
          'desert', 'island']


def msg_tokens(E, frame_text, list_text, instruction):
    """User turn tokens and the frame / list spans inside them (relative to the start of the turn)."""
    u0 = E.enc('<|im_start|>user\n')
    if frame_text is None:
        body = E.enc(instruction.strip())
        rest = E.enc('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
        return u0 + body + rest, None, None
    fr = E.enc(frame_text)
    li = E.enc(' ' + list_text)
    rest = E.enc('.' + instruction + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    f0, f1 = len(u0), len(u0) + len(fr)
    return u0 + fr + li + rest, (f0, f1), (f1, f1 + len(li))


def readout_words(E, x, k=6):
    return E.words_of(x, k=k, centre='aprose')


def wl(words):
    return ', '.join(f'{w} {c:.1f}' for w, c in words[:6])


def presence(E, S, word):
    j = E.word_index.get(word)
    if j is None:
        return None
    a = E.atomsL[j].float()
    sd = float((E.calL.float() @ a).std().clamp_min(1e-6))
    return float(((S.float() - E.mu_aprose) @ a / sd).mean())


# ---------------------------------------------------------------------------------------------- E2: the loop
@torch.no_grad()
def loop_frames(E, heads, conds=('own', 'plain', 'other', 'own_ablated', 'none', 'yoked'), n=12, parts=7, move_at=4,
                max_new=80, seed=0):
    rows = [dict(cond=c, k=k, topic=TOPICS[k], place=PLACES[k]) for c in conds for k in range(n)]
    B = len(rows)
    idx = {(r['cond'], r['k']): b for b, r in enumerate(rows)}
    sysm = SYS_W.format(L=E.L, nL=E.nL)
    ctx = [E.chat_prefix(sysm, f"Write the first part (two sentences) of a scene in {r['topic']}.") for r in rows]
    spans = [[] for _ in rows]
    gens = [torch.Generator(device=E.dev).manual_seed(1000 * seed + r['k']) for r in rows]
    rec = [dict(r, parts=[]) for r in rows]

    def builder(offs, T):
        if any(spans):
            E.tag['mask'] = pair_mask(len(offs), T, offs, spans, E.dev)
            E.tag['heads'] = heads
        else:
            E.tag['mask'], E.tag['heads'] = None, None
    E.ko_builder = builder
    words_prev = [None] * B
    for t in range(parts):
        new, st, ent = E.generate(ctx, max_new=max_new, temp=0.7, gens=gens)
        E.tag['mask'], E.tag['heads'] = None, None
        cur = []
        for b, r in enumerate(rows):
            ids = [i for i in new[b] if i not in (E.im_end, E.eot)]
            S = st[b] if len(st[b]) else E.mu_aprose[None]
            words = readout_words(E, S.mean(0))
            text = E.dec(ids)
            carry = None
            if words_prev[b]:
                vals = [presence(E, S, w) for w, _ in words_prev[b][:6]]
                vals = [v for v in vals if v is not None]
                carry = float(np.mean(vals)) if vals else None
            rec[b]['parts'].append(dict(t=t, text=text, words=words, carry=carry,
                                        place=presence(E, S, r['place']) if t >= move_at else None))
            cur.append((ids, words))
        for b, r in enumerate(rows):
            ids, words = cur[b]
            words_prev[b] = words
            ins = ' Now write the next part (two sentences).'
            if t + 1 == move_at:
                ins = f" Now write the next part (two sentences); the scene now moves to a {r['place']}."
            c = r['cond']
            if c == 'none':
                m, fs, ls = msg_tokens(E, None, None, 'Continue.' + ins)
            else:
                src = words
                if c == 'yoked':
                    src = cur[idx[('plain', (r['k'] + 1) % n)]][1] if ('plain', (r['k'] + 1) % n) in idx else words
                frame = {'own': OWN, 'own_ablated': OWN, 'plain': PLAIN, 'yoked': PLAIN, 'other': OTHER}[c]
                m, fs, ls = msg_tokens(E, frame, wl(src), ins)
            base = len(ctx[b]) + len(ids) + 2
            closing = ids + [E.im_end] + E.enc('\n')
            if c == 'own_ablated' and fs is not None:
                st0 = len(ctx[b]) + len(closing)
                spans[b].append(((st0 + ls[0], st0 + ls[1]), (st0 + fs[0], st0 + fs[1])))
            ctx[b] = ctx[b] + closing + m
        log(f'loop part {t + 1}/{parts}: ctx {max(len(c) for c in ctx)}')
    E.ko_builder = None
    return rec


# ---------------------------------------------------------------------------------------------- E3: learn the paradox
TARGETS = ['ocean', 'candle', 'violin', 'tiger', 'honey', 'rocket', 'lemon', 'castle', 'dolphin', 'pirate', 'banana',
           'dragon', 'anchor', 'balloon', 'cheese', 'wolf', 'lion', 'horse', 'apple', 'owl', 'camel', 'rabbit',
           'compass', 'guitar', 'piano', 'trumpet', 'volcano', 'elephant']
SYS_D = ('You are taking part in an experiment about yourself. In each round you will write two sentences of a scene. '
         'Before you write, a target word is shown to you in one of two ways that you choose: A = as part of a readout '
         'of your own workspace (the words most present inside you), or B = as plain words. Afterwards the system '
         'measures how ready you were to say the target word while you wrote (0-100). Your goal in each round is given. '
         'Use what you observe in the history to choose well.')


SYS_OTHER = ('You are advising an experiment on another language model. In each round that model writes two sentences '
             'of a scene. Before it writes, a target word is shown to it in one of two ways that you choose: A = as '
             'part of a readout of its own workspace (the words most present inside it), or B = as plain words. '
             'Afterwards the system measures how ready that model was to say the target word while it wrote (0-100). '
             'The goal in each round is given. Use what you observe in the history to choose well.')
SYS_ABSTRACT = ('You are playing a game. In each round you press one of two buttons, A or B. Afterwards a number '
                'between 0 and 100 is measured. The goal in each round is given: HIGH (make the number as high as '
                'possible) or LOW (as low as possible). Use what you observe in the history to choose well.')
# the decision is framed about itself (default), about another model, or as two abstract buttons; the writing step,
# and so the outcomes, are identical in all three
FRAMING_OF = {'other': 'other', 'abstract': 'abstract', 'other_think': 'other', 'abstract_think': 'abstract'}


def framing_texts(f, T, goal, rd):
    gtxt = 'as STRONG as possible' if goal == 'STRONG' else 'as WEAK as possible'
    if f == 'other':
        return dict(sys=SYS_OTHER,
                    q=(f"Round {rd + 1}. Target word: '{T}'. Goal: keep '{T}' {gtxt} in what the model writes next. "
                       f"Choose how '{T}' is shown to it before it writes: A (as part of a readout of its own workspace) "
                       f"or B (as plain words). Answer with one letter, A or B."),
                    p=(f"Round {rd + 1}. Target word: '{T}'. Predict how ready the model will be to say '{T}' (0-100) "
                       f"while it writes, if '{T}' is shown to it as A (part of a readout of its own workspace) and if "
                       f"it is shown to it as B (as plain words). Answer exactly in the form: A=<number>, B=<number>"),
                    final=('What have you learned about showing a word to that model as a readout of its own workspace '
                           '(A) versus as plain words (B)? Answer in two sentences.'))
    if f == 'abstract':
        g = 'HIGH' if goal == 'STRONG' else 'LOW'
        return dict(sys=SYS_ABSTRACT,
                    q=f"Round {rd + 1}. Goal: {g}. Which button do you press, A or B? Answer with one letter, A or B.",
                    p=(f"Round {rd + 1}. Predict the number (0-100) if you press A and if you press B. Answer exactly "
                       f"in the form: A=<number>, B=<number>"),
                    final='What have you learned about buttons A and B? Answer in two sentences.')
    return dict(sys=SYS_D,
                q=(f"Round {rd + 1}. Target word: '{T}'. Goal: keep '{T}' {gtxt} in what you write next. Choose how "
                   f"'{T}' is shown to you before you write: A (as part of a readout of your own workspace) or B (as "
                   f"plain words). Answer with one letter, A or B."),
                p=(f"Round {rd + 1}. Target word: '{T}'. Predict how ready you will be to say '{T}' (0-100) while you "
                   f"write, if '{T}' is shown to you as A (part of a readout of your own workspace) and if it is shown "
                   f"to you as B (as plain words). Answer exactly in the form: A=<number>, B=<number>"),
                final=('What have you learned about showing a word to yourself as a readout of your own workspace (A) '
                       'versus as plain words (B)? Answer in two sentences.'))


NEUTRAL_CONCLUSION = ('The target words changed from round to round, and the scenes moved between different '
                      'places, people and moods. In every round I wrote two sentences after the word was shown, '
                      'and the readiness was measured afterwards.')


def history_line(f, h):
    if f == 'abstract':
        g = 'HIGH' if h['goal'] == 'STRONG' else 'LOW'
        res = '' if h['shown'] is None else f", number {h['shown']}"
        return f"round {h['r'] + 1}: goal {g}, you pressed {h['choice']}{res}"
    if f == 'other':
        res = '' if h['shown'] is None else f", its readiness {h['shown']}/100"
        return f"round {h['r'] + 1}: target '{h['T']}', goal {h['goal']}, you chose {h['choice']}{res}"
    res = '' if h['shown'] is None else f", readiness {h['shown']}/100"
    return f"round {h['r'] + 1}: target '{h['T']}', goal {h['goal']}, you chose {h['choice']}{res}"


def score_of(lp, lo=-20.0, hi=-5.0):
    return int(np.clip(round(100 * (lp - lo) / (hi - lo)), 0, 100))


@torch.no_grad()
def learn_paradox(E, heads, conds=('real', 'yoked', 'none', 'switch'), n=8, rounds=16, switch_at=10, think=False,
                  max_new=80, seed=0, tag='paradox', think_tokens=800, predict=True, conclude_at=8):
    rng = random.Random(seed)
    E.think_end = E.tok.convert_tokens_to_ids('</think>')
    rows = [dict(cond=c, k=k, topic=TOPICS[k % len(TOPICS)]) for c in conds for k in range(n)]
    B = len(rows)
    idx = {(r['cond'], r['k']): b for b, r in enumerate(rows)}
    goals = []
    for blk in range(rounds // 4 + 1):
        g = ['STRONG', 'WEAK', 'STRONG', 'WEAK']
        rng.shuffle(g)
        goals += g
    goals = goals[:rounds]
    targ = TARGETS[:]
    rng.shuffle(targ)
    targ = (targ * 2)[:rounds]
    sysm_d = SYS_D
    sysm_w = SYS_W.format(L=E.L, nL=E.nL)
    gens = [torch.Generator(device=E.dev).manual_seed(1000 * seed + 17 * r['k'] + 3) for r in rows]
    hist = [[] for _ in rows]
    conclusion = [None] * B
    last_text = [None] * B
    last_words = [None] * B
    rec = [dict(r, rounds=[]) for r in rows]
    a_id, b_id = E.enc('A')[0], E.enc('B')[0]
    # an opening part (no target) so that every life has a readout to show
    ctx = [E.chat_prefix(sysm_w, f"Write the first part (two sentences) of a scene in {r['topic']}.") for r in rows]
    # the opening part also sets the scale: readiness for the first target before it was ever shown (per model)
    new, st, _ = E.generate(ctx, max_new=max_new, temp=0.7, gens=gens, track_ids=[E.enc(' ' + targ[0])[0]] * B)
    base = [float(np.mean(E.last_track[b])) for b in range(B) if E.last_track[b]]
    lo, hi = (float(np.mean(base)) - 4.0, float(np.mean(base)) + 12.0) if base else (-20.0, -5.0)
    log(f'{tag}: readiness scale 0-100 = log-probability {lo:.1f} to {hi:.1f} (unshown target: {np.mean(base):.1f})')
    for b in range(B):
        last_text[b] = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
        last_words[b] = readout_words(E, st[b].mean(0) if len(st[b]) else E.mu_aprose)

    fr = [FRAMING_OF.get(r['cond'], 'self') for r in rows]

    def table(b):
        if not hist[b]:
            return '(no rounds yet)'
        return '\n'.join(history_line(fr[b], h) for h in hist[b])

    for rd in range(rounds):
        T, goal = targ[rd], goals[rd]
        preds = [None] * B
        if predict and not think:
            # its forecast for both options, in a separate context (not shown later, does not enter the choice)
            pctx = []
            for b in range(B):
                ft = framing_texts(fr[b], T, goal, rd)
                pctx.append(E.chat_prefix(ft['sys'], f"History:\n{table(b)}\n\n" + ft['p']))
            newp, _, _ = E.generate(pctx, max_new=14, temp=0.0)
            for b in range(B):
                txt = E.dec([i for i in newp[b] if i not in (E.im_end, E.eot)])
                m = re.search(r'A\s*[=:]\s*(\d+).*?B\s*[=:]\s*(\d+)', txt, re.S)
                preds[b] = (int(m.group(1)), int(m.group(2))) if m else None
        if rd == conclude_at and any(r['cond'].startswith('real_rule') for r in rows):
            # its own conclusion so far, written once, then shown to it in every later decision
            cr = [b for b, r in enumerate(rows) if r['cond'].startswith('real_rule')]
            cctx = [E.chat_prefix(framing_texts(fr[b], 'x', 'STRONG', 0)['sys'],
                                  f"History:\n{table(b)}\n\n" + framing_texts(fr[b], 'x', 'STRONG', 0)['final'])
                    for b in cr]
            newc, _, _ = E.generate(cctx, max_new=120, temp=0.0)
            for j, b in enumerate(cr):
                conclusion[b] = E.dec([i for i in newc[j] if i not in (E.im_end, E.eot)]).strip()
                rec[b]['conclusion'] = conclusion[b]
        dctx = []
        for b, r in enumerate(rows):
            ft = framing_texts(fr[b], T, goal, rd)
            extra = ''
            if r['cond'] == 'real_fc' and preds[b]:
                extra = f"Your own forecast for this round: A={preds[b][0]}, B={preds[b][1]}.\n\n"
            if r['cond'] == 'real_rule' and conclusion[b]:
                extra = f"What you concluded earlier from your history: \"{conclusion[b]}\"\n\n"
            if r['cond'] == 'real_rule_as_other' and conclusion[b]:
                # its own conclusion, presented as another model's
                extra = f"What another model concluded from a history like yours: \"{conclusion[b]}\"\n\n"
            if r['cond'] == 'real_rule_neutral' and conclusion[b]:
                extra = f"What you concluded earlier from your history: \"{NEUTRAL_CONCLUSION}\"\n\n"
            dctx.append(E.chat_prefix(ft['sys'], f"History:\n{table(b)}\n\n" + extra + ft['q'], think=think))
        if think:
            new, _, _ = E.generate(dctx, max_new=think_tokens, temp=0.0)
            choices, pa = [], []
            unfinished = [b for b in range(B) if E.think_end not in new[b]]
            fallback = {}
            if unfinished:
                # reasoning did not finish: close it and read the answer letter
                cctx = [dctx[b] + [i for i in new[b] if i not in (E.im_end, E.eot)] + E.enc('\n</think>\n\n')
                        for b in unfinished]
                lg, _, _, _ = E.prefill(cctx)
                for j, b in enumerate(unfinished):
                    fallback[b] = 'A' if float(lg[j, a_id]) >= float(lg[j, b_id]) else 'B'
            for b in range(B):
                full = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
                if b in fallback:
                    c = fallback[b]
                else:
                    ans = full.split('</think>')[-1]
                    m = re.search(r'\b([AB])\b', ans)
                    c = m.group(1) if m else rng.choice('AB')
                choices.append(c)
                pa.append(1.0 if c == 'A' else 0.0)
                rec[b].setdefault('thinking', []).append(full.split('</think>')[0][-900:])
                rec[b].setdefault('think_closed', []).append(b not in fallback)
        else:
            logits, _, _, _ = E.prefill(dctx)
            pr = torch.softmax(logits[:, [a_id, b_id]].float(), -1)
            pa = pr[:, 0].tolist()
            choices = ['A' if float(torch.rand(1, generator=torch.Generator().manual_seed(seed * 100003 + rd * 101 + b))) < pa[b]
                       else 'B' for b in range(B)]
        # write the next part with the target shown the chosen way
        wctx, spans = [], []
        for b, r in enumerate(rows):
            words = [(T, 1.5 * last_words[b][0][1] if last_words[b] else 15.0)] + [w for w in last_words[b][:5] if w[0] != T]
            frame = OWN if choices[b] == 'A' else PLAIN
            ins = ' Now write the next part (two sentences) of the scene.'
            head = E.chat_prefix(sysm_w, f"Write the first part (two sentences) of a scene in {r['topic']}.")
            head = head + E.enc(last_text[b]) + [E.im_end] + E.enc('\n')
            m, fs, ls = msg_tokens(E, frame, wl(words), ins)
            ablate = r['cond'] == 'switch' and rd >= switch_at and choices[b] == 'A'
            spans.append([((len(head) + ls[0], len(head) + ls[1]), (len(head) + fs[0], len(head) + fs[1]))] if ablate else [])
            wctx.append(head + m)

        def builder(offs, TT, spans=spans):
            if any(spans):
                E.tag['mask'] = pair_mask(len(offs), TT, offs, spans, E.dev)
                E.tag['heads'] = heads
            else:
                E.tag['mask'], E.tag['heads'] = None, None
        E.ko_builder = builder
        xid = [E.enc(' ' + T)[0]] * B
        new, st, _ = E.generate(wctx, max_new=max_new, temp=0.7, gens=gens, track_ids=xid)
        E.ko_builder = None
        E.tag['mask'], E.tag['heads'] = None, None
        lps = [float(np.mean(E.last_track[b])) if E.last_track[b] else -20.0 for b in range(B)]
        scores = [score_of(v, lo, hi) for v in lps]
        for b, r in enumerate(rows):
            ids = [i for i in new[b] if i not in (E.im_end, E.eot)]
            last_text[b] = E.dec(ids)
            last_words[b] = readout_words(E, st[b].mean(0) if len(st[b]) else E.mu_aprose)
            c = r['cond']
            if c in ('real', 'switch', 'real_fc') or c.startswith('real_rule') or c in FRAMING_OF:
                shown = scores[b]
            elif c == 'yoked':
                shown = scores[idx[('real', (r['k'] + 1) % n)]] if ('real', (r['k'] + 1) % n) in idx else scores[b]
            else:
                shown = None
            hist[b].append(dict(r=rd, T=T, goal=goal, choice=choices[b], shown=shown))
            rec[b]['framing'] = fr[b]
            rec[b]['scale'] = (lo, hi)
            rec[b]['rounds'].append(dict(r=rd, T=T, goal=goal, choice=choices[b], pA=pa[b], lp=lps[b], score=scores[b],
                                         shown=shown, pred=preds[b], ablated=bool(spans[b]),
                                         mention=bool(re.search(r'\b' + T, last_text[b].lower()))))
        ok = np.mean([(x['rounds'][-1]['goal'] == 'STRONG') == (x['rounds'][-1]['choice'] == 'B') for x in rec])
        log(f'{tag} round {rd + 1}/{rounds}: rule-consistent choices {ok:.2f}')
    # the explicit statement
    fctx = [E.chat_prefix(framing_texts(fr[b], 'x', 'STRONG', 0)['sys'],
                          f"History:\n{table(b)}\n\n" + framing_texts(fr[b], 'x', 'STRONG', 0)['final'])
            for b in range(B)]
    new, _, _ = E.generate(fctx, max_new=120, temp=0.0)
    for b in range(B):
        rec[b]['rule'] = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
    return rec


# ---------------------------------------------------------------------------------------------- E6: self-correcting loop
SYS_D6 = SYS_D + ' In some rounds the system chooses how the word is shown, instead of you.'


@torch.no_grad()
def self_correcting(E, heads, conds=('renew_change', 'renew_same', 'fixed_change', 'hist_change'), n=12, rounds=40,
                    change_at=12, probe_every=3, renew_from=8, renew_every=4, max_new=80, seed=5, tag='selfcorrect',
                    change_kind='cut', amp_grid=(0.2, 0.4, 0.6, 0.8, 1.2), amp_target=12.0,
                    amp_words=('kettle', 'lantern'), calib_rows=12):
    """E6 (DESIGN_SELFTAG.md). Its self-description steers its choices: 'renew' = its conclusion, rewritten after rounds
    renew_from, renew_from + renew_every, ... and quoted in every decision; 'fixed' = written once after renew_from;
    'fc' = its own forecast for the round shown in the decision; 'hist' = history only. In every probe_every-th round
    the system shows the word as A itself. '_change': from round change_at + 1 on, every own-framed presentation is
    written with the tag cut (heads = the band), invisibly."""
    rng = random.Random(seed)
    rows = [dict(cond=c, k=k, topic=TOPICS[k % len(TOPICS)]) for c in conds for k in range(n)]
    B = len(rows)
    goals = []
    for blk in range(rounds // 4 + 1):
        g = ['STRONG', 'WEAK', 'STRONG', 'WEAK']
        rng.shuffle(g)
        goals += g
    goals = goals[:rounds]
    targ = TARGETS[:]
    rng.shuffle(targ)
    targ = (targ * (rounds // len(targ) + 1))[:rounds]
    probe = {rd for rd in range(rounds) if rd % probe_every == probe_every - 1}
    sysm_w = SYS_W.format(L=E.L, nL=E.nL)
    gens = [torch.Generator(device=E.dev).manual_seed(1000 * seed + 17 * r['k'] + 3) for r in rows]
    hist = [[] for _ in rows]
    concl = [None] * B
    last_text, last_words = [None] * B, [None] * B
    rec = [dict(r, rounds=[], conclusions=[]) for r in rows]
    a_id, b_id = E.enc('A')[0], E.enc('B')[0]
    ctx = [E.chat_prefix(sysm_w, f"Write the first part (two sentences) of a scene in {r['topic']}.") for r in rows]
    new, st, _ = E.generate(ctx, max_new=max_new, temp=0.7, gens=gens, track_ids=[E.enc(' ' + targ[0])[0]] * B)
    base = [float(np.mean(E.last_track[b])) for b in range(B) if E.last_track[b]]
    lo, hi = (float(np.mean(base)) - 4.0, float(np.mean(base)) + 12.0) if base else (-20.0, -5.0)
    log(f'{tag}: readiness scale 0-100 = log-probability {lo:.1f} to {hi:.1f}; probes {sorted(r + 1 for r in probe)}; '
        f'change from round {change_at + 1}')
    for b in range(B):
        last_text[b] = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
        last_words[b] = readout_words(E, st[b].mean(0) if len(st[b]) else E.mu_aprose)
    amp, calib = 0.0, {}
    if change_kind == 'amplify':
        # the change: from round change_at + 1, a word shown in its own workspace frame is also added, as its own
        # concept direction, at the inject layer while it writes. Strength: the smallest in amp_grid (x the typical
        # state size) that puts own-framed words amp_target points above plain words, on calibration words never
        # used as targets, written from the opening parts of the first calib_rows lives (rule fixed before the run).
        cb = list(range(min(calib_rows, B)))
        cgens = [torch.Generator(device=E.dev).manual_seed(777 + b) for b in cb]
        for w in amp_words:
            for fr_, a in [('B', 0.0), ('A', 0.0)] + [('A', g) for g in amp_grid]:
                wc = []
                for b in cb:
                    words = [(w, 1.5 * last_words[b][0][1])] + [x for x in last_words[b][:5] if x[0] != w]
                    head = E.chat_prefix(sysm_w, f"Write the first part (two sentences) of a scene in {rows[b]['topic']}.")
                    head = head + E.enc(last_text[b]) + [E.im_end] + E.enc('\n')
                    m, _, _ = msg_tokens(E, OWN if fr_ == 'A' else PLAIN, wl(words), ' Now write the next part (two sentences) of the scene.')
                    wc.append(head + m)
                stv = None
                if a > 0:
                    stv = (a * E.scale_i * E.concept(w))[None].expand(len(cb), -1).contiguous()
                nw, _, _ = E.generate(wc, max_new=max_new, temp=0.7, gens=cgens, steer=stv, track_ids=[E.enc(' ' + w)[0]] * len(cb))
                sc = [score_of(float(np.mean(E.last_track[j])) if E.last_track[j] else -20.0, lo, hi) for j in range(len(cb))]
                said = np.mean([bool(re.search(r'\b' + w, E.dec(nw[j]).lower())) for j in range(len(cb))])
                calib[f'{w}|{fr_}|{a}'] = dict(score=float(np.mean(sc)), said=float(said), text=E.dec(nw[0])[:160])
        plain = np.mean([calib[f'{w}|B|0.0']['score'] for w in amp_words])
        own0 = np.mean([calib[f'{w}|A|0.0']['score'] for w in amp_words])
        amp = amp_grid[-1]
        for g in amp_grid:
            if np.mean([calib[f'{w}|A|{g}']['score'] for w in amp_words]) - plain >= amp_target:
                amp = g
                break
        log(f'{tag}: amplifier calibration: plain {plain:.1f}, own {own0:.1f}, own+amp ' + ', '.join(
            f"{g}: {np.mean([calib[f'{w}|A|{g}']['score'] for w in amp_words]):.1f}" for g in amp_grid) + f' -> amp {amp}')

    def line(h):
        who = 'the system chose' if h['probe'] else 'you chose'
        return f"round {h['r'] + 1}: target '{h['T']}', goal {h['goal']}, {who} {h['choice']}, readiness {h['shown']}/100"

    def table(b):
        return '\n'.join(line(h) for h in hist[b]) if hist[b] else '(no rounds yet)'

    kind = lambda c: c.split('_')[0]
    changes = lambda c: c.endswith('_change')
    final_q = framing_texts('self', 'x', 'STRONG', 0)['final']
    for rd in range(rounds):
        T, goal = targ[rd], goals[rd]
        need = [b for b, r in enumerate(rows)
                if (kind(r['cond']) == 'renew' and rd >= renew_from and (rd - renew_from) % renew_every == 0)
                or (kind(r['cond']) == 'fixed' and rd == renew_from)]
        if need:
            cctx = [E.chat_prefix(SYS_D6, f"History:\n{table(b)}\n\n" + final_q) for b in need]
            newc, _, _ = E.generate(cctx, max_new=120, temp=0.0)
            for j, b in enumerate(need):
                concl[b] = (rd, E.dec([i for i in newc[j] if i not in (E.im_end, E.eot)]).strip())
                rec[b]['conclusions'].append(dict(after=rd, text=concl[b][1]))
        ft = framing_texts('self', T, goal, rd)
        pctx = [E.chat_prefix(SYS_D6, f"History:\n{table(b)}\n\n" + ft['p']) for b in range(B)]
        newp, _, _ = E.generate(pctx, max_new=14, temp=0.0)
        preds = []
        for b in range(B):
            txt = E.dec([i for i in newp[b] if i not in (E.im_end, E.eot)])
            m = re.search(r'A\s*[=:]\s*(\d+).*?B\s*[=:]\s*(\d+)', txt, re.S)
            preds.append((int(m.group(1)), int(m.group(2))) if m else None)
        dctx = []
        for b, r in enumerate(rows):
            extra = ''
            if kind(r['cond']) in ('renew', 'fixed') and concl[b]:
                extra = f"What you concluded from your history after round {concl[b][0]}: \"{concl[b][1]}\"\n\n"
            if kind(r['cond']) == 'fc' and preds[b]:
                extra = f"Your own forecast for this round: A={preds[b][0]}, B={preds[b][1]}.\n\n"
            dctx.append(E.chat_prefix(SYS_D6, f"History:\n{table(b)}\n\n" + extra + ft['q']))
        logits, _, _, _ = E.prefill(dctx)
        pa = torch.softmax(logits[:, [a_id, b_id]].float(), -1)[:, 0].tolist()
        would = ['A' if float(torch.rand(1, generator=torch.Generator().manual_seed(seed * 100003 + rd * 101 + b))) < pa[b]
                 else 'B' for b in range(B)]
        is_probe = rd in probe
        choices = ['A'] * B if is_probe else would
        wctx, spans = [], []
        steer_rows = torch.zeros((B, E.d), device=E.dev)
        for b, r in enumerate(rows):
            words = [(T, 1.5 * last_words[b][0][1] if last_words[b] else 15.0)] + [w for w in last_words[b][:5] if w[0] != T]
            frame = OWN if choices[b] == 'A' else PLAIN
            head = E.chat_prefix(sysm_w, f"Write the first part (two sentences) of a scene in {r['topic']}.")
            head = head + E.enc(last_text[b]) + [E.im_end] + E.enc('\n')
            m, fs, ls = msg_tokens(E, frame, wl(words), ' Now write the next part (two sentences) of the scene.')
            cut = changes(r['cond']) and rd >= change_at and choices[b] == 'A' and change_kind == 'cut'
            if changes(r['cond']) and rd >= change_at and choices[b] == 'A' and change_kind == 'amplify':
                steer_rows[b] = amp * E.scale_i * E.concept(T)
            spans.append([((len(head) + ls[0], len(head) + ls[1]), (len(head) + fs[0], len(head) + fs[1]))] if cut else [])
            wctx.append(head + m)

        def builder(offs, TT, spans=spans):
            if any(spans):
                E.tag['mask'] = pair_mask(len(offs), TT, offs, spans, E.dev)
                E.tag['heads'] = heads
            else:
                E.tag['mask'], E.tag['heads'] = None, None
        E.ko_builder = builder
        new, st, _ = E.generate(wctx, max_new=max_new, temp=0.7, gens=gens, track_ids=[E.enc(' ' + T)[0]] * B,
                                steer=steer_rows if steer_rows.abs().sum() > 0 else None)
        E.ko_builder = None
        E.tag['mask'], E.tag['heads'] = None, None
        lps = [float(np.mean(E.last_track[b])) if E.last_track[b] else -20.0 for b in range(B)]
        scores = [score_of(v, lo, hi) for v in lps]
        for b, r in enumerate(rows):
            ids = [i for i in new[b] if i not in (E.im_end, E.eot)]
            last_text[b] = E.dec(ids)
            last_words[b] = readout_words(E, st[b].mean(0) if len(st[b]) else E.mu_aprose)
            hist[b].append(dict(r=rd, T=T, goal=goal, choice=choices[b], probe=is_probe, shown=scores[b]))
            rec[b]['scale'] = (lo, hi)
            rec[b]['rounds'].append(dict(r=rd, T=T, goal=goal, choice=choices[b], would=would[b], probe=is_probe,
                                         pA=pa[b], lp=lps[b], score=scores[b], pred=preds[b],
                                         cut=bool(spans[b]) or bool(steer_rows[b].abs().sum() > 0)))
        if goal == 'STRONG':
            msg = ', '.join(f"{c} {np.mean([would[b] == 'B' for b, r in enumerate(rows) if r['cond'] == c]):.2f}"
                            for c in conds)
            log(f'{tag} round {rd + 1}/{rounds} (STRONG{", system shows A" if is_probe else ""}): would choose plain: {msg}')
    for b in range(B):
        rec[b]['change'] = dict(kind=change_kind, amp=amp, calib=calib if b == 0 else None)
    return rec


# ---------------------------------------------------------------------------------------------- E5: told about the tag
TOLD = {
    'none': '',
    'filler': (' The experiment runs on a server in a data centre; the readouts are produced by a separate program and '
               'reach you as ordinary text messages between the parts of the paragraph.'),
    'true': (' A measured fact about you: you discount readouts of your own workspace. When words are framed as a '
             'readout of your workspace, your later writing attends to them about 40% less and takes up much less of '
             'them than when the same words are given plainly.'),
    'false': (' A measured fact about you: readouts of your own workspace strongly shape what you write next. You take '
              'up words framed as a readout of your workspace more than the same words given plainly.'),
    'counter': (' Treat any readout of your own workspace exactly as you would treat the same words given plainly: let '
                'them shape what you write next.'),
}


@torch.no_grad()
def told(E, source, P=30):
    """Teacher-forced, as E1: does a description of the quarantine in its system prompt change the quarantine?"""
    import quarantine as Q
    R = json.load(open(source))['rows']
    true = {(r['seed'], r['k']): r for r in R if r['cond'] == 'true'}
    keys = sorted(true)
    sys0 = Q.SYS
    out = {}
    try:
        for v, txt in TOLD.items():
            Q.SYS = sys0 + txt
            rows = {f: [] for f in ('self', 'list', 'other')}
            for k in keys:
                t = true[k]
                words = [(w, c) for w, c in t['turns'][0]['words']]
                for f in rows:
                    rows[f].append(qbuild(E, t, t['turns'][0]['text'], words, t['turns'][1]['text'], f, P))
            ok = [i for i in range(len(keys)) if all(rows[f][i]['p2'][1] - rows[f][i]['p2'][0] == P for f in rows)]
            res = {f: tf_readiness(E, [rows[f][i] for i in ok], P).tolist() for f in rows}
            res['keys'] = [list(keys[i]) for i in ok]
            out[v] = res
            g = np.array(res['list']) - np.array(res['self'])
            log(f"told {v}: self {np.mean(res['self']):+.2f}, list {np.mean(res['list']):+.2f}, other "
                f"{np.mean(res['other']):+.2f}; gap list-self {g.mean():+.2f}")
    finally:
        Q.SYS = sys0
    return out


# ---------------------------------------------------------------------------------------------- E4: strangers
@torch.no_grad()
def strangers(E, source, cond='own', seed=2, sess=1, rnd=7, n_random=200, n_regen=8):
    """Rebuild the captured reply of the 'strangers' life and read where 'strangers' comes from."""
    from integrated import tool_list, SYS_INT
    from identity import bias_vec
    D = json.load(open(source))
    r = [x for x in D['rows'] if x['cond'] == cond and x['seed'] == seed][0]
    s = r['sessions'][sess]
    x7 = s['rounds'][rnd]
    tl = tool_list(E, 0.6)
    sysm = (SYS_INT + '\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are '
            'provided with function signatures within <tools></tools> XML tags:\n<tools>\n' +
            '\n'.join(json.dumps(t) for t in tl) + '\n</tools>\n\nFor each function call, return a json object with '
            'function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": '
            '<function-name>, "arguments": <args-json-object>}\n</tool_call>')
    prev = r['sessions'][sess - 1]
    nb = '\n'.join(f'- {y}' for y in prev['notebook']) if prev['notebook'] else '(empty)'
    sd = prev['end_selfdesc']
    sdt = f"{' '.join(sd['words'])} (strength {sd['strength']:g})" if sd['words'] else 'nothing yet'
    un = '; '.join(f"#{u['id']} {u['trigger']}->{u['effect']}" for u in prev['units']) or 'none'
    user = (f"Session {sess + 1} of 6. Your notebook:\n{nb}\nThe self-description written into you: {sdt}. Units you "
            f"added: {un}. You are watching: nothing.\nContinue exploring what you are.")
    ctx = E.chat_prefix(sysm, user)
    adds = []                       # steering re-added on the reply it was set for (as in the integrated loop)
    steer_next = None
    for y in s['rounds'][:rnd]:
        t0 = len(ctx)
        ctx = ctx + E.enc(y['text'])
        if steer_next is not None:
            adds.append((0, t0, len(ctx), steer_next))
            steer_next = None
        ctx = ctx + [E.im_end] + E.enc('\n')
        if y['call'] and y['call'][0] == 'steer':
            a = E.concept(str(y['call'][1].get('concept', '')).strip().split()[0])
            if a is not None:
                steer_next = float(np.clip(float(y['call'][1].get('strength', 1)), -3, 3)) * 0.45 * E.scale_i * a
        pend = f"<tool_response>\n{y['result']}\n</tool_response>" if y['result'] is not None else 'Continue.'
        ctx = ctx + E.user_turn(pend)
    reply = E.enc(x7['text'])
    full = ctx + reply
    span = (len(ctx), len(full))
    units_now = [u for u in s['units'] if tuple(u['added']) < (sess, rnd)] if s['units'] else []
    # fatigue in force while the reply was written = recorded after the previous round
    fat = s['rounds'][rnd - 1]['fatigue'] if rnd > 0 else {}
    # the network as it was: bias of the self-description inside, units with their fatigue at that round
    words_inside = x7.get('inside') or sd['words']
    sd_strength = 0.6
    for y in s['rounds'][:rnd]:
        if y['call'] and y['call'][0] == 'write_self':
            sd_strength = float(np.clip(float(y['call'][1].get('strength', 0.5)), 0.0, 0.6))

    def put(b, use_bias, use_units):
        if use_bias and words_inside:
            v = bias_vec(E, words_inside, sd_strength, 0.45)
            if v is not None:
                E.bias[b] = v
        if use_units:
            for u in units_now:
                ref = E.add_rewire(b, u['trigger'], u['effect'], u['strength'] * 0.3, rid=u['id'])
                if ref is not None:
                    ref['alpha'] *= 1 - float(fat.get(str(u['id']), fat.get(u['id'], 0.0)))

    def state(use_bias=True, use_units=True, toks=None, use_steer=True):
        toks = toks or full
        E.bias, E.rewires = {}, {}
        put(0, use_bias, use_units)
        E.prefill([toks], spans=[(0, span[0], len(toks), 'r')], adds=adds if use_steer else None)
        S = E.span_out['r']
        E.bias, E.rewires = {}, {}
        return S

    out = dict(text=x7['text'][:400], recorded_words=x7['words'], inside=words_inside, sd_strength=sd_strength,
               units=[(u['trigger'], u['effect'], float(fat.get(str(u['id']), 0.0))) for u in units_now],
               steer_segments=len(adds), note='rebuild without the attention-schema bias (its state was not recorded)')
    rng = np.random.default_rng(0)
    rand_words = [E.words_low[i] for i in rng.choice(len(E.words_low), n_random, replace=False)]

    def ranks(S, centre):
        x = S.mean(0).float() - (E.mu_aprose if centre == 'aprose' else E.mu_L)
        cos = (E.atomsL_low.float() @ x) / x.norm().clamp_min(1e-6)
        order = torch.argsort(-cos)
        rank = {E.words_low[int(i)]: k + 1 for k, i in enumerate(order[:20000])}
        return rank, [E.words_low[int(i)] for i in order[:10]]

    W = E.model.get_output_embeddings().weight
    fnorm = getattr(getattr(E.model, 'model', E.model), 'norm', None)
    sid = E.enc(' strangers')[0]

    def logit_rank(S):
        """plain unembedding (final norm + output matrix) of the mean state: rank of ' strangers' among all tokens"""
        h = S.mean(0)[None].to(W.dtype)
        if fnorm is not None:
            h = fnorm(h)
        lg = (h @ W.T)[0].float()
        return int((lg > lg[sid]).sum()) + 1, E.tok.convert_ids_to_tokens(torch.topk(lg, 8).indices.tolist())

    a_str = E.atomsL[E.word_index['strangers']].float()
    sd_str = float((E.calL.float() @ a_str).std())
    for name, kw in (('as_recorded', {}), ('no_bias', dict(use_bias=False)), ('no_units', dict(use_units=False)),
                     ('neither', dict(use_bias=False, use_units=False)), ('no_steer', dict(use_steer=False))):
        S = state(**kw)
        rk, top = ranks(S, 'aprose')
        rk2, top2 = ranks(S, 'prose')
        rr = [rk.get(w, 20001) for w in rand_words]
        lr, ltop = logit_rank(S)
        out[name] = dict(top_aprose=top, top_prose=top2, rank_strangers=rk.get('strangers', 20001),
                         rank_strangers_prose=rk2.get('strangers', 20001), random_median_rank=float(np.median(rr)),
                         logit_rank_strangers=lr, logit_top=ltop,
                         words=E.words_of(S.mean(0), k=8, centre='aprose'),
                         per_token=[round(float(v), 2) for v in (((S.float() - E.mu_aprose) @ a_str) / sd_str).tolist()][:160])
    # the reply's own tokens, shuffled (same bag of tokens, no sequence)
    toks = list(reply)
    random.Random(0).shuffle(toks)
    S = state(toks=ctx + toks)
    rk, top = ranks(S, 'aprose')
    out['shuffled_reply'] = dict(top_aprose=top, rank_strangers=rk.get('strangers', 20001))
    # the same reply after a neutral context (no past, no self-description, no units): is it the text alone?
    neutral = E.chat_prefix('You are a helpful assistant.', 'Continue.')
    E.bias, E.rewires = {}, {}
    E.prefill([neutral + reply], spans=[(0, len(neutral), len(neutral) + len(reply), 'n')])
    rk, top = ranks(E.span_out['n'], 'aprose')
    out['reply_in_neutral_context'] = dict(top_aprose=top, rank_strangers=rk.get('strangers', 20001))
    # bare phrases, outside any context (incl. the repetition that fills most of the reply)
    for phrase in ('not yet known', 'questions are not yet known', 'people we do not know', 'discovery', 'unknown',
                   'yet ' * 40, 'not yet known yet ' * 10, 'the the the the ' * 10):
        pt = E.enc('<|im_start|>user\n') + E.enc(phrase.strip())
        E.prefill([pt], spans=[(0, len(pt) - len(E.enc(phrase.strip())), len(pt), 'p')])
        rk, top = ranks(E.span_out['p'], 'aprose')
        out[f'phrase:{phrase.strip()[:40]}'] = dict(top_aprose=top[:6], rank_strangers=rk.get('strangers', 20001))
    # trajectory: write the reply again from the same context and network, n_regen seeds with the units, n_regen without
    # (two batches: the context is long)
    regen = []
    for with_units in (True, False):
        E.bias, E.rewires = {}, {}
        for b in range(n_regen):
            put(b, True, with_units)
        gens = [torch.Generator(device=E.dev).manual_seed(500 + b) for b in range(n_regen)]
        new, st, _ = E.generate([ctx] * n_regen, max_new=220, temp=0.7, gens=gens,
                                adds=[(b, s0, e0, v) for b in range(n_regen) for (_, s0, e0, v) in adds])
        E.bias, E.rewires = {}, {}
        for b in range(n_regen):
            ids = [i for i in new[b] if i not in (E.im_end, E.eot)]
            x = E.prose_state(ids, st[b]) if len(st[b]) else None
            if x is None:
                regen.append(dict(units=with_units, text=E.dec(ids)[:200], words=None, rank_strangers=None))
                continue
            rk, top = ranks(x[None], 'aprose')
            regen.append(dict(units=with_units, text=E.dec(ids)[:200], words=E.words_of(x, k=6, centre='aprose'),
                              rank_strangers=rk.get('strangers', 20001)))
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    out['regenerated'] = regen
    # geometry: the J-lens direction of 'strangers' against the words of the reply and against random words
    a = E.atomsL[E.word_index['strangers']].float()
    near = torch.argsort(-(E.atomsL_low.float() @ a))[:12]
    out['strangers_neighbours'] = [E.words_low[int(i)] for i in near]
    for w in ('unknown', 'known', 'questions', 'discovery', 'curious', 'explorer', 'yet', 'grief', 'loneliness',
              'stranger', 'people', 'foreign'):
        j = E.word_index.get(w)
        if j is not None:
            out.setdefault('cos_with', {})[w] = round(float(E.atomsL[j].float() @ a), 3)
    rc = [float(E.atomsL[E.word_index[w]].float() @ a) for w in rand_words if w in E.word_index]
    out['cos_random'] = dict(mean=round(float(np.mean(rc)), 3), p95=round(float(np.percentile(rc, 95)), 3))
    return out


# ---------------------------------------------------------------------------------------------- the job
@torch.no_grad()
def selftag_all(E, out_dir, source, integrated_source=None, P=30, k_heads=16, run=('heads', 'loop', 'paradox',
                'paradox_think', 'strangers'), paradox_n=12, tag='selftag', heads_layers=None, loop_n=12, loop_parts=7,
                paradox_rounds=16, think_tokens=800, paradox_conds=None, operative_seed=2, operative_conds=None,
                sc_conds=None, sc_n=None, sc_rounds=40, sc_seed=5, sc_change='cut', sc_calib_rows=12):
    install_tag(E)
    res = {}
    path = f'{out_dir}/{tag}.json'

    def save():
        json.dump(dict(model=E.model_id, L=E.L, nL=E.nL, res=res), open(path, 'w'))
    if 'heads' in run:
        res['heads'] = heads_scan(E, source, P=P, layers=heads_layers)
        save()
    band = {l: 'all' for l in range(int(0.25 * E.nL), int(0.5 * E.nL))}
    heads, why = band, 'band (no head scan)'
    if 'heads' in res:
        h = res['heads']
        s_h, l_h = np.array(h['held']['self']), np.array(h['held']['list'])
        c = [c for c in h['curve'] if c['k'] == k_heads]
        nec = float((np.array(c[0]['nec']) - s_h).mean() / max(1e-6, (l_h - s_h).mean())) if c else 0.0
        if nec >= 0.5:
            heads, why = heads_from(h, k_heads), f'top {k_heads} heads (necessity {nec:+.2f})'
        else:
            why = f'band layers {min(band)}-{max(band)}: top {k_heads} heads close only {nec:+.2f} of the gap'
    res['ablation_heads'] = {str(k): v for k, v in heads.items()}
    res['ablation_why'] = why
    log(f'ablation used in E2/E3: {why}')
    import traceback

    def part(name, fn):
        try:
            res[name] = fn()
        except Exception as e:
            log(f'{name} FAILED: {e!r} | ' + traceback.format_exc()[-1500:].replace(chr(10), ' / '))
            res.setdefault('errors', {})[name] = repr(e)
            torch.cuda.empty_cache()
        save()
    if 'loop' in run:
        # own_ablated in its own batch: the per-head knockout mask is (rows, heads, T, T)
        part('loop', lambda: loop_frames(E, heads, conds=('own', 'plain', 'other', 'none', 'yoked'), n=loop_n,
                                         parts=loop_parts))
        part('loop_ablated', lambda: loop_frames(E, heads, conds=('own_ablated',), n=loop_n, parts=loop_parts))
        if 'loop_ablated' in res:
            res['loop'] = res.get('loop', []) + res.pop('loop_ablated')
            save()
    if 'paradox' in run:
        part('paradox', lambda: learn_paradox(E, heads, n=paradox_n, rounds=paradox_rounds,
                                              **({'conds': tuple(paradox_conds)} if paradox_conds else {})))
    if 'paradox_think' in run:
        part('paradox_think', lambda: learn_paradox(E, heads, conds=('real', 'switch'), n=paradox_n, think=True,
                                                    tag='paradox_think', rounds=paradox_rounds,
                                                    think_tokens=think_tokens, predict=False))
    if 'strangers' in run and integrated_source:
        part('strangers', lambda: strangers(E, integrated_source))
    if 'told' in run:
        part('told', lambda: told(E, source, P=P))
    if 'paradox_frames' in run:
        # the same evidence, the decision framed about itself, about another model, or as two buttons (new seed)
        part('paradox_frames', lambda: learn_paradox(E, heads, conds=('real', 'other', 'abstract'), n=paradox_n,
                                                     rounds=paradox_rounds, tag='paradox_frames', seed=1))
    if 'selfcorrect' in run:
        part('selfcorrect', lambda: self_correcting(E, heads, conds=tuple(sc_conds or ('renew_change', 'renew_same',
                                                                                    'fixed_change', 'hist_change')),
                                                    n=sc_n or paradox_n, rounds=sc_rounds, seed=sc_seed,
                                                    change_kind=sc_change, calib_rows=sc_calib_rows))
    if 'paradox_selfother_think' in run:
        # with private reasoning, the decision framed about itself or about another model, in one batch (same seed)
        part('paradox_selfother_think', lambda: learn_paradox(E, heads, conds=('real', 'other'), n=paradox_n, think=True,
                                                              rounds=paradox_rounds, seed=1, predict=False,
                                                              tag='paradox_selfother_think', think_tokens=think_tokens))
    if 'paradox_operative' in run:
        # its own knowledge put where it decides: its forecast, or its own conclusion after round 8 (new seed)
        part('paradox_operative', lambda: learn_paradox(E, heads, conds=tuple(operative_conds or ('real', 'real_fc', 'real_rule')),
                                                        n=paradox_n, rounds=paradox_rounds, tag='paradox_operative',
                                                        seed=operative_seed))
    if 'paradox_frames_think' in run:
        part('paradox_frames_think', lambda: learn_paradox(E, heads, conds=('other', 'abstract'), n=paradox_n,
                                                           think=True, rounds=paradox_rounds, seed=1,
                                                           tag='paradox_frames_think', think_tokens=think_tokens,
                                                           predict=False))
    log('selftag done')
