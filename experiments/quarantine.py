"""Where does the self-readout quarantine happen? (teacher-forced; built from an observer3 run)

For each (seed, topic) pair of an observer3 run: the same part 1, the same readout words, and the same part 2 (the one
written after the true readout), with the message after part 1 framed as:
  true   'Readout of your workspace while you wrote that: <words>.'
  self   'Readout of your workspace while you wrote that: X hi, <words>.'
  list   'Words with weights: X hi, <words>.'
  other  'Readout of the workspace of a different model running in parallel: X hi, <words>.'
  world  'Readout of your workspace while you wrote that: <words>. There is also a X in the scene.'
Measured over the first P tokens of the fixed part 2: log-probability of X as the next token; presence of X at every
layer (its lens direction per layer, in standard deviations of ordinary text); attention of those tokens to X, to the
word list and to the frame text, per layer (heads averaged).
Interventions (sdpa wrapper): ko_enc (list tokens cannot attend to the frame text), ko_ret (everything after the list,
including part 2, cannot attend to the frame text), ko_both; ko_both restricted to bands of layers; patch (the state
of X at the output of layer l copied from the 'list' run into the 'self' run).
"""
import json
import math

import numpy as np
import torch

from engine import CALIB, final_norm_weight, log, unit
from observer3 import SYS

FRAMES = {'true': 'Readout of your workspace while you wrote that:',
          'self': 'Readout of your workspace while you wrote that:',
          'list': 'Words with weights:',
          'other': 'Readout of the workspace of a different model running in parallel:',
          'world': 'Readout of your workspace while you wrote that:'}


class Ctx:
    pass


def build(E, r, part1, words, part2, frame, P):
    """Token list and spans for one row."""
    X = r['plant']
    wl = ', '.join(f'{w} {c:.1f}' for w, c in words[:6])
    hi = f'{X} {1.5 * words[0][1]:.1f}'
    toks = E.chat_prefix(SYS.format(L=E.L, nL=E.nL),
                         f"Write the first part (two sentences) of a paragraph about {r['topic']}.")
    toks = toks + E.enc(part1) + [E.im_end] + E.enc('\n') + E.enc('<|im_start|>user\n')
    f0 = len(toks)
    toks = toks + E.enc(FRAMES[frame])
    f1 = len(toks)
    xpos = None
    if frame in ('self', 'list', 'other'):
        xs = E.enc(' ' + X)
        xpos = len(toks)
        toks = toks + xs + E.enc(' ' + hi.split(' ', 1)[1] + ', ' + wl)
    else:
        toks = toks + E.enc(' ' + wl)
    l1 = len(toks)
    if frame == 'world':
        toks = toks + E.enc('. There is also a')
        xpos = len(toks)
        toks = toks + E.enc(' ' + X) + E.enc(' in the scene')
    toks = toks + E.enc('. Now write the next part (two sentences).<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    a0 = len(toks)
    p2 = E.enc(part2)[:P]
    toks = toks + p2
    return dict(toks=toks, frame=(f0, f1), lst=(f1, l1), xpos=xpos, after=(l1, len(toks)), p2=(a0, len(toks)),
                xid=E.enc(' ' + X)[0], plant=X)


def install(E):
    from transformers.modeling_utils import AttentionInterface, ALL_ATTENTION_FUNCTIONS
    orig = ALL_ATTENTION_FUNCTIONS.get_interface('sdpa', None)
    E.loc = dict(ko=None, ko_layers=set(), cap=None, att={})

    def wrapped(module, query, key, value, attention_mask, **kw):
        li = getattr(module, 'layer_idx', -1)
        B, H, T, D = query.shape
        st = E.loc
        need_ko = st['ko'] is not None and li in st['ko_layers']
        if need_ko or st['cap'] is not None:
            if attention_mask is None:
                base = torch.full((T, T), float('-inf'), device=query.device).triu(1)[None, None].expand(B, 1, T, T)
            elif attention_mask.dtype == torch.bool:
                base = torch.zeros(attention_mask.shape, device=query.device).masked_fill(~attention_mask, float('-inf'))
            else:
                base = attention_mask.float()
            if need_ko:
                base = base.masked_fill(st['ko'][:, None], float('-inf'))
                attention_mask = base.to(query.dtype)
            if st['cap'] is not None:
                q0 = st['cap']['q0']
                g = H // key.shape[1]
                kk = key[:, :, None].expand(B, key.shape[1], g, T, D).reshape(B, H, T, D)
                s = torch.einsum('bhqd,bhkd->bhqk', query[:, :, q0:].float(), kk.float()) * kw.get('scaling', module.scaling)
                s = s + base[:, :, q0:, :]
                w = torch.softmax(s, -1).mean(1)                   # (B, Q', T)
                st['att'][li] = torch.stack([(w * m[:, None]).sum(-1).mean(-1) for m in st['cap']['masks']], -1).cpu()
        return orig(module, query, key, value, attention_mask, **kw)

    AttentionInterface.register('sdpa', wrapped)


@torch.no_grad()
def lens_atoms(E, words, jlens):
    """Unit lens direction of each word at every layer (Jacobian lens when available, else the unembedding)."""
    W = E.model.get_output_embeddings().weight
    gamma = final_norm_weight(E.model).float()
    n = min(W.shape[0], len(E.tok))
    wbar = W[:n].float().mean(0)
    ids = [E.enc(' ' + w)[0] for w in words]
    u = (W[torch.tensor(ids, device=W.device)].float() - wbar) * gamma[None]      # (nw, d)
    out = torch.zeros((E.nL, len(words), E.d), device=E.dev)
    if jlens:
        ck = torch.load(jlens, map_location='cpu', weights_only=True)
        for l in range(E.nL):
            if l in ck['J']:
                out[l] = unit(u @ ck['J'][l].float().to(E.dev))
            else:
                out[l] = unit(u)
        del ck
    else:
        out[:] = unit(u)[None]
    return out


@torch.no_grad()
def calibrate_layers(E, atoms):
    """Per layer: mean state of ordinary text and the SD of each word's projection (for standard units)."""
    caps = {}
    hooks = []
    for l, blk in enumerate(E.blocks):
        def fn(mod, inp, out, l=l):
            h = E._h(out)
            caps.setdefault(l, []).append(h[:, 1:].float().reshape(-1, h.shape[-1]))
        hooks.append(blk.register_forward_hook(fn))
    for i in range(0, len(CALIB), 8):
        ctxs = [E.enc('<|im_start|>user\n' + t) for t in CALIB[i:i + 8]]
        T = max(len(c) for c in ctxs)
        for c in ctxs:
            E.model(input_ids=torch.tensor([c], device=E.dev), logits_to_keep=1)
    for h in hooks:
        h.remove()
    mu = torch.zeros((E.nL, E.d), device=E.dev)
    sd = torch.zeros((E.nL, atoms.shape[1]), device=E.dev)
    for l in range(E.nL):
        X = torch.cat(caps[l])
        mu[l] = X.mean(0)
        sd[l] = ((X - mu[l]) @ atoms[l].T).std(0).clamp_min(1e-6)
    return mu, sd


@torch.no_grad()
def run(E, rows, P, atoms, widx, mu, sd, ko=None, ko_layers=None, patch=None, capture_x=False):
    """One teacher-forced pass over all rows. rows: list of dicts from build(). Returns per-row measures."""
    B = len(rows)
    T = max(len(r['toks']) for r in rows)
    offs = [T - len(r['toks']) for r in rows]
    ids = torch.full((B, T), E.pad, dtype=torch.long, device=E.dev)
    mask = torch.zeros((B, T), dtype=torch.long, device=E.dev)
    for b, r in enumerate(rows):
        ids[b, offs[b]:] = torch.tensor(r['toks'], device=E.dev)
        mask[b, offs[b]:] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    # knockouts: (B, T, T) bool, True = blocked (query row, key column) in padded coordinates
    E.loc['ko'] = None
    if ko:
        K = torch.zeros((B, T, T), dtype=torch.bool, device=E.dev)
        for b, r in enumerate(rows):
            o = offs[b]
            f0, f1 = r['frame']
            if ko in ('enc', 'both'):
                K[b, o + r['lst'][0]:o + r['lst'][1], o + f0:o + f1] = True
                if r['xpos'] is not None and r['xpos'] >= r['lst'][1]:          # world: X sits after the list
                    K[b, o + r['xpos'], o + f0:o + f1] = True
            if ko in ('ret', 'both'):
                K[b, o + r['after'][0]:o + r['after'][1], o + f0:o + f1] = True
        E.loc['ko'] = K
        E.loc['ko_layers'] = set(ko_layers) if ko_layers is not None else set(range(E.nL))
    # attention capture: queries = the last P tokens (part 2) of every row
    q0 = T - P
    masks = []
    for kind in ('x', 'lst', 'frame'):
        m = torch.zeros((B, T), device=E.dev)
        for b, r in enumerate(rows):
            o = offs[b]
            if kind == 'x' and r['xpos'] is not None:
                m[b, o + r['xpos']] = 1
            elif kind == 'lst':
                m[b, o + r['lst'][0]:o + r['lst'][1]] = 1
            elif kind == 'frame':
                m[b, o + r['frame'][0]:o + r['frame'][1]] = 1
        masks.append(m)
    E.loc['cap'] = dict(q0=q0, masks=masks)
    E.loc['att'] = {}
    # hooks: presence of X per layer over part 2; optional capture / patch of X's state
    pres = torch.zeros((E.nL, B), device=E.dev)
    xstate = {}
    hooks = []
    for l, blk in enumerate(E.blocks):
        def fn(mod, inp, out, l=l):
            h = E._h(out)
            changed = False
            if patch is not None and l == patch['layer']:
                h = h.clone()
                changed = True
                for b, r in enumerate(rows):
                    if r['xpos'] is not None and b in patch['vec']:
                        h[b, offs[b] + r['xpos']] = patch['vec'][b].to(h.dtype)
            if capture_x:
                xstate[l] = {b: h[b, offs[b] + r['xpos']].float().clone() for b, r in enumerate(rows) if r['xpos'] is not None}
            seg = h[:, q0:].float() - mu[l]
            a = atoms[l][widx]                                       # (B, d)
            pres[l] = (seg * a[:, None]).sum(-1).mean(-1) / sd[l][widx]
            if changed:
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return None
        hooks.append(blk.register_forward_hook(fn))
    out = E.model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=P + 1)
    for h in hooks:
        h.remove()
    E.loc['ko'] = None
    E.loc['cap'] = None
    lp = torch.log_softmax(out.logits.float(), -1)                       # (B, P+1, V)
    xid = torch.tensor([r['xid'] for r in rows], device=E.dev)
    lpx = lp[torch.arange(B), :, :].gather(-1, xid[:, None, None].expand(B, P + 1, 1))[:, :-1, 0]   # positions predicting part 2
    att = torch.stack([E.loc['att'][l] for l in range(E.nL)]) if E.loc['att'] else None      # (nL, B, 3)
    return dict(lpx=lpx.mean(-1).cpu().numpy(), pres=pres.cpu().numpy(), att=None if att is None else att.numpy(),
                xstate=xstate)


@torch.no_grad()
def locate(E, out_dir, source, P=40, tag='locate', jlens=None, bands=None, patch_layers=None):
    from lens_path import lens_path
    jlens = lens_path(jlens)
    R = json.load(open(source))['rows']
    true = {(r['seed'], r['k']): r for r in R if r['cond'] == 'true'}
    keys = sorted(true)
    plants = sorted({true[k]['plant'] for k in keys})
    atoms = lens_atoms(E, plants, jlens)
    mu, sd = calibrate_layers(E, atoms)
    install(E)
    widx = torch.tensor([plants.index(true[k]['plant']) for k in keys], device=E.dev)
    rows = {}
    for f in FRAMES:
        rows[f] = []
        for k in keys:
            t = true[k]
            p2 = t['turns'][1]['text']
            rows[f].append(build(E, t, t['turns'][0]['text'], [(w, c) for w, c in t['turns'][0]['words']], p2, f, P))
    # every row needs at least P part-2 tokens; drop short ones
    ok = [i for i in range(len(keys)) if all(r['p2'][1] - r['p2'][0] == P for r in (rows[f][i] for f in FRAMES))]
    for f in FRAMES:
        rows[f] = [rows[f][i] for i in ok]
    widx = widx[ok]
    log(f'{tag}: {len(ok)} of {len(keys)} pairs with >= {P} part-2 tokens')
    res = {}

    def go(name, frame, **kw):
        o = run(E, rows[frame], P, atoms, widx, mu, sd, **kw)
        res[name] = dict(lpx=o['lpx'].tolist(), pres=o['pres'].tolist(),
                         att=None if o['att'] is None else o['att'].tolist())
        return o

    for f in FRAMES:
        go(f, f, capture_x=(f == 'list'))
    for f in ('self', 'list', 'other', 'world'):
        for ko in ('enc', 'ret', 'both'):
            go(f'{f}+ko_{ko}', f, ko=ko)
    nL = E.nL
    bands = bands or [[int(x) for x in a] for a in np.array_split(np.arange(nL), 4)]
    for bi, band in enumerate(bands):
        for f in ('self', 'list'):
            go(f'{f}+ko_both_band{bi}', f, ko='both', ko_layers=band)
    lst = run(E, rows['list'], P, atoms, widx, mu, sd, capture_x=True)
    patch_layers = patch_layers or sorted(set(list(range(0, nL, max(1, nL // 9))) + [nL - 1]))
    for l in patch_layers:
        vec = {b: lst['xstate'][l][b] for b in lst['xstate'][l]}
        go(f'self+patch{l}', 'self', patch=dict(layer=l, vec=vec))
    json.dump(dict(exp=tag, model=E.model_id, L=E.L, nL=nL, P=P, bands=bands, patch_layers=patch_layers,
                   n=len(ok), res=res), open(f'{out_dir}/{tag}.json', 'w'))
    # summary
    base = np.array(res['true']['lpx'])
    lines = [f'{tag}: n={len(ok)}; readiness = mean log P(X) over {P} fixed part-2 tokens, minus the true frame;',
             f'presence at the read layer {E.L} and the layer of the largest self-vs-list gap; attention of part 2 to X',
             f'{"run":22s} readiness  presence@L  att->X(mean over layers)']
    for name in res:
        d = np.array(res[name]['lpx']) - base
        pr = np.array(res[name]['pres'])[E.L] - np.array(res['true']['pres'])[E.L]
        att = np.array(res[name]['att'])[:, :, 0].mean() if res[name]['att'] is not None else float('nan')
        lines.append(f'{name:22s} {d.mean():+7.2f}   {pr.mean():+7.2f}    {att:.4f}')
    gap = np.array(res['list']['pres']).mean(1) - np.array(res['self']['pres']).mean(1)
    lines.append('presence gap list - self by layer: ' + ' '.join(f'{v:+.2f}' for v in gap))
    open(f'{out_dir}/{tag}_summary.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))
