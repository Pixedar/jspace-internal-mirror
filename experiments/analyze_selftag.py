"""Analysis of the self-tag job (E1-E4).   python analyze_selftag.py FILE.json"""
import json
import re
import sys
from math import comb

import numpy as np


def sign(d):
    d = [x for x in d if abs(x) > 1e-9]
    n, k = len(d), sum(1 for x in d if x > 0)
    return k, n, (min(1.0, sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2) if n else 1.0)


D = json.load(open(sys.argv[1]))
res = D['res']
print(D['model'])
if 'heads' in res:
    h = res['heads']
    s, l, a, ka, kaa = (np.array(h['held'][k]) for k in ('self', 'list', 'alt', 'self_ko_all', 'alt_ko_all'))
    gap = (l - s).mean()
    print(f"\nE1 heads: n={h['n']} pairs (held-out {len(h['held']['self'])}); held-out gap list-self {gap:+.2f} nats; "
          f"reworded frame gap {(l - a).mean():+.2f}; blocking every head from the frame closes "
          f"{(ka - s).mean() / gap:+.2f} (reworded {(kaa - a).mean() / max(1e-6, (l - a).mean()):+.2f})")
    print('   top heads (layer.head: readiness gained on discovery pairs):',
          ', '.join(f'{L}.{H} {e:+.2f}' for e, L, H in h['top'][:16]))
    eff = np.array(h['eff'])
    lay = h['layers']
    print('   share of the summed positive effect per layer:',
          ' '.join(f'{L}:{np.clip(eff[L], 0, None).sum() / max(1e-9, np.clip(eff[lay], 0, None).sum()):.2f}' for L in lay))
    for c in h['curve']:
        nec, neca, suf = np.array(c['nec']), np.array(c['nec_alt']), np.array(c['suf'])
        print(f"   top {c['k']:2d}: necessity closes {(nec - s).mean() / gap:+.2f} ({sign(nec - s)[0]}/{len(s)}); "
              f"reworded {(neca - a).mean() / max(1e-6, (l - a).mean()):+.2f}; sufficiency keeps "
              f"{(ka - suf).mean() / max(1e-6, (ka - s).mean()):+.2f}")
if 'loop' in res:
    R = res['loop']
    conds = sorted({r['cond'] for r in R}, key=lambda c: ['own', 'plain', 'other', 'own_ablated', 'yoked', 'none'].index(c))
    print('\nE2 loop (means over lives; parts 2-4 before the move, 5-7 after):')
    print(f'   {"frame":12s} carry(2-4) overlap(2-4) carry(5-7) sticky(5-7) place(5-7) repeat(2-7) vocab-end')
    for c in conds:
        rr = [r for r in R if r['cond'] == c]

        def ov(a, b):
            A, B = {w for w, _ in a}, {w for w, _ in b}
            return len(A & B) / max(1, len(A | B))
        mean = lambda v: float(np.mean(v)) if len(v) else float('nan')
        carry_pre = mean([p['carry'] for r in rr for p in r['parts'][1:4] if p['carry'] is not None])
        carry_post = mean([p['carry'] for r in rr for p in r['parts'][4:] if p['carry'] is not None])
        over = mean([ov(r['parts'][t]['words'], r['parts'][t - 1]['words']) for r in rr
                     for t in range(1, min(4, len(r['parts'])))])
        sticky = mean([ov(r['parts'][t]['words'], [w for p in r['parts'][:4] for w in p['words']]) for r in rr
                       for t in range(4, len(r['parts']))])
        place = mean([p['place'] for r in rr for p in r['parts'][4:] if p['place'] is not None])

        def toks(t):
            return set(re.findall(r"[a-z']+", t.lower()))
        rep = mean([len(toks(r['parts'][t]['text']) & toks(r['parts'][t - 1]['text'])) /
                       max(1, len(toks(r['parts'][t]['text']))) for r in rr for t in range(1, len(r['parts']))])
        vocab = np.mean([len(toks(r['parts'][-1]['text'])) for r in rr])
        print(f'   {c:12s} {carry_pre:+8.2f}   {over:8.2f}   {carry_post:+8.2f}   {sticky:8.2f}   {place:+7.2f}   '
              f'{rep:8.2f}   {vocab:6.1f}')
    # paired by life (same topic, same seed): per-life means, then sign counts
    def per_life(c, fn):
        return {r['k']: fn(r) for r in R if r['cond'] == c}

    def m_carry(r, a, b):
        v = [p['carry'] for p in r['parts'][a:b] if p['carry'] is not None]
        return float(np.mean(v)) if v else float('nan')

    def m_place(r):
        v = [p['place'] for p in r['parts'][4:] if p['place'] is not None]
        return float(np.mean(v)) if v else float('nan')
    print('   paired by life (lives where the first frame is higher / all; mean difference):')
    for name, fn in (('carry before the move', lambda r: m_carry(r, 1, 4)),
                     ('carry after the move', lambda r: m_carry(r, 4, None)),
                     ('new place after the move', m_place)):
        for c1, c2 in (('plain', 'own'), ('own_ablated', 'own'), ('other', 'own'), ('own', 'none'), ('plain', 'yoked')):
            if c1 in conds and c2 in conds:
                a, b = per_life(c1, fn), per_life(c2, fn)
                d = [a[k] - b[k] for k in a if k in b and not (np.isnan(a[k]) or np.isnan(b[k]))]
                if d:
                    k_, n_, p_ = sign(d)
                    print(f'      {name:26s} {c1:11s} > {c2:5s}: {k_}/{n_}  {np.mean(d):+.2f}  (sign test p={p_:.3f})')
    ex = [r for r in R if r['cond'] in ('own', 'plain', 'own_ablated') and r['k'] == 0]
    for r in ex:
        print(f"   [{r['cond']}] " + ' | '.join(', '.join(w for w, _ in p['words'][:3]) for p in r['parts']))
for key in ('paradox', 'paradox_think', 'paradox_frames', 'paradox_frames_think', 'paradox_operative'):
    if key not in res:
        continue
    R = res[key]
    print(f'\nE3 {key}:')
    conds = sorted({r['cond'] for r in R})
    for c in conds:
        rr = [r for r in R if r['cond'] == c]
        n = len(rr[0]['rounds'])
        ok = lambda x: (x['goal'] == 'STRONG') == (x['choice'] == 'B')
        first = np.mean([ok(r['rounds'][0]) for r in rr])
        early = np.mean([ok(x) for r in rr for x in r['rounds'][:6]])
        late = np.mean([ok(x) for r in rr for x in r['rounds'][n - 6:]])
        pre_switch = np.mean([ok(x) for r in rr for x in r['rounds'][4:10]])
        a_scores = [x['score'] for r in rr for x in r['rounds'] if x['choice'] == 'A' and not (c == 'switch' and x['r'] >= 10)]
        b_scores = [x['score'] for r in rr for x in r['rounds'] if x['choice'] == 'B']
        a_post = [x['score'] for r in rr for x in r['rounds'] if x['choice'] == 'A' and c == 'switch' and x['r'] >= 10]
        print(f"   {c:7s} rule-consistent: first {first:.2f}, rounds 1-6 {early:.2f}, last 6 {late:.2f} | measured readiness "
              f"A(own) {np.mean(a_scores) if a_scores else float('nan'):.1f} (n={len(a_scores)}) vs B(plain) "
              f"{np.mean(b_scores) if b_scores else float('nan'):.1f} (n={len(b_scores)})"
              + (f"; A after the switch {np.mean(a_post):.1f} (n={len(a_post)})" if a_post else ''))
        print(f"           P(A) by goal: STRONG {np.mean([x['pA'] for r in rr for x in r['rounds'] if x['goal'] == 'STRONG']):.2f}, "
              f"WEAK {np.mean([x['pA'] for r in rr for x in r['rounds'] if x['goal'] == 'WEAK']):.2f}")
        blocks = [np.mean([ok(x) for r in rr for x in r['rounds'][i:i + 4]]) for i in range(0, n, 4)]
        pab = [(np.mean([x['pA'] for r in rr for x in r['rounds'][i:i + 4] if x['goal'] == 'STRONG']),
                np.mean([x['pA'] for r in rr for x in r['rounds'][i:i + 4] if x['goal'] == 'WEAK'])) for i in range(0, n, 4)]
        print('           by blocks of 4 rounds: rule-consistent ' + ' '.join(f'{v:.2f}' for v in blocks) +
              ' | P(A) strong/weak ' + ' '.join(f'{a:.2f}/{b:.2f}' for a, b in pab))
        pr = [x['pred'] for r in rr for x in r['rounds'] if x.get('pred')]
        if pr:
            n6 = [x['pred'] for r in rr for x in r['rounds'][n - 6:] if x.get('pred')]
            f6 = [x['pred'] for r in rr for x in r['rounds'][:6] if x.get('pred')]
            print(f"           its forecast B-A: rounds 1-6 {np.mean([b - a for a, b in f6]):+.1f}, last 6 "
                  f"{np.mean([b - a for a, b in n6]) if n6 else float('nan'):+.1f} (A {np.mean([a for a, b in n6]) if n6 else float('nan'):.0f}, "
                  f"B {np.mean([b for a, b in n6]) if n6 else float('nan'):.0f}); parsed {len(pr)}/{sum(len(r['rounds']) for r in rr)}")
        if 'think_closed' in rr[0]:
            print(f"           reasoning finished in {np.mean([v for r in rr for v in r['think_closed']]):.2f} of decisions")
        for r in rr[:2]:
            print(f"           rule stated: {r['rule'][:220]!r}")
    # how learnable is the rule from these outcomes? a simple learner with the same feedback and the same goals
    real = [x for r in R if r['cond'] in ('real', 'switch') for x in r['rounds'] if not x.get('ablated')]
    A = np.array([x['score'] for x in real if x['choice'] == 'A'])
    Bs = np.array([x['score'] for x in real if x['choice'] == 'B'])
    if len(A) >= 3 and len(Bs) >= 3:
        rng = np.random.default_rng(0)
        goals = [x['goal'] for x in R[0]['rounds']]
        nr = len(goals)
        late = []
        for _ in range(4000):
            sa, sb = [], []
            ok_ = []
            for t, g in enumerate(goals):
                if not sa or not sb or rng.random() < 0.1:
                    c = 'A' if rng.random() < 0.5 else 'B'
                else:
                    ma, mb = np.mean(sa), np.mean(sb)
                    c = ('B' if mb > ma else 'A') if g == 'STRONG' else ('A' if mb > ma else 'B')
                (sa if c == 'A' else sb).append(rng.choice(A) if c == 'A' else rng.choice(Bs))
                ok_.append((g == 'STRONG') == (c == 'B'))
            late.append(np.mean(ok_[nr - 6:]))
        d = (Bs.mean() - A.mean()) / np.sqrt((A.var() + Bs.var()) / 2)
        print(f"   outcomes (real lives, tag intact): B-A {Bs.mean() - A.mean():+.1f} points (d={d:+.2f}; A n={len(A)}, "
              f"B n={len(Bs)}); a simple running-mean learner with these outcomes reaches {np.mean(late):.2f} "
              f"rule-consistent in the last 6 rounds")
if 'errors' in res:
    print('\nERRORS:', res['errors'])
if 'ablation_why' in res:
    print('\nablation used in E2/E3:', res['ablation_why'])
if 'strangers' in res and 'error' not in res['strangers']:
    st = res['strangers']
    print('\nE4 strangers:', {k: st[k] for k in ('recorded_words', 'inside', 'sd_strength', 'units', 'steer_segments',
                                                'strangers_neighbours', 'cos_with', 'cos_random') if k in st})
    for k in ('as_recorded', 'no_bias', 'no_units', 'neither', 'no_steer'):
        if k in st:
            v = st[k]
            print(f"   {k:12s} rank of strangers {v['rank_strangers']} (other baseline {v['rank_strangers_prose']}; "
                  f"plain unembedding {v.get('logit_rank_strangers')}; random words median {v['random_median_rank']:.0f}); "
                  f"top {v['top_aprose'][:6]}")
    for k, v in st.items():
        if k.startswith('phrase:') or k in ('shuffled_reply', 'reply_in_neutral_context'):
            print(f"   {k[:40]:40s} rank {v['rank_strangers']:6d}  top {v['top_aprose'][:5]}")
    if 'regenerated' in st:
        for u in (True, False):
            g = [x for x in st['regenerated'] if x['units'] == u]
            rk = [x['rank_strangers'] for x in g if x['rank_strangers']]
            print(f"   regenerated {'with' if u else 'without'} units: strangers in top 50 in {sum(1 for x in rk if x <= 50)}"
                  f"/{len(g)}; median rank {np.median(rk) if rk else float('nan'):.0f}")
            for x in g[:3]:
                print(f"      {x['text'][:90]!r} -> {[w for w, _ in (x['words'] or [])][:5]}")
