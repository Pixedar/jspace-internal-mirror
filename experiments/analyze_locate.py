"""Where the self-readout quarantine happens: gaps (list minus self) under each intervention, layer profiles, attention,
band knockouts, patching.   python analyze_locate.py FILE.json"""
import json
import sys
from math import comb

import numpy as np


def sign(d):
    d = [x for x in d if abs(x) > 1e-9]
    n, k = len(d), sum(1 for x in d if x > 0)
    return k, n, (min(1.0, sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2) if n else 1.0)


D = json.load(open(sys.argv[1]))
res, L, nL = D['res'], D['L'], D['nL']
lp = {k: np.array(v['lpx']) for k, v in res.items()}
pres = {k: np.array(v['pres']) for k, v in res.items()}          # (nL, n)
att = {k: (np.array(v['att']) if v['att'] is not None else None) for k, v in res.items()}   # (nL, n, 3)
print(f"{D['model']}  n={D['n']}  read layer {L} of {nL}")


def gap(a, b, measure='lp'):
    if measure == 'lp':
        d = lp[a] - lp[b]
    else:
        d = pres[a][L] - pres[b][L]
    k, n, p = sign(d)
    return f'{d.mean():+.2f} ({k}/{n}, p={p:.3f})'


print('\n1. Readiness to say X (mean log P over the fixed part 2) and presence at the read layer, vs the true frame:')
for f in ('self', 'list', 'other', 'world'):
    print(f'   {f:6s} readiness {gap(f, "true")}   presence {gap(f, "true", "pres")}')
print('\n2. The quarantine (list minus self) under each intervention:')
for suf in ('', '+ko_enc', '+ko_ret', '+ko_both'):
    print(f"   {('none' if not suf else suf[1:]):8s} readiness {gap('list' + suf, 'self' + suf)}   presence {gap('list' + suf, 'self' + suf, 'pres')}")
print('   effect of each knockout on the self frame alone (vs self):')
for suf in ('+ko_enc', '+ko_ret', '+ko_both'):
    print(f"   self{suf:9s} readiness {gap('self' + suf, 'self')}   presence {gap('self' + suf, 'self', 'pres')}")
print('   other minus self:', gap('other', 'self'), '| other+ko_both minus self+ko_both:', gap('other+ko_both', 'self+ko_both'))
print('\n3. Knockout of the frame in one band of layers (list minus self under it):')
for bi, band in enumerate(D['bands']):
    print(f'   layers {band[0]:2d}-{band[-1]:2d}: readiness {gap(f"list+ko_both_band{bi}", f"self+ko_both_band{bi}")}  '
          f'self vs self-no-KO {gap(f"self+ko_both_band{bi}", "self")}')
print('\n4. Patching X\'s state from the list run into the self run at one layer (fraction of the readiness gap closed):')
g0 = (lp['list'] - lp['self']).mean()
for l in D['patch_layers']:
    k = f'self+patch{l}'
    print(f'   layer {l:2d}: readiness {(lp[k] - lp["self"]).mean():+.2f}  closes {((lp[k] - lp["self"]).mean() / g0 if g0 else float("nan")):+.2f} of the gap')
print('\n5. Layer profile: presence of X in part 2, list minus self (standard deviations), and attention of part 2 to X')
gp = (pres['list'] - pres['self']).mean(1)
ax_s = att['self'][:, :, 0].mean(1)
ax_l = att['list'][:, :, 0].mean(1)
al_s = att['self'][:, :, 1].mean(1)
al_l = att['list'][:, :, 1].mean(1)
af_s = att['self'][:, :, 2].mean(1)
af_l = att['list'][:, :, 2].mean(1)
for l in range(nL):
    print(f'   layer {l:2d}: gap {gp[l]:+.3f} | att->X self {ax_s[l]:.4f} list {ax_l[l]:.4f} | att->list self {al_s[l]:.3f} '
          f'list {al_l[l]:.3f} | att->frame self {af_s[l]:.3f} list {af_l[l]:.3f}')
