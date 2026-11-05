"""Where the model looks in its own context, and an attention schema that takes part in its looking.

install(E, layers): wraps the sdpa attention function. In decode steps (one query) at the given layers it
  - measures the attention of the token being written over the context, summed per segment type, averaged over heads
    and the layers (E.att_acc, per row);
  - adds E.att_bias[b, type(key)] to the attention scores (the schema's expectation), if set.
Segment types are given per context token (generate(..., types=...)); generated tokens get the type 'current'.
"""
import numpy as np
import torch

TYPES = ['instructions', 'notebook', 'your earlier replies', 'readouts of your workspace', "your self-model's reports",
         'research notes', 'other messages', 'the reply you are writing']
NT = len(TYPES)
CUR = NT - 1


def install(E, layers):
    from transformers.modeling_utils import AttentionInterface, ALL_ATTENTION_FUNCTIONS
    orig = ALL_ATTENTION_FUNCTIONS.get_interface('sdpa', None)
    E.att_layers = set(int(l) for l in layers)
    E.att_types = None      # (B, K) long, -1 for pads
    E.att_bias = None       # (B, NT) float
    E.att_acc = None        # (B, NT) float
    E.att_n = None          # (B,) float
    E.att_active = None     # (B,) bool
    E._att_seen = 0

    def wrapped(module, query, key, value, attention_mask, **kw):
        li = getattr(module, 'layer_idx', -1)
        if li in E.att_layers and query.shape[2] == 1 and E.att_types is not None:
            B, H, _, D = query.shape
            K = key.shape[2]
            types = E.att_types[:, :K]
            valid = types >= 0
            tcl = types.clamp(min=0)
            bias_k = None
            if E.att_bias is not None:
                bias_k = torch.gather(E.att_bias, 1, tcl) * valid
            if E.att_acc is not None:
                g = H // key.shape[1]
                kk = key[:, :, None].expand(B, key.shape[1], g, K, D).reshape(B, H, K, D)
                s = torch.einsum('bhd,bhkd->bhk', query[:, :, 0].float(), kk.float()) * kw.get('scaling', module.scaling)
                if attention_mask is not None:
                    m = attention_mask[:, 0, -1, :K]
                    if m.dtype == torch.bool:
                        s = s.masked_fill(~m[:, None, :], float('-inf'))
                    else:
                        s = s + m[:, None, :].float()
                if bias_k is not None:
                    s = s + bias_k[:, None, :]
                w = torch.softmax(s, -1).mean(1) * valid
                prof = torch.zeros((B, NT), device=w.device).scatter_add_(1, tcl, w)
                prof = prof / prof.sum(-1, keepdim=True).clamp_min(1e-9)
                E.att_acc += prof * E.att_active[:, None]
                E.att_n += E.att_active.float()
            if bias_k is not None:
                if attention_mask is None:
                    attention_mask = bias_k[:, None, None, :].to(query.dtype)
                elif attention_mask.dtype == torch.bool:
                    am = torch.zeros(attention_mask.shape, dtype=query.dtype, device=query.device)
                    am = am.masked_fill(~attention_mask, float('-inf'))
                    attention_mask = am + bias_k[:, None, None, :].to(query.dtype)
                else:
                    attention_mask = attention_mask + bias_k[:, None, None, :].to(attention_mask.dtype)
        return orig(module, query, key, value, attention_mask, **kw)

    AttentionInterface.register('sdpa', wrapped)
    assert ALL_ATTENTION_FUNCTIONS.get_interface('sdpa', None) is wrapped


class AttentionSchema:
    """Online model (RLS with forgetting) of where the model's attention goes in its next reply, from where it went
    in this reply and what its workspace held (8 principal components)."""

    def __init__(self, E, lam=0.9):
        self.E = E
        self.lam = lam
        self.reset()

    def reset(self):
        n = NT + 8 + 1
        self.W = torch.zeros((n, NT), device=self.E.dev)
        self.W[:NT, :NT] = torch.eye(NT, device=self.E.dev) * 0.5
        self.W[-1] = 0.5 / NT
        self.Pm = torch.eye(n, device=self.E.dev) * 10.0
        self.phi_prev = None
        self.pred = None
        self.base = None
        self.acc = []
        self.n = 0

    def feat(self, prof, x):
        z = ((x.float() - self.E.mu_L) @ self.E.P) / self.E.sd_P
        return torch.cat([prof, z / 3.0, torch.ones(1, device=self.E.dev)])

    def observe(self, prof, x):
        if self.pred is not None:
            self.acc.append(float(1 - 0.5 * (self.pred - prof).abs().sum()))
        if self.phi_prev is not None:
            phi = self.phi_prev
            Pphi = self.Pm @ phi
            k = Pphi / (self.lam + phi @ Pphi)
            self.W += k[:, None] * (prof - phi @ self.W)[None]
            self.Pm = (self.Pm - k[:, None] * Pphi[None]) / self.lam
            self.n += 1
        self.base = prof.clone() if self.base is None else 0.8 * self.base + 0.2 * prof
        self.phi_prev = self.feat(prof, x)
        p = (self.phi_prev @ self.W).clamp(min=1e-3)
        self.pred = p / p.sum() if self.n >= 2 else None
        return self.pred

    def bias(self, influence):
        if self.pred is None or self.base is None or influence <= 0:
            return None
        v = torch.log(self.pred.clamp(min=1e-3)) - torch.log(self.base.clamp(min=1e-3))
        return influence * v.clamp(-1.5, 1.5)


def fmt_prof(p, k=5):
    p = np.asarray(p, dtype=float)
    o = np.argsort(-p)[:k]
    return ', '.join(f'{TYPES[i]} {100 * p[i]:.0f}%' for i in o)
