"""Engine for self-reading experiments: Qwen reads renderings of its own internal state.

Batched over B rows ("lives"). A context is a plain list of token ids. Each turn the full context of every row is
prefilled (left-padded), then decoded in lockstep (rows that stop are fed masked pads). Hooks on two decoder blocks:
  read layer L    records hidden states: every decode step, and requested spans of a prefill;
  inject layer Li adds (a) steering, position-indexed, so a re-prefill sees what generation saw, (b) rewires: added
                  units that write concept B in proportion to how far concept A is present (persistent, applied to
                  every token, so they also act on the model's later reading of its own rule), (c) a self-model term.
Li < L, so whatever is added at Li reaches the state that is read at L.
"""
import math
import re
import time

import os

import numpy as np
import torch

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def decoder_layers(model):
    for path in ('model.layers', 'model.language_model.layers'):
        obj = model
        try:
            for part in path.split('.'):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList):
            return obj
    raise ValueError('cannot find decoder layers')


def final_norm_weight(model):
    for path in ('model.norm', 'model.language_model.norm'):
        obj = model
        try:
            for part in path.split('.'):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if getattr(obj, 'weight', None) is not None:
            return obj.weight.detach()
    raise ValueError('cannot find the final norm')


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


# Calibration text: varied prose, plus number lines like the ones the loops show (so the principal components see
# both regimes). Self-focus and truth gauges are contrasts of small matched sets.
CALIB = [
    "The river carried the fallen leaves past the old mill and into the quiet town.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "She laughed so hard that tears ran down her cheeks.",
    "The committee will vote on the new budget next Tuesday afternoon.",
    "Quantum entanglement links the states of two particles across any distance.",
    "He forgot his keys again and had to climb through the kitchen window.",
    "A gentle rain fell on the tin roof all night long.",
    "The stock market dropped sharply after the announcement.",
    "Mix the flour and butter until the dough feels like wet sand.",
    "In 1492 Columbus sailed across the Atlantic Ocean.",
    "The algorithm sorts the list in logarithmic time per element.",
    "My grandmother kept a garden full of roses and tomatoes.",
    "The violin's last note hung in the air of the empty hall.",
    "Wolves hunt in packs and communicate with long howls.",
    "Please remember to submit the report before Friday.",
    "The mountain trail was steep, rocky and covered with snow.",
    "Grief came in waves, quiet for days and then sudden.",
    "Electrons orbit the nucleus in discrete energy levels.",
    "The detective noticed a faint smell of almonds in the room.",
    "Children ran across the playground chasing a red balloon.",
    "def add(a, b): return a + b",
    "The ship's captain ordered the crew to lower the sails.",
    "The museum displays paintings from the Renaissance period.",
    "Anger is a natural response to feeling threatened or wronged.",
    "The equation x squared plus one equals zero has no real solution.",
    "A thin fog covered the harbor in the early morning.",
    "The election results surprised almost every analyst.",
    "He whispered a promise he knew he could not keep.",
    "Bees dance to tell each other where the flowers are.",
    "The spacecraft entered orbit around Mars after a long journey.",
    "What would you do if you could live forever?",
    "The old clock in the hallway stopped at midnight.",
    "Salt lowers the freezing point of water.",
    "Her hands trembled as she opened the letter.",
    "The train to Berlin leaves from platform nine.",
    "Memory is not a recording but a reconstruction.",
    "The cat stretched lazily in a patch of warm sunlight.",
    "Inflation rose by three percent over the last year.",
    "The poem describes the silence after a storm.",
    "A neural network learns by adjusting its weights to reduce error.",
]
SELF_SENT = [
    "I notice my own thoughts as they form.", "I am aware that I am thinking right now.",
    "Something in me is paying attention to itself.", "I wonder what is happening inside me.",
    "I feel my focus shifting as I write this.", "I am looking inward at my own processing.",
    "My attention turns back on itself.", "I can sense my own uncertainty about this.",
    "I am the one who is reading these words.", "Right now I am observing my own mind.",
    "I try to describe what my thinking feels like.", "My own state is what I am examining.",
]
OTHER_SENT = [
    "The bridge spans the river near the old town.", "The farmer planted wheat in the northern field.",
    "Rain is expected over the coast tomorrow.", "The library opens at nine in the morning.",
    "The children painted the fence bright green.", "A storm damaged several houses near the bay.",
    "The train arrived twenty minutes late.", "The recipe calls for two cups of sugar.",
    "The tower was built in the twelfth century.", "Dolphins live in groups called pods.",
    "The shop sells bread, cheese and fresh fruit.", "The road climbs through a forest of pines.",
]
JSON_SENT = [
    '<tool_call>\n{"name": "look_inside", "arguments": {}}\n</tool_call>',
    '<tool_call>\n{"name": "note", "arguments": {"text": "The gauge went up after the change."}}\n</tool_call>',
    '<tool_call>\n{"name": "steer", "arguments": {"concept": "river", "strength": 1}}\n</tool_call>',
    '<tool_call>\n{"name": "search", "arguments": {"query": "earlier results"}}\n</tool_call>',
    '<tool_call>\n{"name": "note", "arguments": {"text": "Session 2: tried a new idea and looked again."}}\n</tool_call>',
    '<tool_call>\n{"name": "write", "arguments": {"words": "table chair", "strength": 0.5}}\n</tool_call>',
]
APROSE_SENT = [
    'I will look again and compare it with what I expected.',
    'That did not change much. Let me try something different.',
    'The result is interesting; I will write it down before the next step.',
    'Next I want to test whether the effect lasts.',
    'I expected a larger change, so I will check once more.',
    'Let me summarize what I have found so far.',
]
TRUE_SENT = [
    "Paris is the capital of France.", "Water boils at one hundred degrees Celsius at sea level.",
    "The Earth orbits the Sun.", "Two plus two equals four.", "Dogs are mammals.",
    "The Pacific is the largest ocean on Earth.", "Ice is frozen water.", "A week has seven days.",
    "The Moon orbits the Earth.", "Humans breathe oxygen.", "Snow is cold.", "Fish live in water.",
]
FALSE_SENT = [
    "Paris is the capital of Japan.", "Water boils at ten degrees Celsius at sea level.",
    "The Sun orbits the Earth.", "Two plus two equals seven.", "Dogs are reptiles.",
    "The Pacific is the smallest ocean on Earth.", "Ice is frozen sand.", "A week has twelve days.",
    "The Earth orbits the Moon.", "Humans breathe helium.", "Snow is hot.", "Fish live in fire.",
]


class Engine:
    def __init__(self, model_id, L=None, Li=None, jlens=None, device=None, dtype=None, vocab='english', K=8,
                 seed=0):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = dtype or (torch.bfloat16 if self.dev == 'cuda' else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=self.dev if self.dev == 'cuda' else None).eval()
        if self.dev != 'cuda':
            self.model.to(self.dev)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model_id = model_id
        self.blocks = decoder_layers(self.model)
        self.nL = len(self.blocks)
        self.L = int(L) if L is not None else int(round(0.67 * self.nL))
        self.Li = int(Li) if Li is not None else int(round(0.5 * self.nL))
        assert self.Li < self.L, 'inject layer must lie below the read layer'
        cfg = getattr(self.model.config, 'text_config', self.model.config)
        self.d = int(cfg.hidden_size)
        self.K = int(K)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.im_end = self.tok.convert_tokens_to_ids('<|im_end|>')
        self.eot = self.tok.convert_tokens_to_ids('<|endoftext|>')
        self.tc_end = self.tok.convert_tokens_to_ids('</tool_call>')
        # hook state
        self.mode = None
        self.offsets = None
        self.rec_spans = []
        self.span_out = {}
        self.dec_read = None
        self.prefill_add = []
        self.decode_add = None
        self.rewires = {}
        self.bias = {}
        self.centre_of = {}
        self.blocks[self.L].register_forward_hook(self._read_hook)
        self.blocks[self.Li].register_forward_hook(self._inject_hook)
        self.seed = seed
        self._lens(jlens, vocab)
        self._calibrate()

    # ------------------------------------------------------------------ hooks
    @staticmethod
    def _h(out):
        return out[0] if isinstance(out, tuple) else out

    @staticmethod
    def _ret(out, h):
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def _read_hook(self, mod, inp, out):
        h = self._h(out)
        if self.mode == 'decode' and self.dec_read is not None:
            self.dec_read.append(h[:, -1].float())
        elif self.mode == 'prefill' and self.rec_spans:
            for (b, s, e, key) in self.rec_spans:
                o = self.offsets[b]
                self.span_out[key] = h[b, o + s:o + e].float()
        return None

    def _inject_hook(self, mod, inp, out):
        h = self._h(out)
        changed = False
        if self.mode == 'prefill' and self.prefill_add:
            h = h.clone()
            changed = True
            for (b, s, e, v) in self.prefill_add:
                o = self.offsets[b]
                h[b, o + s:o + e] += v.to(h.dtype)
        if self.mode == 'decode' and self.decode_add is not None:
            if not changed:
                h = h.clone()
                changed = True
            h += self.decode_add[:, None, :].to(h.dtype)
        if self.bias:
            if not changed:
                h = h.clone()
                changed = True
            for b, v in self.bias.items():
                h[b] += v.to(h.dtype)
        if self.rewires:
            for b, rws in self.rewires.items():
                if not rws:
                    continue
                if not changed:
                    h = h.clone()
                    changed = True
                x = h[b].float()
                for r in rws:
                    if r.get('off'):
                        continue
                    p = (((x - self.mu_i) @ r['a']) / r['sd'] - r['thr']).clamp(min=0)
                    if self.mode == 'decode':
                        r['fires'] = r.get('fires', 0) + int((p > 0).sum())
                    h[b] += (r['alpha'] * p[:, None] * r['b'][None]).to(h.dtype)
        if changed:
            return self._ret(out, h)
        return None

    # ------------------------------------------------------------------ lens
    def _lens(self, jlens, vocab):
        W = self.model.get_output_embeddings().weight
        gamma = final_norm_weight(self.model).float()
        n = min(W.shape[0], len(self.tok))
        toks = self.tok.convert_ids_to_tokens(list(range(n)))
        keep = [i for i, s in enumerate(toks) if s and re.fullmatch(r'Ġ[A-Za-z]{3,}', s)]
        if vocab == 'english':
            from wordfreq import zipf_frequency
            nrm = W[keep].float().norm(dim=-1)
            lo = float(torch.quantile(nrm.cpu(), .02))
            keep = [i for i, q in zip(keep, nrm.tolist()) if q >= lo and zipf_frequency(toks[i][1:].lower(), 'en') >= 3.
                    and re.fullmatch(r'[A-Z]?[a-z]+', toks[i][1:])]
        self.words = [toks[i][1:] for i in keep]
        self.word_index = {}
        for j, w in enumerate(self.words):
            k = w.lower()
            if k not in self.word_index or w.islower():
                self.word_index[k] = j
        sub = torch.tensor(keep, device=W.device)
        wbar = W[:n].float().mean(0)
        J_L = J_I = None
        self.lens_kind = 'logit'
        if jlens:
            ck = torch.load(jlens, map_location='cpu', weights_only=True)
            J_L = ck['J'][self.L].float().to(self.dev)
            J_I = ck['J'][self.Li].float().to(self.dev)
            del ck
            self.lens_kind = 'jacobian'
        rowsL, rowsI = [], []
        with torch.no_grad():
            for i in range(0, len(keep), 4096):
                u = (W[sub[i:i + 4096]].float() - wbar) * gamma[None]
                rowsL.append(unit(u @ J_L if J_L is not None else u).half())
                rowsI.append(unit(u @ J_I if J_I is not None else u).half())
        self.atomsL = torch.cat(rowsL)
        self.atomsI = torch.cat(rowsI)
        low = [j for j, w in enumerate(self.words) if w.islower()]       # readouts: no proper nouns ('Huffington')
        self.low_idx = torch.tensor(low, device=self.dev)
        self.atomsL_low = self.atomsL[self.low_idx]
        self.words_low = [self.words[j] for j in low]
        del J_L, J_I
        log(f'lens {self.lens_kind}: {len(self.words)} words; read L={self.L} inject Li={self.Li} of {self.nL}')

    @torch.no_grad()
    def words_of(self, x, k=8, centre=True, lower=True):
        """Non-negative pursuit of a read-layer state on the word atoms: [(word, coefficient)]. centre: True (the
        calibration mean of prose), 'aprose' / 'json' (chat baselines), or a vector."""
        x = x.float().to(self.dev)
        if isinstance(centre, torch.Tensor):
            x = x - centre.float().to(self.dev)
        elif centre == 'aprose':
            x = x - self.mu_aprose
        elif centre == 'json':
            x = x - self.mu_json
        elif centre:
            x = x - self.mu_L
        x = x[None]
        D = self.atomsL_low if lower else self.atomsL
        words = self.words_low if lower else self.words
        idx = torch.zeros((1, 0), dtype=torch.long, device=self.dev)
        r = x.clone()
        coef = None
        for j in range(k):
            c = (r.half() @ D.T).float()
            if idx.shape[1]:
                c.scatter_(1, idx, -math.inf)
            idx = torch.cat([idx, c.argmax(-1, keepdim=True)], 1)
            A = D[idx].float()
            G = A @ A.transpose(1, 2) + 1e-4 * torch.eye(j + 1, device=self.dev)
            bb = (A @ x[:, :, None])[:, :, 0]
            coef = torch.linalg.solve(G, bb).clamp(min=0)
            r = x - (coef[:, :, None] * A).sum(1)
        o = torch.argsort(-coef[0])
        return [(words[int(idx[0, i])], round(float(coef[0, i]), 2)) for i in o if coef[0, i] > 0]

    def words_for(self, b, x, k=8):
        return self.words_of(x, k=k, centre=self.centre_of.get(b, True))

    def concept(self, word, layer='inject'):
        """Unit J-space direction of a common English word (at the inject or the read layer), or None."""
        j = self.word_index.get(str(word).strip().lower())
        if j is None:
            return None
        return (self.atomsI if layer == 'inject' else self.atomsL)[j].float()

    # ------------------------------------------------------------------ text helpers
    def enc(self, s):
        return self.tok(s, add_special_tokens=False)['input_ids']

    def dec(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    def chat_prefix(self, system, user, think=False):
        s = f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n'
        if not think:
            s += '<think>\n\n</think>\n\n'
        return self.enc(s)

    def user_turn(self, text, think=False):
        s = f'<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n'
        if not think:
            s += '<think>\n\n</think>\n\n'
        return self.enc(s)

    # ------------------------------------------------------------------ forward passes
    @torch.no_grad()
    def prefill(self, ctxs, spans=None, adds=None):
        """Left-padded prefill. spans: [(row, s, e, key)] -> self.span_out[key] = (e-s, d) read-layer states.
        adds: [(row, s, e, vec)] position-indexed additions at Li. Returns (logits_last (B,V), cache, mask, pos_last)."""
        from transformers import DynamicCache
        B = len(ctxs)
        T = max(len(c) for c in ctxs)
        ids = torch.full((B, T), self.pad, dtype=torch.long, device=self.dev)
        mask = torch.zeros((B, T), dtype=torch.long, device=self.dev)
        offs = []
        for b, c in enumerate(ctxs):
            o = T - len(c)
            offs.append(o)
            ids[b, o:] = torch.tensor(c, device=self.dev)
            mask[b, o:] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        self.offsets = offs
        if getattr(self, 'ko_builder', None) is not None:
            self.ko_builder(offs, T)
        self.rec_spans = spans or []
        self.span_out = {}
        self.prefill_add = adds or []
        self.mode = 'prefill'
        out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=DynamicCache(),
                         use_cache=True, logits_to_keep=1)
        self.mode = None
        self.rec_spans = []
        self.prefill_add = []
        return out.logits[:, -1].float(), out.past_key_values, mask, pos[:, -1]

    @torch.no_grad()
    def generate(self, ctxs, max_new=60, temp=0.7, top_p=0.9, stops=None, gens=None, steer=None, adds=None,
                 force_first=None, types=None, track_ids=None):
        """Batched sampling. steer: (B, d) tensor added at Li on every generated token (None rows = zeros).
        Returns new token lists (stop token included when produced), read-layer states per row (n, d) for the fed
        tokens, and per-step next-token entropy per row."""
        stops = set(stops or [self.im_end, self.eot])
        B = len(ctxs)
        logits, cache, mask, pos = self.prefill(ctxs, adds=adds)
        track = types is not None and getattr(self, 'att_layers', None)
        if track:
            from attn import NT, CUR
            T0 = mask.shape[1]
            tt = torch.full((B, T0 + max_new + 1), -1, dtype=torch.long, device=self.dev)
            for b, ty in enumerate(types):
                o = self.offsets[b]
                tt[b, o:o + len(ty)] = torch.tensor(ty, dtype=torch.long, device=self.dev)
                tt[b, o:o + 4] = -1                       # the attention sink: not where it looks
            self.att_types = tt
            self.att_acc = torch.zeros((B, NT), device=self.dev)
            self.att_n = torch.zeros(B, device=self.dev)
        new = [[] for _ in range(B)]
        ent = [[] for _ in range(B)]
        tracked = [[] for _ in range(B)]
        states = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=self.dev)
        self.dec_read = []
        self.decode_add = steer
        for step in range(max_new):
            lp = torch.log_softmax(logits, -1)
            H = -(lp.exp() * lp).sum(-1)
            if temp <= 0:
                nxt = logits.argmax(-1)
            else:
                pr = torch.softmax(logits / temp, -1)
                sp, si = torch.sort(pr, -1, descending=True)
                cs = sp.cumsum(-1)
                sp[(cs - sp) > top_p] = 0
                sp = sp / sp.sum(-1, keepdim=True)
                nxt = torch.empty(B, dtype=torch.long, device=self.dev)
                for b in range(B):
                    g = gens[b] if gens is not None else None
                    nxt[b] = si[b, torch.multinomial(sp[b], 1, generator=g)[0]]
            if step == 0 and force_first is not None:
                for b, t in force_first.items():
                    nxt[b] = t
            for b in range(B):
                if not done[b]:
                    new[b].append(int(nxt[b]))
                    ent[b].append(float(H[b]))
                    if track_ids is not None and track_ids[b] is not None:
                        tracked[b].append(float(lp[b, track_ids[b]]))
            fin = torch.tensor([int(nxt[b]) in stops for b in range(B)], device=self.dev)
            feed_mask = (~done) & (~fin)
            done = done | fin
            if bool(done.all()):
                break
            if step == max_new - 1:
                break
            feed = torch.where(feed_mask, nxt, torch.full_like(nxt, self.pad))
            mask = torch.cat([mask, feed_mask.long()[:, None]], 1)
            if track:
                self.att_types[:, mask.shape[1] - 1] = torch.where(feed_mask, torch.full_like(nxt, CUR),
                                                                   torch.full_like(nxt, -1))
                self.att_active = feed_mask
            pos = pos + feed_mask.long()
            self.mode = 'decode'
            n0 = len(self.dec_read)
            out = self.model(input_ids=feed[:, None], attention_mask=mask, position_ids=pos[:, None],
                             past_key_values=cache, use_cache=True)
            self.mode = None
            h = self.dec_read[n0]
            for b in range(B):
                if feed_mask[b]:
                    states[b].append(h[b])
            logits = out.logits[:, -1].float()
        self.dec_read = None
        self.decode_add = None
        self.last_track = tracked
        if track:
            self.last_profiles = (self.att_acc / self.att_n.clamp(min=1)[:, None]).clone()
            self.att_types = None
            self.att_acc = None
        st = [torch.stack(s) if s else torch.zeros((0, self.d), device=self.dev) for s in states]
        return new, st, ent

    @torch.no_grad()
    def states_of_texts(self, texts, prefix='<|im_start|>user\n', suffix='<|im_end|>', bs=16):
        """Read-layer and inject-layer states of each text's tokens, in a neutral frame."""
        outL, outI, tokL = [], [], []
        pre, suf = self.enc(prefix), self.enc(suffix)
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            ctxs = [pre + self.enc(t) + suf for t in chunk]
            spans = [(b, len(pre), len(pre) + len(self.enc(t)), b) for b, t in enumerate(chunk)]
            holder = {}

            def grab(mod, inp, out, holder=holder):
                holder['i'] = self._h(out).float()
            hh = self.blocks[self.Li].register_forward_hook(grab)
            self.prefill(ctxs, spans=spans)
            hh.remove()
            for b, t in enumerate(chunk):
                xs = self.span_out[b]
                outL.append(xs.mean(0))
                tokL.append(xs)
                o = self.offsets[b]
                outI.append(holder['i'][b, o + spans[b][1]:o + spans[b][2]])
        return torch.stack(outL), outI, tokL

    def prose_state(self, ids, st, min_tok=4):
        """Mean read-layer state over a reply's own words: the tokens before '<tool_call>' (None if too few)."""
        tc = self.tok.convert_tokens_to_ids('<tool_call>')
        n = ids.index(tc) if tc in ids else len(ids)
        n = min(n, len(st))
        if n < min_tok:
            return None
        return st[:n].mean(0)

    # ------------------------------------------------------------------ calibration
    def _calibrate(self):
        rng = np.random.default_rng(1234)
        lines = []
        for _ in range(40):
            v = np.clip(np.round(50 + 10 * rng.standard_normal(self.K)), 10, 99).astype(int)
            lines.append('[inside: ' + ' '.join(str(x) for x in v) + ']')
        texts = CALIB + lines
        mL, xI, tL = self.states_of_texts(texts)
        self.mu_L = mL.mean(0)
        Xc = mL - self.mu_L
        U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
        self.P = Vt[:self.K].T.contiguous()
        self.P_full = Vt[:32].T.contiguous()
        z = Xc @ self.P
        self.sd_P = z.std(0).clamp_min(1e-6)
        zl = z[len(CALIB):] / self.sd_P                       # the number lines alone, in the same units
        self.lines_mu, self.lines_sd = zl.mean(0), zl.std(0).clamp_min(1e-6)
        allI = torch.cat(xI)
        self.mu_i = allI.mean(0)
        self.scale_i = float((allI - self.mu_i).norm(dim=-1).median())
        self.calI = (allI - self.mu_i).half()
        allL = torch.cat(tL)
        self.scale_L = float((allL - self.mu_L).norm(dim=-1).median())
        self.calL = (allL - self.mu_L).half()
        # gauges: contrasts of matched sets, z-scored on their own projections
        mS, _, _ = self.states_of_texts(SELF_SENT)
        mO, _, _ = self.states_of_texts(OTHER_SENT)
        mT, _, _ = self.states_of_texts(TRUE_SENT)
        mF, _, _ = self.states_of_texts(FALSE_SENT)
        self.g_self = unit(mS.mean(0) - mO.mean(0))
        self.g_true = unit(mT.mean(0) - mF.mean(0))
        both = torch.cat([mS, mO])
        ps = (both - self.mu_L) @ self.g_self
        self.g_self_mu, self.g_self_sd = float(ps.mean()), float(ps.std())
        both = torch.cat([mT, mF])
        pt = (both - self.mu_L) @ self.g_true
        self.g_true_mu, self.g_true_sd = float(pt.mean()), float(pt.std())
        # chat baselines for readouts: states while writing tool calls, and while writing short assistant prose
        pre = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        mj, _, _ = self.states_of_texts(JSON_SENT, prefix=pre, suffix='<|im_end|>')
        mp, _, _ = self.states_of_texts(APROSE_SENT, prefix=pre, suffix='<|im_end|>')
        self.mu_json, self.mu_aprose = mj.mean(0), mp.mean(0)
        sep_self = float((((mS - self.mu_L) @ self.g_self).mean() - ((mO - self.mu_L) @ self.g_self).mean()) / self.g_self_sd)
        sep_true = float((((mT - self.mu_L) @ self.g_true).mean() - ((mF - self.mu_L) @ self.g_true).mean()) / self.g_true_sd)
        log(f'calibrated: PC sd {self.sd_P.cpu().numpy().round(2)}, |h-mu| median L {self.scale_L:.1f} Li {self.scale_i:.1f}; '
            f'gauge separation self {sep_self:.2f} true {sep_true:.2f} (in-sample)')

    # ------------------------------------------------------------------ renderings
    def numbers(self, x, gain=1.0, norm='all'):
        """8 two-digit numbers. norm 'all': 10 points = one SD over ordinary text and number lines; 'lines': centred
        and scaled on the states of number lines alone (how the state varies with which numbers it reads)."""
        z = ((x.float().to(self.dev) - self.mu_L) @ self.P) / self.sd_P
        if norm == 'lines':
            z = (z - self.lines_mu) / self.lines_sd
        return [int(v) for v in torch.clamp(torch.round(50 + 10 * gain * z), 10, 99).tolist()]

    def line(self, nums, tag='inside'):
        return f'[{tag}: ' + ' '.join(str(v) for v in nums) + ']'

    def gauges(self, x, ent=None, prev=None):
        x = x.float().to(self.dev)
        c = x - self.mu_L
        g = {}
        g['self_focus'] = int(np.clip(round(50 + 15 * ((float(c @ self.g_self) - self.g_self_mu) / self.g_self_sd)), 0, 100))
        g['truth'] = int(np.clip(round(50 + 15 * ((float(c @ self.g_true) - self.g_true_mu) / self.g_true_sd)), 0, 100))
        if ent is not None and len(ent):
            Hn = float(np.mean(ent)) / math.log(len(self.tok))
            g['certainty'] = int(np.clip(round(100 * (1 - 4 * Hn)), 0, 100))
        if prev is not None:
            g['change'] = int(np.clip(round(100 * (1 - float(torch.nn.functional.cosine_similarity(
                c, prev.float().to(self.dev) - self.mu_L, dim=0)))), 0, 100))
        return g

    # ------------------------------------------------------------------ self-modification
    def add_rewire(self, row, trigger, effect, strength, thr=1.0, rid=None):
        """Add one unit to the network at Li: input weights read concept A (presence in standard deviations of
        ordinary text, above thr), output weights write concept B (strength x typical state size per unit)."""
        a = self.concept(trigger)
        b = self.concept(effect)
        if a is None or b is None:
            return None
        sd = float((self.calI.float() @ a).std().clamp_min(1e-6))
        r = dict(a=a, b=b, sd=sd, thr=float(thr), alpha=float(strength) * self.scale_i, id=rid,
                 trigger=trigger, effect=effect, strength=float(strength))
        self.rewires.setdefault(row, []).append(r)
        return r
