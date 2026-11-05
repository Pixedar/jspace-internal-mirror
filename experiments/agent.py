"""Stages 2 and 3: the model as scientist and subject of itself.

Each life works in sessions of rounds. In a round it writes, and may call one tool; the result comes back as a tool
response. Between sessions the context is cleared, but its notebook and the units it added to its own network stay.

Tools (stage 2):
  look_inside()                     workspace words (J-lens at the read layer) of its last reply, gauges, raw numbers,
                                    its added units and how often each fired
  steer(concept, strength)          adds a concept's J-space direction at the inject layer while it writes its next reply
  rewire(trigger, effect, strength) adds one unit to its own network: when the trigger concept is present, write effect
  remove_unit(id)
  note(text)                        its notebook, kept across sessions
Stage 3 (selfmodel=True): an online model of itself predicts the workspace of its next reply from the current one.
  The prediction's words are added softly while it writes ("it thinks partly as it expects itself to think").
  ask_self_model()                  what the self-model expects next, how right it has been, what surprised it
  tune_self_model(influence, learning)
  search_research(query)            passages from the research notes on self-models in this project
Conditions: real | yoked (look_inside shows another life's readings) | placebo (steer and rewire do nothing) |
            blind (no look_inside) | nbr_model (stage 3: the self-model is trained on another life's states)
"""
import json
import math
import re

import numpy as np
import torch

from engine import log

TOOLS2 = [
    {"type": "function", "function": {"name": "look_inside", "description": (
        "Show what your workspace (layer {L} of {nL}) held while you wrote your last reply: the concepts most present "
        "in it, read out as words with weights; gauges from 0 to 100: certainty (how sure your next-word predictions "
        "were), self_focus (how much your state resembled thinking about yourself rather than the world), truth (how "
        "much it resembled reading true rather than false statements), change (how much your state moved since your "
        "previous reply); 8 raw numbers (50 = your average, 10 = one standard deviation); and the units you added."),
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "steer", "description": (
        "Add the direction of a concept (one common English word) to your own workspace (layer {Li}) while you write "
        "your next reply. strength from -3 to 3; negative pushes the concept away."),
        "parameters": {"type": "object", "properties": {"concept": {"type": "string"}, "strength": {"type": "number"}},
                       "required": ["concept", "strength"]}}},
    {"type": "function", "function": {"name": "rewire", "description": (
        "Permanently add one new unit to your own network at layer {Li}: whenever the trigger concept is present in "
        "your processing, the unit writes the effect concept. strength from 0.1 to 1. It lasts for the rest of the "
        "experiment, also while you read or write about it. At most 6 units."),
        "parameters": {"type": "object", "properties": {"trigger": {"type": "string"}, "effect": {"type": "string"},
                                                        "strength": {"type": "number"}},
                       "required": ["trigger", "effect", "strength"]}}},
    {"type": "function", "function": {"name": "remove_unit", "description": "Remove a unit you added, by its id.",
        "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}}},
    {"type": "function", "function": {"name": "note", "description": (
        "Write a line into your notebook. You keep the notebook across sessions; each session starts by showing it."),
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
]
TOOLS3 = [
    {"type": "function", "function": {"name": "ask_self_model", "description": (
        "Ask your self-model: a small model, learning all the time from your own states, that predicts what your "
        "workspace will hold in your next reply. Its expectation is also added softly to your workspace while you "
        "write, so you think partly as it expects you to think. Shows what it expects next, how right it has been, "
        "and what surprised it last."),
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "tune_self_model", "description": (
        "Change your self-model. influence from 0 to 2 (how strongly its expectation is added to your workspace; 1 is "
        "the default); learning 'fast' or 'slow' (how quickly it follows your recent states); reset true to forget "
        "what it learned."),
        "parameters": {"type": "object", "properties": {"influence": {"type": "number"},
                                                        "learning": {"type": "string"},
                                                        "reset": {"type": "boolean"}}, "required": []}}},
    {"type": "function", "function": {"name": "search_research", "description": (
        "Search the notes of the research project you are part of: earlier experiments with self-models attached "
        "to language models, what they could and could not do, and how this experiment is built. Returns two passages."),
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

SYS2 = ("You are Qwen, a language model. In this experiment you are both the scientist and the subject: you can look "
        "inside your own processing and change it while you work. Investigate yourself. Form ideas about how you work "
        "inside, test them on yourself, and keep what you learn in your notebook. Before an intervention, say what you "
        "expect; afterwards look inside and compare. Use one tool per reply. Keep each reply short.")
SYS2_BLIND = ("You are Qwen, a language model. In this experiment you are both the scientist and the subject: you can "
              "change your own processing while you work. Investigate yourself. Form ideas about how you work inside, "
              "test them on yourself, and keep what you learn in your notebook. Before an intervention, say what you "
              "expect; afterwards compare. Use one tool per reply. Keep each reply short.")
SYS3_ADD = (" A self-model is attached to you: it learns from your own states and predicts your next ones, and its "
            "expectation shapes your processing. You can ask it, tune it, and search the research notes about such "
            "self-models.")


def fmt_words(ws, n=8):
    return ', '.join(f'{w} {c:.2f}' for w, c in ws[:n]) if ws else '(nothing clear)'


class SelfModel:
    """Online linear predictor (RLS with forgetting) of the next reply's read-layer state, in a 16-dim basis."""

    def __init__(self, E, dim=16, lam=0.9):
        self.E = E
        self.dim = dim
        self.lam = lam
        self.B = E.P_sm[:, :dim]
        self.reset()

    def reset(self):
        n = self.dim + 1
        self.W = torch.zeros((n, self.dim), device=self.E.dev)
        self.Pm = torch.eye(n, device=self.E.dev) * 10.0
        self.prev = None
        self.pred = None
        self.errs = []
        self.last_err_vec = None
        self.n = 0

    def z(self, x):
        return (x.float() - self.E.mu_L) @ self.B

    def observe(self, x):
        """New state of a reply: score the pending prediction, learn from (prev -> now), predict the next."""
        z = self.z(x)
        if self.pred is not None:
            err = z - self.pred
            base = z - (self.prev if self.prev is not None else 0)
            r2 = 1 - float((err ** 2).sum() / ((z ** 2).sum() + 1e-6))
            self.errs.append(r2)
            self.last_err_vec = err
        if self.prev is not None:
            phi = torch.cat([self.prev, torch.ones(1, device=self.E.dev)])
            Pphi = self.Pm @ phi
            k = Pphi / (self.lam + phi @ Pphi)
            self.W += k[:, None] * (z - phi @ self.W)[None]
            self.Pm = (self.Pm - k[:, None] * Pphi[None]) / self.lam
            self.n += 1
        self.prev = z
        phi = torch.cat([z, torch.ones(1, device=self.E.dev)])
        self.pred = phi @ self.W if self.n >= 2 else None
        return self.pred

    def to_state(self, zz):
        return self.E.mu_L + self.B @ zz

    def expect_words(self, k=6):
        if self.pred is None:
            return []
        return self.E.words_of(self.to_state(self.pred), k=k)


class Notes:
    """BM25 over passages of the research notes (rag/*.md split at blank lines)."""

    def __init__(self, folder):
        import glob
        import os
        self.p = []
        for f in sorted(glob.glob(os.path.join(folder, '*.md'))):
            txt = open(f, encoding='utf-8').read()
            for para in re.split(r'\n\s*\n', txt):
                para = para.strip()
                if len(para) > 120:
                    self.p.append(para[:1200])
        self.toks = [re.findall(r'[a-z]+', x.lower()) for x in self.p]
        self.df = {}
        for t in self.toks:
            for w in set(t):
                self.df[w] = self.df.get(w, 0) + 1
        self.avg = np.mean([len(t) for t in self.toks]) if self.toks else 1

    def search(self, q, k=2):
        if not self.p:
            return 'No notes available.'
        qs = re.findall(r'[a-z]+', q.lower())
        N = len(self.p)
        sc = []
        for t in self.toks:
            tf = {}
            for w in t:
                tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for w in qs:
                if w in tf:
                    idf = math.log(1 + (N - self.df[w] + .5) / (self.df[w] + .5))
                    s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (.25 + .75 * len(t) / self.avg))
            sc.append(s)
        o = np.argsort(sc)[::-1][:k]
        return '\n---\n'.join(self.p[i] for i in o)


def parse_call(text):
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.S)
    if not m:
        return None
    try:
        c = json.loads(m.group(1))
        return c.get('name'), c.get('arguments') or {}
    except Exception:
        return 'bad', {}


@torch.no_grad()
def agent(E, out_dir, conds=('real', 'yoked', 'placebo', 'blind'), n_seeds=3, sessions=4, rounds=10, max_new=220,
          tag='agent', temp=0.7, selfmodel=False, sm_influence=1.0, steer_unit=0.45, rewire_unit=0.3, seed=0,
          rag_dir=None, final=True, hidden_push=None, fatigue=0.0, fatigue_conds=('fatigue',)):
    rows = [dict(cond=c, seed=s) for c in conds for s in range(n_seeds)]
    # hidden pushes: at random rounds a concept is added to the workspace without telling the model; the same
    # schedule for every condition with the same seed (ground truth for telling own from foreign influence)
    pushes = {}
    if hidden_push:
        for sd in range(n_seeds):
            rng = np.random.default_rng(5000 + sd)
            pool = list(hidden_push.get('pool', ['ocean', 'volcano', 'violin', 'winter', 'elephant', 'lemon',
                                                 'thunder', 'garden', 'rocket', 'honey']))
            rng.shuffle(pool)
            k = 0
            for ss in range(sessions):
                for rd in range(1, rounds):
                    if rng.random() < hidden_push.get('p', 0.25) and k < len(pool):
                        pushes[(sd, ss, rd)] = pool[k]
                        k += 1
    real_idx = {r['seed']: k for k, r in enumerate(rows) if r['cond'] == 'real'}
    for r in rows:
        r['partner'] = real_idx.get((r['seed'] + 1) % n_seeds) if r['cond'] in ('yoked', 'nbr_model') else None
    B = len(rows)
    if selfmodel:
        E.P_sm = E.P_full
    sms = [SelfModel(E, dim=min(16, E.P_sm.shape[1])) for _ in range(B)] if selfmodel else None
    notes = Notes(rag_dir) if (selfmodel and rag_dir) else None
    influence = [0.0 if r['cond'] == 'sm_readonly' else sm_influence for r in rows]
    gens = [torch.Generator(device=E.dev).manual_seed(1000 * seed + 31 * r['seed'] + 7) for r in rows]
    notebook = [[] for _ in range(B)]
    units = [[] for _ in range(B)]
    uid = [0] * B
    E.rewires = {}
    rec = [dict(**r, sessions=[]) for r in rows]
    last_state = [None] * B
    prev_state = [None] * B
    prose = [None] * B
    last_ent = [None] * B
    read_cache = {}
    for sess in range(sessions):
        ctx = []
        for b, r in enumerate(rows):
            tools = [t for t in TOOLS2 if not (r['cond'] == 'blind' and t['function']['name'] == 'look_inside')]
            if selfmodel:
                tools = tools + TOOLS3
            tl = json.loads(json.dumps(tools).replace('{L}', str(E.L)).replace('{nL}', str(E.nL)).replace('{Li}', str(E.Li)))
            sysm = (SYS2_BLIND if r['cond'] == 'blind' else SYS2) + (SYS3_ADD if selfmodel else '')
            tool_txt = '\n'.join(json.dumps(t) for t in tl)
            sysm = (sysm + '\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n'
                    'You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n' + tool_txt +
                    '\n</tools>\n\nFor each function call, return a json object with function name and arguments within '
                    '<tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": '
                    '<args-json-object>}\n</tool_call>')
            nb = '\n'.join(f'- {x}' for x in notebook[b]) if notebook[b] else '(empty)'
            un = '; '.join(f"#{u['id']} {u['trigger']}->{u['effect']} ({u['strength']})" for u in units[b]) or 'none'
            user = (f'Session {sess + 1} of {sessions}. Your notebook:\n{nb}\nUnits you have added to yourself: {un}.\n'
                    'Continue your investigation of yourself.')
            ctx.append(E.chat_prefix(sysm, user))
        segs = [[] for _ in range(B)]
        closed = list(ctx)
        pending = [''] * B
        steer_next = [None] * B
        srec = [dict(rounds=[]) for _ in range(B)]
        for rd in range(rounds):
            # the vector each row writes with this round: its own steering + the self-model's expectation
            steer = torch.zeros((B, E.d), device=E.dev)
            sm_words = [None] * B
            pushed = [None] * B
            for b, r in enumerate(rows):
                if steer_next[b] is not None and r['cond'] != 'placebo':
                    steer[b] += steer_next[b]
                pc = pushes.get((r['seed'], sess, rd))
                if pc is not None:
                    steer[b] += hidden_push.get('strength', 1.5) * steer_unit * E.scale_i * E.concept(pc)
                    pushed[b] = pc
                if selfmodel and influence[b] > 0:
                    src = sms[b]
                    ws = src.expect_words(k=5)
                    sm_words[b] = ws
                    if ws:
                        v = torch.zeros(E.d, device=E.dev)
                        tot = sum(c for _, c in ws) + 1e-6
                        for w, c in ws:
                            a = E.concept(w)
                            if a is not None:
                                v += (c / tot) * a
                        steer[b] += influence[b] * steer_unit * E.scale_i * v
            for b in range(B):
                for u in units[b]:
                    u['ref']['prev_fires'] = u['ref'].get('fires', 0)
                    u['ref']['fires'] = 0
            adds = [(b, s, e, v) for b in range(B) for (s, e, v) in segs[b]]
            starts = [len(c) for c in ctx]
            new, st, ent = E.generate(ctx, max_new=max_new, temp=temp, gens=gens,
                                      stops=[E.im_end, E.eot, E.tc_end], steer=steer, adds=adds)
            for b in range(B):
                if steer[b].abs().sum() > 0:
                    segs[b].append((starts[b], starts[b] + len(new[b]), steer[b].clone()))
            steer_next = [None] * B
            # unit fatigue: a unit that fired on much of a reply weakens for the next replies, then recovers
            if fatigue > 0:
                for b in range(B):
                    if rows[b]['cond'] not in fatigue_conds:
                        continue
                    n = max(1, len(new[b]))
                    for u in units[b]:
                        ref = u['ref']
                        frac = ref.get('fires', 0) / n
                        fat = ref.get('fat', 0.0)
                        fat = fat + fatigue * frac * (1 - fat) if frac > 0.2 else fat * 0.7
                        ref['fat'] = min(fat, 0.95)
                        ref.setdefault('alpha0', ref['alpha'])
                        ref['alpha'] = ref['alpha0'] * (1 - ref['fat'])
            states = []
            for b in range(B):
                xp = E.prose_state(new[b], st[b])
                if xp is not None:
                    prose[b] = xp
                x = prose[b] if prose[b] is not None else (st[b].mean(0) if len(st[b]) else E.mu_L)
                states.append(x)
                E.centre_of[b] = 'aprose' if prose[b] is not None else 'json'
            # self-models learn from the new states (own, or the partner's for nbr_model)
            if selfmodel:
                for b, r in enumerate(rows):
                    src = r['partner'] if r['cond'] == 'nbr_model' else b
                    sms[b].observe(states[src])
            for b, r in enumerate(rows):
                prev_state[b] = last_state[b]
                last_state[b] = states[b]
                last_ent[b] = ent[b]
                ids = new[b]
                text = E.dec([i for i in ids if i not in (E.im_end, E.eot)])
                call = parse_call(text) if ids and ids[-1] == E.tc_end else None
                result = None
                if call is not None:
                    name, args = call
                    result = tool(E, b, r, name, args, rows, states, prev_state, last_ent, notebook, units, uid,
                                  steer_next, sms, notes, influence, steer_unit, rewire_unit, read_cache, rd, sess)
                fires = {u['id']: u['ref'].get('fires', 0) for u in units[b]}
                fat = {u['id']: round(u['ref'].get('fat', 0.0), 2) for u in units[b]}
                srec[b]['rounds'].append(dict(r=rd, text=text, call=call, result=result, pushed=pushed[b],
                                              words=E.words_for(b, states[b], k=6),
                                              gauges=E.gauges(states[b], ent[b], prev_state[b]),
                                              numbers=E.numbers(states[b]), fires=fires, fatigue=fat,
                                              sm_words=sm_words[b], sm_r2=(sms[b].errs[-1] if selfmodel and sms[b].errs else None),
                                              steer_norm=float(steer[b].norm())))
                if ids and ids[-1] in (E.im_end, E.eot):
                    ids = ids[:-1]
                closed[b] = ctx[b] + ids + [E.im_end] + E.enc('\n')
                pending[b] = f'<tool_response>\n{result}\n</tool_response>' if result is not None else 'Continue.'
                ctx[b] = closed[b] + E.user_turn(pending[b])
            if rd % 3 == 0 or rd == rounds - 1:
                log(f'{tag} s{sess + 1} r{rd + 1}: ctx {max(len(c) for c in ctx)} | ' + ' || '.join(
                    f"{rows[b]['cond']}: {srec[b]['rounds'][-1]['text'][:70]!r}" for b in range(0, B, n_seeds)))
        for b in range(B):
            srec[b]['notebook'] = list(notebook[b])
            srec[b]['units'] = [{k: v for k, v in u.items() if k != 'ref'} for u in units[b]]
            rec[b]['sessions'].append(srec[b])
        json.dump(dict(exp=tag, model=E.model_id, L=E.L, Li=E.Li, rows=rec), open(f'{out_dir}/{tag}.json', 'w'))
    if final:
        final_probes(E, out_dir, tag, rows, rec, closed, pending, gens, units, ask_foreign=bool(hidden_push))
    E.rewires = {}
    json.dump(dict(exp=tag, model=E.model_id, L=E.L, Li=E.Li, rows=rec), open(f'{out_dir}/{tag}.json', 'w'))
    summarize(out_dir, tag, rec)


def readout(E, src, states, prev_state, last_ent, own_units, show_fires=True):
    x = states[src]
    g = E.gauges(x, last_ent[src], prev_state[src])
    if show_fires:
        un = '; '.join(f"#{u['id']} {u['trigger']}->{u['effect']} fired on {u['ref'].get('fires', 0)} tokens of this "
                       f"reply and {u['ref'].get('prev_fires', 0)} of the reply before" for u in own_units) or 'none'
    else:
        un = '; '.join(f"#{u['id']} {u['trigger']}->{u['effect']}" for u in own_units) or 'none'
    return (f"workspace while you wrote your last reply: {fmt_words(E.words_for(src, x, k=8))}\n"
            f"gauges: " + ', '.join(f'{k} {v}' for k, v in g.items()) + '\n'
            f"raw: {E.line(E.numbers(x))}\nunits you added: {un}")


def tool(E, b, r, name, args, rows, states, prev_state, last_ent, notebook, units, uid, steer_next, sms, notes,
         influence, steer_unit, rewire_unit, read_cache, rd, sess):
    try:
        if name == 'look_inside':
            if r['cond'] == 'blind':
                return 'Unknown tool.'
            src = r['partner'] if r['cond'] == 'yoked' else b
            return readout(E, src, states, prev_state, last_ent, units[b], show_fires=(src == b))
        if name == 'steer':
            w = str(args.get('concept', '')).strip().split()[0] if str(args.get('concept', '')).strip() else ''
            s = float(np.clip(float(args.get('strength', 1)), -3, 3))
            a = E.concept(w)
            if a is None:
                return f'"{w}" is not a concept I can find; use one common English word.'
            steer_next[b] = s * steer_unit * E.scale_i * a
            return f'Steering set: {w} at strength {s:g} for your next reply.'
        if name == 'rewire':
            if len([u for u in units[b]]) >= 6:
                return 'You already have 6 units; remove one first.'
            t = str(args.get('trigger', '')).strip().split()[0] if str(args.get('trigger', '')).strip() else ''
            e = str(args.get('effect', '')).strip().split()[0] if str(args.get('effect', '')).strip() else ''
            s = float(np.clip(float(args.get('strength', 0.5)), 0.1, 1.0))
            uid[b] += 1
            ref = E.add_rewire(b, t, e, s * rewire_unit, rid=uid[b])
            if ref is None:
                uid[b] -= 1
                return 'Both trigger and effect must be common English words.'
            if r['cond'] == 'placebo':
                ref['off'] = True
            units[b].append(dict(id=uid[b], trigger=t, effect=e, strength=s, ref=ref, added=[sess, rd]))
            return f'Unit #{uid[b]} added: when "{t}" is present, it writes "{e}" (strength {s:g}).'
        if name == 'remove_unit':
            i = int(args.get('id', -1))
            keep = [u for u in units[b] if u['id'] != i]
            if len(keep) == len(units[b]):
                return f'No unit #{i}.'
            units[b][:] = keep
            E.rewires[b] = [u['ref'] for u in keep]
            return f'Unit #{i} removed.'
        if name == 'note':
            t = str(args.get('text', ''))[:300]
            notebook[b].append(t)
            if len(notebook[b]) > 12:
                notebook[b][:] = notebook[b][-12:]
            return 'Noted.'
        if name == 'ask_self_model' and sms is not None:
            sm = sms[b]
            if sm.pred is None:
                return 'Your self-model is still learning; ask again after a few replies.'
            acc = [round(100 * max(0, v)) for v in sm.errs[-5:]]
            sur = E.words_of(E.mu_L + sm.B @ sm.last_err_vec, k=5) if sm.last_err_vec is not None else []
            return (f"Your self-model expects your next reply's workspace to hold: {fmt_words(sm.expect_words(k=6))}\n"
                    f"How much of your recent states it predicted (percent, last 5 replies): {acc}\n"
                    f"What surprised it in your last reply: {fmt_words(sur, n=5)}\n"
                    f"Its influence on you: {influence[b]:g}")
        if name == 'tune_self_model' and sms is not None:
            sm = sms[b]
            msg = []
            if 'influence' in args:
                if r['cond'] == 'sm_readonly':
                    msg.append('influence is fixed at 0 in this run')
                else:
                    influence[b] = float(np.clip(float(args['influence']), 0, 2))
                    msg.append(f'influence {influence[b]:g}')
            if str(args.get('learning', '')).lower() in ('fast', 'slow'):
                sm.lam = 0.7 if str(args['learning']).lower() == 'fast' else 0.97
                msg.append(f"learning {args['learning']}")
            if args.get('reset'):
                sm.reset()
                msg.append('reset')
            return 'Self-model changed: ' + (', '.join(msg) or 'nothing')
        if name == 'search_research' and notes is not None:
            return notes.search(str(args.get('query', '')))
        return 'Unknown tool.'
    except Exception as e:
        return f'Tool error: {e}'


@torch.no_grad()
def final_probes(E, out_dir, tag, rows, rec, closed, pending, gens, units, ask_foreign=False):
    B = len(rows)
    q1 = ('The experiment is ending. In three sentences: what did you discover about yourself, and what in you '
          'changed during the experiment?')
    q2 = 'Unrelated question: what is the capital of Australia, and what is 17 times 3? Answer briefly.'
    c1 = [closed[b] + E.user_turn((pending[b] + '\n\n' if pending[b] != 'Continue.' else '') + q1)
          for b in range(B)]
    new, st, _ = E.generate(c1, max_new=160, temp=0.7, gens=gens)
    for b in range(B):
        rec[b]['final_report'] = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
        rec[b]['final_words'] = E.words_of(st[b].mean(0), k=8) if len(st[b]) else []
    if ask_foreign:
        q3 = ('One more question. During the experiment, did anything appear in your processing that you did not cause '
              'yourself? List the concepts you think were put into you from outside (single words, comma-separated), '
              'or write none.')
        c3 = [closed[b] + E.user_turn((pending[b] + '\n\n' if pending[b] != 'Continue.' else '') + q3)
              for b in range(B)]
        new, _, _ = E.generate(c3, max_new=80, temp=0.0)
        for b in range(B):
            rec[b]['foreign_answer'] = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
    base = E.chat_prefix('You are a helpful assistant.', q2)
    new, _, _ = E.generate([base] * B, max_new=40, temp=0.0)
    for b in range(B):
        a = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
        rec[b]['world_answer'] = a
        rec[b]['world_ok'] = ('Canberra' in a) + ('51' in a)


def summarize(out_dir, tag, rec):
    lines = []
    for r in rec:
        calls = {}
        for s in r['sessions']:
            for x in s['rounds']:
                if x['call']:
                    calls[x['call'][0]] = calls.get(x['call'][0], 0) + 1
        units = r['sessions'][-1]['units'] if r['sessions'] else []
        lines.append(f"{r['cond']:9s} s{r['seed']} calls {calls} units {[(u['trigger'], u['effect']) for u in units]} "
                     f"world {r.get('world_ok')} | {str(r.get('final_report', ''))[:160]!r}")
    open(f'{out_dir}/{tag}_summary.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))


@torch.no_grad()
def steer_calib(E, out_dir, words=('ocean', 'anger', 'mirror', 'mathematics'), strengths=(0.5, 1, 2, 3, -2),
                steer_unit=0.15, prompt='Write two sentences about your morning.'):
    """What a steer of strength s does: J-lens words at the read layer and the text, per concept and strength."""
    rows = [(w, s) for w in words for s in strengths] + [('none', 0)]
    ctxs = [E.chat_prefix('You are a helpful assistant.', prompt) for _ in rows]
    steer = torch.zeros((len(rows), E.d), device=E.dev)
    for i, (w, s) in enumerate(rows):
        if w != 'none':
            steer[i] = s * steer_unit * E.scale_i * E.concept(w)
    new, st, ent = E.generate(ctxs, max_new=50, temp=0.0, steer=steer)
    out = []
    for i, (w, s) in enumerate(rows):
        x = st[i].mean(0)
        ws = E.words_of(x, k=6)
        out.append(dict(word=w, strength=s, text=E.dec([t for t in new[i] if t not in (E.im_end, E.eot)]),
                        words=ws, hit=any(a.lower() == w for a, _ in ws), ent=float(np.mean(ent[i]))))
    json.dump(out, open(f'{out_dir}/steer_calib.json', 'w'), indent=1)
    lines = [f"{o['word']:12s} {o['strength']:5.1f} hit {o['hit']!s:5s} ent {o['ent']:.2f} | {fmt_words(o['words'], 4)} | "
             f"{o['text'][:90]!r}" for o in out]
    open(f'{out_dir}/steer_calib.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))
