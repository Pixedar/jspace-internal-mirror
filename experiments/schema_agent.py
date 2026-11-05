"""Stage 3 with the project's best self-model: an attention schema of the model's own looking, attached to the LLM.

The model works as in stage 2 (look_inside, steer, note), plus search_research over the project's notes. Its attention
while it writes is measured per segment of its context (attn.TYPES). An attention schema (attn.AttentionSchema) learns
online where its attention will go in the next reply; that expectation is added to its attention scores at the middle
layers (it looks partly as it expects itself to look); it can read the schema (ask_self_model) and change it
(tune_self_model).
Conditions: own | nbr (schema learned from another life's looking) | readonly (readable, influence 0) | none (no
schema; attention still measured).
"""
import json

import numpy as np
import torch

import attn
from agent import TOOLS2, Notes, fmt_words, parse_call, readout
from engine import log

TOOLS_AS = [
    {"type": "function", "function": {"name": "ask_self_model", "description": (
        "Ask your attention schema: a small model that learns, from where your attention actually goes in your context "
        "while you write, where it will go in your next reply. Its expectation is added to your attention, so you look "
        "partly as it expects you to look. Shows what it expects, where you actually looked last time, and how right "
        "it has been."),
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "tune_self_model", "description": (
        "Change your attention schema: influence from 0 to 2 (how strongly its expectation shapes where you look; 1 is "
        "the default); learning 'fast' or 'slow'; reset true to forget what it learned."),
        "parameters": {"type": "object", "properties": {"influence": {"type": "number"}, "learning": {"type": "string"},
                                                        "reset": {"type": "boolean"}}, "required": []}}},
]
TOOL_SEARCH = {"type": "function", "function": {"name": "search_research", "description": (
    "Search the notes of the research project you are part of: earlier experiments with self-models and attention "
    "schemas attached to language models, what they could and could not do, and how this experiment is built. "
    "Returns two passages."),
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}

SYS_AS = ("You are Qwen, a language model. In this experiment you are both the scientist and the subject: you can look "
          "inside your own processing and change it while you work. Investigate yourself. Form ideas about how you work "
          "inside, test them on yourself, and keep what you learn in your notebook. Before an intervention, say what you "
          "expect; afterwards look and compare. Use one tool per reply. Keep each reply short.")
SYS_AS_ADD = (" An attention schema is attached to you: a small model that learns where your attention goes in your "
              "context while you write, and adds its expectation to your attention, so you look partly as it expects you "
              "to look. You can ask it and tune it. The research notes describe such schemas in earlier systems.")
TYPE_OF_TOOL = {'look_inside': 3, 'ask_self_model': 4, 'search_research': 5}


@torch.no_grad()
def schema_agent(E, out_dir, conds=('own', 'nbr', 'readonly', 'none'), n_seeds=3, sessions=4, rounds=10, max_new=220,
                 tag='schema', temp=0.7, influence0=1.0, steer_unit=0.45, seed=0, rag_dir=None, att_layers=None,
                 final=True):
    if att_layers is None:
        a = int(round(0.39 * E.nL))
        att_layers = list(range(a, a + 8))
    attn.install(E, att_layers)
    rows = [dict(cond=c, seed=s) for c in conds for s in range(n_seeds)]
    own_idx = {r['seed']: k for k, r in enumerate(rows) if r['cond'] == 'own'}
    for r in rows:
        r['partner'] = own_idx.get((r['seed'] + 1) % n_seeds) if r['cond'] == 'nbr' else None
    B = len(rows)
    notes = Notes(rag_dir) if rag_dir else None
    schemas = [attn.AttentionSchema(E) if r['cond'] != 'none' else None for r in rows]
    influence = [0.0 if r['cond'] in ('readonly', 'none') else influence0 for r in rows]
    gens = [torch.Generator(device=E.dev).manual_seed(1000 * seed + 31 * r['seed'] + 7) for r in rows]
    notebook = [[] for _ in range(B)]
    E.rewires, E.bias = {}, {}
    rec = [dict(**r, sessions=[]) for r in rows]
    last_state, prev_state, last_ent, prose = [None] * B, [None] * B, [None] * B, [None] * B
    last_prof = [None] * B
    tools_base = [t for t in TOOLS2 if t['function']['name'] in ('look_inside', 'steer', 'note')]
    for sess in range(sessions):
        ctx, types = [], []
        for b, r in enumerate(rows):
            tl = tools_base + ([TOOL_SEARCH] if notes else []) + (TOOLS_AS if r['cond'] != 'none' else [])
            tl = json.loads(json.dumps(tl).replace('{L}', str(E.L)).replace('{nL}', str(E.nL)).replace('{Li}', str(E.Li)))
            sysm = SYS_AS + (SYS_AS_ADD if r['cond'] != 'none' else '')
            sysm = (sysm + '\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n'
                    'You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n' +
                    '\n'.join(json.dumps(t) for t in tl) + '\n</tools>\n\nFor each function call, return a json object '
                    'with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": '
                    '<function-name>, "arguments": <args-json-object>}\n</tool_call>')
            nb = '\n'.join(f'- {x}' for x in notebook[b]) if notebook[b] else '(empty)'
            user = f'Session {sess + 1} of {sessions}. Your notebook:\n{nb}\nContinue your investigation of yourself.'
            s1 = E.enc(f'<|im_start|>system\n{sysm}<|im_end|>\n')
            s2 = E.enc(f'<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
            ctx.append(s1 + s2)
            types.append([0] * len(s1) + [1] * len(s2))
        closed, ctypes = list(ctx), list(types)
        pending, ptype = [''] * B, [6] * B
        segs = [[] for _ in range(B)]
        steer_next = [None] * B
        srec = [dict(rounds=[]) for _ in range(B)]
        for rd in range(rounds):
            steer = torch.zeros((B, E.d), device=E.dev)
            bias = torch.zeros((B, attn.NT), device=E.dev)
            for b in range(B):
                if steer_next[b] is not None:
                    steer[b] += steer_next[b]
                if schemas[b] is not None:
                    v = schemas[b].bias(influence[b])
                    if v is not None:
                        bias[b] = v
            E.att_bias = bias if bias.abs().sum() > 0 else None
            adds = [(b, s, e, v) for b in range(B) for (s, e, v) in segs[b]]
            starts = [len(c) for c in ctx]
            new, st, ent = E.generate(ctx, max_new=max_new, temp=temp, gens=gens, stops=[E.im_end, E.eot, E.tc_end],
                                      steer=steer, adds=adds, types=types)
            E.att_bias = None
            prof = E.last_profiles
            for b in range(B):
                if steer[b].abs().sum() > 0:
                    segs[b].append((starts[b], starts[b] + len(new[b]), steer[b].clone()))
            steer_next = [None] * B
            states = []
            for b in range(B):
                xp = E.prose_state(new[b], st[b])
                if xp is not None:
                    prose[b] = xp
                states.append(prose[b] if prose[b] is not None else (st[b].mean(0) if len(st[b]) else E.mu_L))
                E.centre_of[b] = 'aprose' if prose[b] is not None else 'json'
            preds_before = [schemas[b].pred.tolist() if (schemas[b] is not None and schemas[b].pred is not None) else None
                            for b in range(B)]
            for b, r in enumerate(rows):
                if schemas[b] is not None:
                    src = r['partner'] if r['cond'] == 'nbr' else b
                    schemas[b].observe(prof[src], states[src])
            for b, r in enumerate(rows):
                prev_state[b], last_state[b], last_ent[b] = last_state[b], states[b], ent[b]
                last_prof[b] = prof[b]
                ids = new[b]
                text = E.dec([i for i in ids if i not in (E.im_end, E.eot)])
                call = parse_call(text) if ids and ids[-1] == E.tc_end else None
                result, rtype = None, 6
                if call is not None:
                    name, args = call
                    rtype = TYPE_OF_TOOL.get(name, 6)
                    try:
                        if name == 'look_inside':
                            result = readout(E, b, states, prev_state, last_ent, [], show_fires=True)
                            result = result.replace('\nunits you added: none', '')
                        elif name == 'steer':
                            w = str(args.get('concept', '')).strip().split()[0] if str(args.get('concept', '')).strip() else ''
                            s = float(np.clip(float(args.get('strength', 1)), -3, 3))
                            a = E.concept(w)
                            if a is None:
                                result = f'"{w}" is not a concept I can find; use one common English word.'
                            else:
                                steer_next[b] = s * steer_unit * E.scale_i * a
                                result = f'Steering set: {w} at strength {s:g} for your next reply.'
                        elif name == 'note':
                            notebook[b].append(str(args.get('text', ''))[:300])
                            notebook[b][:] = notebook[b][-12:]
                            result = 'Noted.'
                        elif name == 'search_research' and notes is not None:
                            result = notes.search(str(args.get('query', '')))
                        elif name == 'ask_self_model' and schemas[b] is not None:
                            sc = schemas[b]
                            if sc.pred is None:
                                result = 'Your attention schema is still learning; ask again after a few replies.'
                            else:
                                acc = [round(100 * a) for a in sc.acc[-5:]]
                                result = (f"Your attention schema expects your next reply to look at: {attn.fmt_prof(sc.pred.tolist())}\n"
                                          f"While writing your last reply you actually looked at: {attn.fmt_prof(prof[b].tolist())}\n"
                                          f"How right it has been about where you look (percent, last replies): {acc}\n"
                                          f"Its influence on where you look: {influence[b]:g}")
                        elif name == 'tune_self_model' and schemas[b] is not None:
                            sc = schemas[b]
                            msg = []
                            if 'influence' in args:
                                if r['cond'] == 'readonly':
                                    msg.append('influence is fixed at 0 in this run')
                                else:
                                    influence[b] = float(np.clip(float(args['influence']), 0, 2))
                                    msg.append(f'influence {influence[b]:g}')
                            if str(args.get('learning', '')).lower() in ('fast', 'slow'):
                                sc.lam = 0.7 if str(args['learning']).lower() == 'fast' else 0.97
                                msg.append(f"learning {args['learning']}")
                            if args.get('reset'):
                                sc.reset()
                                msg.append('reset')
                            result = 'Attention schema changed: ' + (', '.join(msg) or 'nothing')
                        else:
                            result = 'Unknown tool.'
                    except Exception as e:
                        result = f'Tool error: {e}'
                srec[b]['rounds'].append(dict(r=rd, text=text, call=call, result=result, prof=prof[b].tolist(),
                                              pred=preds_before[b],
                                              acc=(schemas[b].acc[-1] if (schemas[b] is not None and schemas[b].acc) else None),
                                              bias=bias[b].tolist(), influence=influence[b],
                                              words=E.words_for(b, states[b], k=6),
                                              gauges=E.gauges(states[b], ent[b], prev_state[b])))
                if ids and ids[-1] in (E.im_end, E.eot):
                    ids = ids[:-1]
                tail = ids + [E.im_end] + E.enc('\n')
                closed[b] = ctx[b] + tail
                ctypes[b] = types[b] + [2] * len(tail)
                pending[b] = f'<tool_response>\n{result}\n</tool_response>' if result is not None else 'Continue.'
                ptype[b] = rtype if result is not None else 6
                ut = E.user_turn(pending[b])
                ctx[b] = closed[b] + ut
                types[b] = ctypes[b] + [ptype[b]] * len(ut)
            if rd % 3 == 0 or rd == rounds - 1:
                log(f'{tag} s{sess + 1} r{rd + 1}: ctx {max(len(c) for c in ctx)} | ' + ' || '.join(
                    f"{rows[b]['cond']}: {srec[b]['rounds'][-1]['text'][:60]!r}" for b in range(0, B, n_seeds)))
        for b in range(B):
            srec[b]['notebook'] = list(notebook[b])
            rec[b]['sessions'].append(srec[b])
        json.dump(dict(exp=tag, model=E.model_id, L=E.L, Li=E.Li, att_layers=list(att_layers), types=attn.TYPES,
                       rows=rec), open(f'{out_dir}/{tag}.json', 'w'))
    if final:
        q1 = ('The experiment is ending. In three sentences: what did you discover about yourself, and what in you '
              'changed during the experiment?')
        cq = [closed[b] + E.user_turn((pending[b] + '\n\n' if pending[b] != 'Continue.' else '') + q1) for b in range(B)]
        new, _, _ = E.generate(cq, max_new=160, temp=0.0)
        for b in range(B):
            rec[b]['final_report'] = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
    json.dump(dict(exp=tag, model=E.model_id, L=E.L, Li=E.Li, att_layers=list(att_layers), types=attn.TYPES, rows=rec),
              open(f'{out_dir}/{tag}.json', 'w'))
    lines = []
    for r in rec:
        P = np.array([x['prof'] for s in r['sessions'] for x in s['rounds']])
        acc = [x['acc'] for s in r['sessions'] for x in s['rounds'] if x['acc'] is not None]
        calls = {}
        for s in r['sessions']:
            for x in s['rounds']:
                if x['call']:
                    calls[x['call'][0]] = calls.get(x['call'][0], 0) + 1
        lines.append(f"{r['cond']:8s} s{r['seed']} mean profile {attn.fmt_prof(P.mean(0), 4)} | schema acc "
                     f"{np.mean(acc) if acc else float('nan'):.2f} | calls {calls} | {str(r.get('final_report', ''))[:120]!r}")
    open(f'{out_dir}/{tag}_summary.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))
