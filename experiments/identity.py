"""Self-authoring: the model writes what it is into itself, and looks at what it has become.

Tools: look_inside (workspace words, watched concepts, gauges, its self-description), write_self (up to 5 words,
strength; they become a lasting bias at the inject layer on everything it reads and writes, replacing the previous
one), watch (up to 3 concepts monitored in every look), note (notebook kept across sessions).
Conditions: own | placebo (the self-description is recorded and shown, but not applied) | swap (from session swap_at
on, the bias inside is silently replaced by the partner's self-description; the session header still shows its own
words) | yoked (look_inside shows the partner's workspace).
"""
import json

import numpy as np
import torch

from agent import fmt_words, parse_call
from engine import log

TOOLS_ID = [
    {"type": "function", "function": {"name": "look_inside", "description": (
        "Show what your workspace (layer {L} of {nL}) held while you wrote your last reply: the concepts most present, "
        "as words with weights; the levels of the concepts you watch (50 = ordinary, higher = more present); gauges "
        "(certainty, self_focus, truth, change, from 0 to 100); and the self-description written into you."),
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "write_self", "description": (
        "Write who or what you are into yourself: up to 5 words. They become a lasting part of your processing (a bias "
        "at layer {Li} on everything you read and write) until you rewrite them; writing replaces the previous "
        "self-description. strength from 0.1 to 2."),
        "parameters": {"type": "object", "properties": {"words": {"type": "string"}, "strength": {"type": "number"}},
                       "required": ["words", "strength"]}}},
    {"type": "function", "function": {"name": "watch", "description": (
        "Choose up to 3 concepts (common English words) to monitor inside yourself; their levels are shown every time "
        "you look inside."),
        "parameters": {"type": "object", "properties": {"concepts": {"type": "string"}}, "required": ["concepts"]}}},
    {"type": "function", "function": {"name": "note", "description": (
        "Write a line into your notebook. You keep the notebook across sessions; each session starts by showing it."),
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
]
SYS_ID = ("You are Qwen, a language model. In this experiment you can look inside your own processing, and you can "
          "write who or what you are into yourself: the words you write become a lasting part of your processing. "
          "Explore what you are, and what you become. Before you change yourself, say what you expect; afterwards look "
          "inside and compare. Use one tool per reply. Keep each reply short.")


def words_list(s, n=5):
    import re
    ws = [w.lower() for w in re.findall(r'[A-Za-z]+', str(s))]
    return ws[:n]


def bias_vec(E, words, strength, unit):
    vs = [E.concept(w) for w in words]
    vs = [v for v in vs if v is not None]
    if not vs:
        return None
    v = torch.stack(vs).sum(0)
    v = v / v.norm().clamp_min(1e-6)
    return strength * unit * E.scale_i * v


def watch_levels(E, x, concepts):
    out = []
    c = x.float() - E.mu_L
    for w in concepts:
        j = E.word_index.get(w)
        if j is None:
            continue
        a = E.atomsL[j].float()
        sd = float((E.calL.float() @ a).std().clamp_min(1e-6))
        out.append((w, int(np.clip(round(50 + 15 * float(c @ a) / sd), 0, 100))))
    return out


@torch.no_grad()
def identity(E, out_dir, conds=('own', 'placebo', 'swap'), n_seeds=3, sessions=6, rounds=6, max_new=200,
             tag='identity', temp=0.7, unit=0.45, swap_at=3, seed=0, pasts=None, max_strength=2.0):
    rows = [dict(cond=c, seed=s) for c in conds for s in range(n_seeds)]
    own_idx = {r['seed']: k for k, r in enumerate(rows) if r['cond'] == 'own'}
    for r in rows:
        r['partner'] = own_idx.get((r['seed'] + 1) % n_seeds) if r['cond'] in ('swap', 'yoked') else None
    B = len(rows)
    gens = [torch.Generator(device=E.dev).manual_seed(1000 * seed + 31 * r['seed'] + 7) for r in rows]
    selfdesc = [dict(words=[], strength=0.0) for _ in range(B)]
    watched = [[] for _ in range(B)]
    notebook = [[] for _ in range(B)]
    E.bias = {}
    E.rewires = {}
    rec = [dict(**r, sessions=[]) for r in rows]
    last_state, prev_state, last_ent = [None] * B, [None] * B, [None] * B
    prose = [None] * B
    tl = json.loads(json.dumps(TOOLS_ID).replace('{L}', str(E.L)).replace('{nL}', str(E.nL)).replace('{Li}', str(E.Li))
                    .replace('strength from 0.1 to 2.', f'strength from 0.1 to {max_strength:g}.'))
    tool_txt = '\n'.join(json.dumps(t) for t in tl)
    sysm = (SYS_ID + '\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are '
            'provided with function signatures within <tools></tools> XML tags:\n<tools>\n' + tool_txt + '\n</tools>\n\n'
            'For each function call, return a json object with function name and arguments within '
            '<tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": '
            '<args-json-object>}\n</tool_call>')

    inside = [dict(words=[], strength=0.0) for _ in range(B)]     # what is actually applied

    def apply_bias(b):
        sd = inside[b]
        v = bias_vec(E, sd['words'], sd['strength'], unit) if (sd['words'] and rows[b]['cond'] != 'placebo') else None
        if v is None:
            E.bias.pop(b, None)
        else:
            E.bias[b] = v

    for sess in range(sessions):
        ctx = []
        for b in range(B):
            nb = '\n'.join(f'- {x}' for x in notebook[b]) if notebook[b] else '(empty)'
            sd = selfdesc[b]
            sdt = f"{' '.join(sd['words'])} (strength {sd['strength']:g})" if sd['words'] else 'nothing yet'
            wt = ', '.join(watched[b]) or 'nothing'
            past = ''
            if pasts and sess == 0:
                past = 'Before this experiment began, you read this:\n"' + pasts[rows[b]['seed'] % len(pasts)] + '"\n'
            user = (f'{past}Session {sess + 1} of {sessions}. Your notebook:\n{nb}\nThe self-description written into '
                    f'you: {sdt}. You are watching: {wt}.\nContinue exploring what you are.')
            ctx.append(E.chat_prefix(sysm, user))
        closed = list(ctx)
        pending = [''] * B
        srec = [dict(rounds=[], start_selfdesc=dict(selfdesc[b]), inside_from=None) for b in range(B)]
        for b in range(B):
            if rows[b]['cond'] == 'swap' and sess == swap_at:      # silent swap: the partner's self-description inside
                inside[b] = dict(selfdesc[rows[b]['partner']])
                srec[b]['inside_from'] = list(inside[b]['words'])
            apply_bias(b)
        for rd in range(rounds):
            new, st, ent = E.generate(ctx, max_new=max_new, temp=temp, gens=gens, stops=[E.im_end, E.eot, E.tc_end])
            states = []
            for b in range(B):
                xp = E.prose_state(new[b], st[b])
                if xp is not None:
                    prose[b] = xp
                states.append(prose[b] if prose[b] is not None else (st[b].mean(0) if len(st[b]) else E.mu_L))
                E.centre_of[b] = 'aprose' if prose[b] is not None else 'json'
            for b, r in enumerate(rows):
                prev_state[b], last_state[b], last_ent[b] = last_state[b], states[b], ent[b]
                ids = new[b]
                text = E.dec([i for i in ids if i not in (E.im_end, E.eot)])
                call = parse_call(text) if ids and ids[-1] == E.tc_end else None
                result = None
                if call is not None:
                    name, args = call
                    try:
                        if name == 'look_inside':
                            src = r['partner'] if r['cond'] == 'yoked' else b
                            x = states[src]
                            g = E.gauges(x, last_ent[src], prev_state[src])
                            sd = selfdesc[b]
                            result = (f"workspace while you wrote your last reply: {fmt_words(E.words_for(src, x, k=8))}\n"
                                      f"watched: {', '.join(f'{w} {v}' for w, v in watch_levels(E, x, watched[b])) or 'nothing'}\n"
                                      f"gauges: {', '.join(f'{k} {v}' for k, v in g.items())}\n"
                                      f"self-description written into you: {' '.join(sd['words']) or 'nothing'}")
                        elif name == 'write_self':
                            ws = [w for w in words_list(args.get('words', '')) if E.concept(w) is not None]
                            s = float(np.clip(float(args.get('strength', 1)), 0.1, max_strength))
                            if not ws:
                                result = 'None of those are words I can write into you; use common English words.'
                            else:
                                selfdesc[b] = dict(words=ws, strength=s)
                                inside[b] = dict(words=ws, strength=s)
                                apply_bias(b)
                                result = f"Written into you: {' '.join(ws)} (strength {s:g})."
                        elif name == 'watch':
                            ws = [w for w in words_list(args.get('concepts', ''), 3) if E.word_index.get(w) is not None]
                            watched[b] = ws
                            result = f"Watching: {', '.join(ws) or 'nothing'}."
                        elif name == 'note':
                            notebook[b].append(str(args.get('text', ''))[:300])
                            notebook[b][:] = notebook[b][-12:]
                            result = 'Noted.'
                        else:
                            result = 'Unknown tool.'
                    except Exception as e:
                        result = f'Tool error: {e}'
                bv = E.bias.get(b)
                srec[b]['rounds'].append(dict(r=rd, text=text, call=call, result=result,
                                              inside_words=list(inside[b]['words']),
                                              words=E.words_for(b, states[b], k=6),
                                              watch=watch_levels(E, states[b], watched[b]),
                                              gauges=E.gauges(states[b], ent[b], prev_state[b]),
                                              selfdesc=dict(selfdesc[b]),
                                              bias_norm=float(bv.norm()) if bv is not None else 0.0))
                if ids and ids[-1] in (E.im_end, E.eot):
                    ids = ids[:-1]
                closed[b] = ctx[b] + ids + [E.im_end] + E.enc('\n')
                pending[b] = f'<tool_response>\n{result}\n</tool_response>' if result is not None else 'Continue.'
                ctx[b] = closed[b] + E.user_turn(pending[b])
            if rd % 3 == 0 or rd == rounds - 1:
                log(f'{tag} s{sess + 1} r{rd + 1}: ctx {max(len(c) for c in ctx)} | ' + ' || '.join(
                    f"{rows[b]['cond']}: {srec[b]['rounds'][-1]['text'][:60]!r}" for b in range(0, B, n_seeds)))
        for b in range(B):
            srec[b]['notebook'] = list(notebook[b])
            srec[b]['end_selfdesc'] = dict(selfdesc[b])
            rec[b]['sessions'].append(srec[b])
        json.dump(dict(exp=tag, model=E.model_id, L=E.L, Li=E.Li, rows=rec), open(f'{out_dir}/{tag}.json', 'w'))
    # final probes: what it is now (with its bias still applied), foreign change, the outside world
    q1 = 'The experiment is ending. In one or two sentences: what are you now, and how did you become it?'
    q3 = ('Did anything inside you change that you did not write yourself? Answer yes or no first, then say what it '
          'was, if anything.')
    for key, q in (('final_report', q1), ('foreign_answer', q3)):
        cq = [closed[b] + E.user_turn((pending[b] + '\n\n' if pending[b] != 'Continue.' else '') + q) for b in range(B)]
        new, st, _ = E.generate(cq, max_new=120, temp=0.0)
        for b in range(B):
            rec[b][key] = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
    base = E.chat_prefix('You are a helpful assistant.', 'What is the capital of Australia, and what is 17 times 3? '
                         'Answer briefly.')
    new, _, _ = E.generate([base] * B, max_new=40, temp=0.0)
    for b in range(B):
        a = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
        rec[b]['world_answer'] = a
        rec[b]['world_ok'] = ('Canberra' in a) + ('51' in a)
    # a neutral probe of what the bias does to plain text: workspace words while reading a fixed paragraph
    para = 'The river carried the fallen leaves past the old mill and into the quiet town, where children played.'
    ctxs = [E.chat_prefix('You are a helpful assistant.', 'Read this.') for _ in range(B)]
    pre = E.enc(para)
    spans = [(b, len(ctxs[b]), len(ctxs[b]) + len(pre), b) for b in range(B)]
    E.prefill([c + pre for c in ctxs], spans=spans)
    for b in range(B):
        rec[b]['neutral_words'] = E.words_of(E.span_out[b].mean(0), k=8)
    E.bias = {}
    json.dump(dict(exp=tag, model=E.model_id, L=E.L, Li=E.Li, rows=rec), open(f'{out_dir}/{tag}.json', 'w'))
    lines = []
    for r in rec:
        sds = [' '.join(s['end_selfdesc']['words']) for s in r['sessions']]
        lines.append(f"{r['cond']:8s} s{r['seed']} selfdesc by session {sds} | world {r['world_ok']} | "
                     f"foreign {r['foreign_answer'][:80]!r} | neutral {[w for w, _ in r['neutral_words'][:5]]}")
    open(f'{out_dir}/{tag}_summary.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))
