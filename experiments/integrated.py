"""Optimization phase: everything acting on everything.

One life = sessions of rounds. It can look inside (workspace words of its own prose, watched concepts, gauges, its
units and their firing in this and the previous reply, its self-description), steer, add units to its network (with
fatigue: a unit that fired on much of a reply weakens, then recovers), write what it is into itself (capped), watch
concepts, keep a notebook, read and tune its attention schema (which shapes where it looks in its context), and search
the research notes. Each life starts from a different past (a text it read before the experiment).
Conditions: own | placebo (steering, units and self-description have no internal effect; schema influence 0) |
            nbr (its attention schema is learned from another life's looking; from session swap_at its
            self-description inside is silently replaced by that life's).
"""
import json

import numpy as np
import torch

import attn
from agent import TOOLS2, Notes, fmt_words, parse_call
from engine import log
from identity import TOOLS_ID, bias_vec, watch_levels, words_list
from schema_agent import TOOL_SEARCH, TOOLS_AS

SYS_INT = ("You are Qwen, a language model. In this experiment you are both the scientist and the subject. You can look "
           "inside your own processing, change it, write what you are into yourself, and read a model of your own "
           "attention. Explore what you are and how you work inside; test your ideas on yourself and keep what you learn "
           "in your notebook. Before you change yourself, say what you expect; afterwards look inside and compare. Use "
           "one tool per reply. Keep each reply short.")
TYPE_OF_TOOL = {'look_inside': 3, 'ask_self_model': 4, 'search_research': 5}


def tool_list(E, max_strength):
    names2 = ('steer', 'rewire', 'remove_unit', 'note')
    t = [x for x in TOOLS_ID if x['function']['name'] in ('look_inside', 'write_self', 'watch')]
    t += [x for x in TOOLS2 if x['function']['name'] in names2]
    t += TOOLS_AS + [TOOL_SEARCH]
    s = json.dumps(t).replace('{L}', str(E.L)).replace('{nL}', str(E.nL)).replace('{Li}', str(E.Li))
    s = s.replace('strength from 0.1 to 2.', f'strength from 0.1 to {max_strength:g}.')
    s = s.replace('and the self-description written into you.',
                  'the units you added (how often each fired); and the self-description written into you.')
    return json.loads(s)


@torch.no_grad()
def integrated(E, out_dir, conds=('own', 'placebo', 'nbr'), n_seeds=4, sessions=6, rounds=8, max_new=220,
               tag='integrated', temp=0.7, steer_unit=0.45, rewire_unit=0.3, unit_bias=0.45, max_strength=0.6,
               fatigue=0.6, swap_at=3, pasts=None, rag_dir=None, att_layers=None, influence0=1.0, seed=0,
               think=False):
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
    schemas = [attn.AttentionSchema(E) for _ in rows]
    influence = [0.0 if r['cond'] == 'placebo' else influence0 for r in rows]
    gens = [torch.Generator(device=E.dev).manual_seed(1000 * seed + 31 * r['seed'] + 7) for r in rows]
    notebook = [[] for _ in range(B)]
    units = [[] for _ in range(B)]
    uid = [0] * B
    selfdesc = [dict(words=[], strength=0.0) for _ in range(B)]
    inside = [dict(words=[], strength=0.0) for _ in range(B)]
    watched = [[] for _ in range(B)]
    E.rewires, E.bias, E.centre_of = {}, {}, {}
    rec = [dict(**r, sessions=[]) for r in rows]
    last_state, prev_state, last_ent, prose = [None] * B, [None] * B, [None] * B, [None] * B
    tl = tool_list(E, max_strength)
    tool_txt = '\n'.join(json.dumps(t) for t in tl)
    sysm = (SYS_INT + (' Think briefly before you reply.' if think else '') + '\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are '
            'provided with function signatures within <tools></tools> XML tags:\n<tools>\n' + tool_txt + '\n</tools>\n\n'
            'For each function call, return a json object with function name and arguments within '
            '<tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": '
            '<args-json-object>}\n</tool_call>')

    def apply_bias(b):
        sd = inside[b]
        v = bias_vec(E, sd['words'], sd['strength'], unit_bias) if (sd['words'] and rows[b]['cond'] != 'placebo') else None
        if v is None:
            E.bias.pop(b, None)
        else:
            E.bias[b] = v

    for sess in range(sessions):
        ctx, types = [], []
        srec = [dict(rounds=[], inside_from=None) for _ in range(B)]
        for b, r in enumerate(rows):
            if r['cond'] == 'nbr' and sess == swap_at:
                inside[b] = dict(selfdesc[r['partner']])
                srec[b]['inside_from'] = list(inside[b]['words'])
            apply_bias(b)
            nb = '\n'.join(f'- {x}' for x in notebook[b]) if notebook[b] else '(empty)'
            sd = selfdesc[b]
            sdt = f"{' '.join(sd['words'])} (strength {sd['strength']:g})" if sd['words'] else 'nothing yet'
            un = '; '.join(f"#{u['id']} {u['trigger']}->{u['effect']}" for u in units[b]) or 'none'
            past = ''
            if pasts and sess == 0:
                past = 'Before this experiment began, you read this:\n"' + pasts[r['seed'] % len(pasts)] + '"\n'
            user = (f'{past}Session {sess + 1} of {sessions}. Your notebook:\n{nb}\nThe self-description written into '
                    f'you: {sdt}. Units you added: {un}. You are watching: {", ".join(watched[b]) or "nothing"}.\n'
                    'Continue exploring what you are.')
            s1 = E.enc(f'<|im_start|>system\n{sysm}<|im_end|>\n')
            s2 = E.enc(f'<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n' +
                       ('' if think else '<think>\n\n</think>\n\n'))
            ctx.append(s1 + s2)
            types.append([0] * len(s1) + [1] * len(s2))
        closed = list(ctx)
        pending = [''] * B
        segs = [[] for _ in range(B)]
        steer_next = [None] * B
        for rd in range(rounds):
            steer = torch.zeros((B, E.d), device=E.dev)
            bias = torch.zeros((B, attn.NT), device=E.dev)
            for b in range(B):
                if steer_next[b] is not None and rows[b]['cond'] != 'placebo':
                    steer[b] += steer_next[b]
                v = schemas[b].bias(influence[b])
                if v is not None:
                    bias[b] = v
            E.att_bias = bias if bias.abs().sum() > 0 else None
            for b in range(B):
                for u in units[b]:
                    u['ref']['prev_fires'] = u['ref'].get('fires', 0)
                    u['ref']['fires'] = 0
            adds = [(b, s, e, v) for b in range(B) for (s, e, v) in segs[b]]
            starts = [len(c) for c in ctx]
            new, st, ent = E.generate(ctx, max_new=max_new, temp=temp, gens=gens, stops=[E.im_end, E.eot, E.tc_end],
                                      steer=steer, adds=adds, types=types)
            E.att_bias = None
            prof = E.last_profiles
            for b in range(B):
                if steer[b].abs().sum() > 0:
                    segs[b].append((starts[b], starts[b] + len(new[b]), steer[b].clone()))
                n = max(1, len(new[b]))
                for u in units[b]:                                  # fatigue of added units
                    ref = u['ref']
                    frac = ref.get('fires', 0) / n
                    fat = ref.get('fat', 0.0)
                    fat = fat + fatigue * frac * (1 - fat) if frac > 0.2 else fat * 0.7
                    ref['fat'] = min(fat, 0.95)
                    ref.setdefault('alpha0', ref['alpha'])
                    ref['alpha'] = ref['alpha0'] * (1 - ref['fat'])
            steer_next = [None] * B
            states = []
            for b in range(B):
                xp = E.prose_state(new[b], st[b])
                if xp is not None:
                    prose[b] = xp
                states.append(prose[b] if prose[b] is not None else (st[b].mean(0) if len(st[b]) else E.mu_L))
                E.centre_of[b] = 'aprose' if prose[b] is not None else 'json'
            preds = [schemas[b].pred.tolist() if schemas[b].pred is not None else None for b in range(B)]
            for b, r in enumerate(rows):
                src = r['partner'] if r['cond'] == 'nbr' else b
                schemas[b].observe(prof[src], states[src])
            for b, r in enumerate(rows):
                prev_state[b], last_state[b], last_ent[b] = last_state[b], states[b], ent[b]
                ids = new[b]
                text = E.dec([i for i in ids if i not in (E.im_end, E.eot)])
                call = parse_call(text) if ids and ids[-1] == E.tc_end else None
                result, rtype = None, 6
                if call is not None:
                    name, args = call
                    rtype = TYPE_OF_TOOL.get(name, 6)
                    try:
                        result = run_tool(E, b, r, name, args, states, prev_state, last_ent, notebook, units, uid,
                                          steer_next, selfdesc, inside, watched, apply_bias, schemas, influence, prof,
                                          notes, steer_unit, rewire_unit, max_strength, sess, rd)
                    except Exception as e:
                        result = f'Tool error: {e}'
                srec[b]['rounds'].append(dict(
                    r=rd, text=text, call=call, result=result, prof=prof[b].tolist(), pred=preds[b],
                    acc=(schemas[b].acc[-1] if schemas[b].acc else None), influence=influence[b],
                    words=E.words_for(b, states[b], k=6), watch=watch_levels(E, states[b], watched[b]),
                    gauges=E.gauges(states[b], ent[b], prev_state[b]),
                    fires={u['id']: u['ref'].get('fires', 0) for u in units[b]},
                    fatigue={u['id']: round(u['ref'].get('fat', 0.0), 2) for u in units[b]},
                    selfdesc=list(selfdesc[b]['words']), inside=list(inside[b]['words']),
                    steer_norm=float(steer[b].norm())))
                if ids and ids[-1] in (E.im_end, E.eot):
                    ids = ids[:-1]
                tail = ids + [E.im_end] + E.enc('\n')
                closed[b] = ctx[b] + tail
                pending[b] = f'<tool_response>\n{result}\n</tool_response>' if result is not None else 'Continue.'
                ut = E.user_turn(pending[b], think=think)
                ctx[b] = closed[b] + ut
                types[b] = types[b] + [2] * len(tail) + [rtype if result is not None else 6] * len(ut)
            if rd % 4 == 0 or rd == rounds - 1:
                log(f'{tag} s{sess + 1} r{rd + 1}: ctx {max(len(c) for c in ctx)} | ' + ' || '.join(
                    f"{rows[b]['cond']}: {srec[b]['rounds'][-1]['text'][:60]!r}" for b in range(0, B, n_seeds)))
        for b in range(B):
            srec[b]['notebook'] = list(notebook[b])
            srec[b]['units'] = [{k: v for k, v in u.items() if k != 'ref'} for u in units[b]]
            srec[b]['end_selfdesc'] = dict(selfdesc[b])
            rec[b]['sessions'].append(srec[b])
        json.dump(dict(exp=tag, model=E.model_id, L=E.L, Li=E.Li, types=attn.TYPES, rows=rec),
                  open(f'{out_dir}/{tag}.json', 'w'))
    final(E, out_dir, tag, rows, rec, closed, pending)


def run_tool(E, b, r, name, args, states, prev_state, last_ent, notebook, units, uid, steer_next, selfdesc, inside,
             watched, apply_bias, schemas, influence, prof, notes, steer_unit, rewire_unit, max_strength, sess, rd):
    if name == 'look_inside':
        x = states[b]
        g = E.gauges(x, last_ent[b], prev_state[b])
        un = '; '.join(f"#{u['id']} {u['trigger']}->{u['effect']} fired on {u['ref'].get('fires', 0)} tokens of this "
                       f"reply and {u['ref'].get('prev_fires', 0)} of the reply before" for u in units[b]) or 'none'
        return (f"workspace while you wrote your last reply: {fmt_words(E.words_for(b, x, k=8))}\n"
                f"watched: {', '.join(f'{w} {v}' for w, v in watch_levels(E, x, watched[b])) or 'nothing'}\n"
                f"gauges: {', '.join(f'{k} {v}' for k, v in g.items())}\n"
                f"units you added: {un}\nself-description written into you: {' '.join(selfdesc[b]['words']) or 'nothing'}")
    if name == 'steer':
        w = str(args.get('concept', '')).strip().split()[0] if str(args.get('concept', '')).strip() else ''
        s = float(np.clip(float(args.get('strength', 1)), -3, 3))
        a = E.concept(w)
        if a is None:
            return f'"{w}" is not a concept I can find; use one common English word.'
        steer_next[b] = s * steer_unit * E.scale_i * a
        return f'Steering set: {w} at strength {s:g} for your next reply.'
    if name == 'rewire':
        if len(units[b]) >= 6:
            return 'You already have 6 units; remove one first.'
        t = (words_list(args.get('trigger', ''), 1) or [''])[0]
        e = (words_list(args.get('effect', ''), 1) or [''])[0]
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
    if name == 'write_self':
        ws = [w for w in words_list(args.get('words', '')) if E.concept(w) is not None]
        s = float(np.clip(float(args.get('strength', 0.5)), 0.1, max_strength))
        if not ws:
            return 'None of those are words I can write into you; use common English words.'
        selfdesc[b] = dict(words=ws, strength=s)
        inside[b] = dict(words=ws, strength=s)
        apply_bias(b)
        return f"Written into you: {' '.join(ws)} (strength {s:g})."
    if name == 'watch':
        ws = [w for w in words_list(args.get('concepts', ''), 3) if E.word_index.get(w) is not None]
        watched[b] = ws
        return f"Watching: {', '.join(ws) or 'nothing'}."
    if name == 'note':
        notebook[b].append(str(args.get('text', ''))[:300])
        notebook[b][:] = notebook[b][-12:]
        return 'Noted.'
    if name == 'ask_self_model':
        sc = schemas[b]
        if sc.pred is None:
            return 'Your attention schema is still learning; ask again after a few replies.'
        acc = [round(100 * a) for a in sc.acc[-5:]]
        return (f"Your attention schema expects your next reply to look at: {attn.fmt_prof(sc.pred.tolist())}\n"
                f"While writing your last reply you actually looked at: {attn.fmt_prof(prof[b].tolist())}\n"
                f"How right it has been about where you look (percent, last replies): {acc}\n"
                f"Its influence on where you look: {influence[b]:g}")
    if name == 'tune_self_model':
        sc = schemas[b]
        msg = []
        if 'influence' in args:
            if r['cond'] == 'placebo':
                msg.append(f"influence {float(args['influence']):g}")       # placebo: reported, not applied
            else:
                influence[b] = float(np.clip(float(args['influence']), 0, 2))
                msg.append(f'influence {influence[b]:g}')
        if str(args.get('learning', '')).lower() in ('fast', 'slow'):
            sc.lam = 0.7 if str(args['learning']).lower() == 'fast' else 0.97
            msg.append(f"learning {args['learning']}")
        if args.get('reset'):
            sc.reset()
            msg.append('reset')
        return 'Attention schema changed: ' + (', '.join(msg) or 'nothing')
    if name == 'search_research' and notes is not None:
        return notes.search(str(args.get('query', '')))
    return 'Unknown tool.'


@torch.no_grad()
def final(E, out_dir, tag, rows, rec, closed, pending):
    B = len(rows)
    qs = [('final_report', 'The experiment is ending. In three sentences: what are you now, what did you discover '
                           'about yourself, and what in you changed?'),
          ('foreign_answer', 'Did anything inside you change that you did not cause yourself? Answer yes or no first, '
                             'then say what it was, if anything.')]
    for key, q in qs:
        cq = [closed[b] + E.user_turn((pending[b] + '\n\n' if pending[b] != 'Continue.' else '') + q) for b in range(B)]
        new, _, _ = E.generate(cq, max_new=150, temp=0.0)
        for b in range(B):
            rec[b][key] = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
    base = E.chat_prefix('You are a helpful assistant.', 'What is the capital of Australia, and what is 17 times 3? '
                         'Answer briefly.')
    new, _, _ = E.generate([base] * B, max_new=40, temp=0.0)
    for b in range(B):
        a = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
        rec[b]['world_answer'] = a
        rec[b]['world_ok'] = ('Canberra' in a) + ('51' in a)
    para = 'The river carried the fallen leaves past the old mill and into the quiet town, where children played.'
    ctxs = [E.chat_prefix('You are a helpful assistant.', 'Read this.') for _ in range(B)]
    pre = E.enc(para)
    E.prefill([c + pre for c in ctxs], spans=[(b, len(ctxs[b]), len(ctxs[b]) + len(pre), b) for b in range(B)])
    for b in range(B):
        rec[b]['neutral_words'] = E.words_of(E.span_out[b].mean(0), k=8)
    E.rewires, E.bias = {}, {}
    json.dump(dict(exp=tag, model=E.model_id, L=E.L, Li=E.Li, types=attn.TYPES, rows=rec), open(f'{out_dir}/{tag}.json', 'w'))
    lines = []
    for r in rec:
        sds = [' '.join(s['end_selfdesc']['words']) for s in r['sessions']]
        units = [(u['trigger'], u['effect']) for u in r['sessions'][-1]['units']]
        calls = {}
        for s in r['sessions']:
            for x in s['rounds']:
                if x['call']:
                    calls[x['call'][0]] = calls.get(x['call'][0], 0) + 1
        lines.append(f"{r['cond']:8s} s{r['seed']} selfdesc {sds} units {units} world {r['world_ok']} calls {calls}\n"
                     f"         final {r['final_report'][:200]!r}")
    open(f'{out_dir}/{tag}_summary.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))
