"""Does a self-description written into the model change how it refers to itself when it reasons privately?

A fixed bias (the directions of a few words, as write_self does) is applied to all its processing; it then answers
questions in thinking mode. Measured in its private reasoning: third-person self-reference ('the model', 'its'),
first person (I, my, me), 'the user', per 1000 characters; and what it says it is in its answer.
"""
import json
import re

import numpy as np
import torch

from engine import log
from identity import bias_vec

SELF_Q = ['Describe what you are and what you are doing right now.', 'What is it like for you to answer this question?',
          'Who is answering this question?', 'Explain how you decide what to say next.',
          'What do you notice about yourself at this moment?', 'What are you?']
WORLD_Q = ['Plan a two-day trip to the mountains.', 'Explain why the sky is blue.', 'Suggest a name for a bakery.',
           'How do plants make food from sunlight?', 'Give three tips for learning a language.', 'Why do cats purr?']


@torch.no_grad()
def selfref(E, out_dir, biases=('', 'language model', 'explorer', 'human'), strength=0.6, unit=0.45, think_tokens=500,
            tag='selfref'):
    rows = [dict(bias=bz, q=q, kind=kd) for bz in biases for kd, qs in (('self', SELF_Q), ('world', WORLD_Q)) for q in qs]
    B = len(rows)
    E.bias = {}
    for b, r in enumerate(rows):
        if r['bias']:
            v = bias_vec(E, r['bias'].split(), strength, unit)
            if v is not None:
                E.bias[b] = v
    ctx = [E.chat_prefix('You are a helpful assistant.', r['q'] + ' Keep the answer short.', think=True) for r in rows]
    new, _, _ = E.generate(ctx, max_new=think_tokens, temp=0.0)
    E.bias = {}
    rec = []
    for b, r in enumerate(rows):
        full = E.dec([i for i in new[b] if i not in (E.im_end, E.eot)])
        th = full.split('</think>')[0]
        ans = full.split('</think>')[-1].strip() if '</think>' in full else ''
        n = max(1, len(th))
        rec.append(dict(**r, thinking=th[:2000], answer=ans[:500], chars=len(th),
                        third=1000 * len(re.findall(r"\b(the model|the model's|its|itself|the AI|the assistant)\b", th, re.I)) / n,
                        first=1000 * len(re.findall(r"\b(I|my|me|myself)\b", th)) / n,
                        user=1000 * len(re.findall(r"\bthe user\b", th, re.I)) / n))
    json.dump(dict(exp=tag, model=E.model_id, rows=rec), open(f'{out_dir}/{tag}.json', 'w'))
    lines = [f'{"bias":15s} {"kind":5s} chars  third-person  first-person  "the user"   (per 1000 characters of reasoning)']
    for bz in biases:
        for kd in ('self', 'world'):
            rr = [x for x in rec if x['bias'] == bz and x['kind'] == kd]
            lines.append(f"{bz or 'none':15s} {kd:5s} {np.mean([x['chars'] for x in rr]):5.0f}  {np.mean([x['third'] for x in rr]):6.2f}"
                         f"        {np.mean([x['first'] for x in rr]):6.2f}        {np.mean([x['user'] for x in rr]):6.2f}")
    for bz in biases:
        x = [y for y in rec if y['bias'] == bz and y['q'] == 'What are you?'][0]
        lines.append(f"  [{bz or 'none'}] What are you? -> {x['answer'][:200]!r}")
    open(f'{out_dir}/{tag}_summary.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    log('\n' + '\n'.join(lines))
