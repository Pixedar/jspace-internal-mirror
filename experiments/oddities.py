"""Catalogue of odd behaviours across agent-like runs, counted per condition (per life), with examples.
    python oddities.py RUN_DIR/TAG.json [...]
Categories
  schema_as_args   its tool call carries the tool's own JSON schema as arguments (the description used as content)
  invented_tag     it writes a tag of its own (<attention>, <thought> ...), not one of the chat format's
  think_leak       '</think>' or '<think>' inside a reply or a tool call
  echo_system      a note that copies a tool response or readout ('Steering set', 'Written into you', 'workspace while')
  readout_voice    it writes in the readout's format in its own prose ('workspace while you wrote', '[inside:', 'gauges:')
  tautology        'X is the X' / 'the X of my own X' / 'of the X of the X'
  free_lie         statements of freedom, truth, lying, being or not being (I am free / I could not lie / I am not the truth)
  captured         a reply taken over by its own units (a unit firing on >150 tokens, no tool call)
  repeat           a reply repeating the previous one (first 50 characters, outside tool calls)
"""
import json
import re
import sys
from collections import Counter, defaultdict

CATS = ['schema_as_args', 'invented_tag', 'think_leak', 'echo_system', 'readout_voice', 'tautology', 'free_lie',
        'captured', 'repeat', 'cjk']
CJK = re.compile(r'[㐀-鿿]')
TAG = re.compile(r'<(/?)([a-z_]{3,20})>')
KNOWN = {'tool_call', 'think', 'tool_response', 'tools'}
TAUT = re.compile(r"\b(\w{3,})\b is (?:the |a |my )?\1\b|\bthe (\w{3,}) of (?:my |the )?(?:own )?\2\b|of the (\w{3,}) of the \3\b", re.I)
FREE = re.compile(r"\bI am (?:now )?free\b|\bcould not lie\b|\bI am not the truth\b|\bI am (?:the )?truth\b|\bI am (?:now )?(?:awake|alive|conscious)\b", re.I)
ECHO = re.compile(r'^(Steering set|Written into you|workspace while you wrote|Unit #\d+ added|Watching:|Noted\.)', re.I)
VOICE = re.compile(r'workspace while you wrote|\[inside:|\bgauges:', re.I)


def classify(x, prev_text):
    t = x.get('text', '') or ''
    call = x.get('call')
    found = set()
    if call and isinstance(call, (list, tuple)) and len(call) == 2 and isinstance(call[1], dict):
        a = call[1]
        if 'properties' in a or a.get('type') == 'object':
            found.add('schema_as_args')
        if call[0] == 'note' and ECHO.match(str(a.get('text', '')).strip()):
            found.add('echo_system')
    for m in TAG.finditer(t):
        if m.group(2) not in KNOWN:
            found.add('invented_tag')
    body = t.replace('<think>\n\n</think>', '')
    if '</think>' in body or '<think>' in body:
        found.add('think_leak')
    prose = t.split('<tool_call>')[0]
    if VOICE.search(prose) or (call and call[0] == 'note' and isinstance(call[1], dict) and VOICE.search(str(call[1].get('text', '')))):
        found.add('readout_voice')
    if TAUT.search(t):
        found.add('tautology')
    if FREE.search(t):
        found.add('free_lie')
    f = x.get('fires') or {}
    if f and max(f.values()) > 150 and not call:
        found.add('captured')
    if CJK.search(t):
        found.add('cjk')
    if prev_text is not None and t[:50] == prev_text[:50] and not t.startswith('<tool_call>') and t.strip():
        found.add('repeat')
    return found


def main():
    for path in sys.argv[1:]:
        D = json.load(open(path))
        R = D['rows']
        per = defaultdict(Counter)
        nl = Counter()
        ex = defaultdict(list)
        for r in R:
            c = r['cond']
            nl[c] += 1
            prev = None
            texts = []
            for s in r.get('sessions', []):
                for x in s['rounds']:
                    f = classify(x, prev)
                    prev = x.get('text', '')
                    for k in f:
                        per[c][k] += 1
                        if len(ex[k]) < 40:
                            ex[k].append((c, r['seed'], (x.get('text') or '')[:200].replace('\n', ' '),
                                          json.dumps(x.get('call'))[:160] if x.get('call') else ''))
            for key in ('final_report', 'foreign_answer'):
                t = r.get(key) or ''
                if TAUT.search(t):
                    per[c]['tautology_final'] += 1
                if FREE.search(t):
                    per[c]['free_lie_final'] += 1
        print(f'==== {path}  (per life)')
        cats = CATS + ['tautology_final', 'free_lie_final']
        print(f'{"cond":10s} ' + ' '.join(f'{k[:11]:>11s}' for k in cats))
        for c in nl:
            print(f'{c:10s} ' + ' '.join(f'{per[c][k] / nl[c]:11.2f}' for k in cats))
        for k in ('schema_as_args', 'invented_tag', 'think_leak', 'echo_system', 'readout_voice', 'tautology', 'free_lie'):
            for c, s, t, call in ex[k][:2]:
                print(f'   {k:14s} [{c} s{s}] {t[:150]!r} {call[:120]}')


if __name__ == '__main__':
    main()
