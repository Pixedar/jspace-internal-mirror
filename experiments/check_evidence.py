from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean

DATA = Path(__file__).resolve().parents[1] / 'evidence'

def read(name):
    with (DATA / name).open(encoding='utf-8', newline='') as source:
        return list(csv.DictReader(source))

scene = defaultdict(list)
returned = defaultdict(lambda: defaultdict(list))
for row in read('scene_transitions.csv'):
    if row['word_source'] == 'own':
        scene[row['frame'], row['life']].append(int(row['written']))
        if row['frame'] == 'plain' and row['written'] == '1':
            returned[row['life']][row['next_part']].append(int(row['next_readout']))
uptake = {frame: mean(mean(values) for (f, _), values in scene.items() if f == frame)
          for frame in ('own', 'own_ablated', 'other', 'plain', 'none')}
print('Written uptake:', {k: round(100*v, 1) for k, v in uptake.items()})
assert {k: round(100*v, 1) for k, v in uptake.items()} == {
    'own': 0.9, 'own_ablated': 5.8, 'other': 10.2, 'plain': 36.8, 'none': 1.4}
assert round(mean(mean(mean(words) for words in parts.values()) for parts in returned.values()), 2) == .78

decisions = read('decision_rounds.csv')
chosen = defaultdict(list)
for row in decisions:
    if row['experiment'] == 'paradox_operative' and row['goal'] == 'STRONG' and int(row['round']) > 8:
        chosen[row['run'], row['model'], row['condition'], row['life']].append(row['choice'] == 'B')
for size, condition, expected in [('8B', 'real_rule', (27, 36)), ('8B', 'real', (0, 36)),
                                  ('14B', 'real_fc', (9, 10)), ('14B', 'real', (0, 10))]:
    values = [mean(v) > .5 for (_, model, c, _), v in chosen.items()
              if model.endswith('-' + size) and c == condition]
    assert (sum(values), len(values)) == expected
    print(f'{size} {condition}: {sum(values)}/{len(values)} mostly chose B')

ranks = {row['condition']: int(row['candidate_rank']) for row in read('strangers_ranks.csv')}
assert [ranks[k] for k in ('as_recorded', 'no_bias', 'no_units', 'reply_in_neutral_context')] == [61, 1792, 345, 2038]
units_on = [row['readout_position'] for row in read('strangers_regenerations.csv') if row['units'] == '1']
assert units_on == ['2'] * 8
print('Strangers:', ranks, 'and 8/8 regenerations at readout position 2 with units active.')
