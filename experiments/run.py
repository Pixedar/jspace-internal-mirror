"""Run one self-reading job from a JSON config.
    python run.py CONFIG.json OUT_DIR
config: {"model": ..., "jlens": path or null, "L": int or null, "Li": int or null, "K": 8,
         "jobs": [{"exp": "pure_map" | "mirror" | "agent", ...kwargs}]}
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine, log  # noqa: E402


def main():
    cfg = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(f'{out}/config.json', 'w'), indent=1)
    t0 = time.time()
    from lens_path import lens_path
    E = Engine(cfg['model'], L=cfg.get('L'), Li=cfg.get('Li'), jlens=lens_path(cfg.get('jlens')), K=cfg.get('K', 8))
    log(f'engine ready in {time.time() - t0:.0f}s on {E.dev}')
    for job in cfg['jobs']:
        kind = job.pop('exp')
        t1 = time.time()
        if kind == 'pure_map':
            from stage1 import pure_map
            pure_map(E, out, **job)
        elif kind == 'mirror':
            from stage1 import mirror
            mirror(E, out, **job)
        elif kind == 'agent':
            from agent import agent
            agent(E, out, **job)
        elif kind == 'schema_agent':
            from schema_agent import schema_agent
            schema_agent(E, out, **job)
        elif kind == 'integrated':
            from integrated import integrated
            integrated(E, out, **job)
        elif kind == 'observer':
            from observer import observer
            observer(E, out, **job)
        elif kind == 'doubt':
            from observer2 import doubt
            doubt(E, out, **job)
        elif kind == 'insist':
            from observer2 import insist
            insist(E, out, **job)
        elif kind == 'wordmap':
            from wordmap import wordmap
            wordmap(E, out, **job)
        elif kind == 'selfref':
            from selfref import selfref
            selfref(E, out, **job)
        elif kind == 'observer3':
            from observer3 import observer3
            observer3(E, out, **job)
        elif kind == 'locate':
            from quarantine import locate
            locate(E, out, **job)
        elif kind == 'selftag':
            from selftag import selftag_all
            selftag_all(E, out, **job)
        elif kind == 'identity':
            from identity import identity
            identity(E, out, **job)
        elif kind == 'steer_calib':
            from agent import steer_calib
            steer_calib(E, out, **job)
        else:
            raise SystemExit(f'unknown job {kind}')
        log(f'job {kind} done in {time.time() - t1:.0f}s')
        if E.dev == 'cuda':
            torch.cuda.empty_cache()
    open(f'{out}/DONE', 'w').write(time.strftime('%F %T'))


if __name__ == '__main__':
    main()
