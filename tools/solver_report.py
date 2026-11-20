"""Summarize completed self-play records: outcomes, Wilson binomial CI, opening strategies,
and recorded search-only wall times. Usage: python tools/solver_report.py BEFORE_DIR AFTER_DIR.
The timing excludes engine.step and is contaminated by concurrent machine load.
"""
import gzip, json, glob, sys, math, collections, statistics
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h
for d in sys.argv[1:]:
    out = collections.Counter(); t0 = collections.Counter(); ms = []; turns = []
    pick0 = collections.Counter()
    for f in sorted(glob.glob(d + '/g*.jsonl.gz')):
        lines = [json.loads(l) for l in gzip.open(f, 'rt')]
        res = lines[-1]['result']; out[res['outcome']] += 1; turns.append(res['turns'])
        dec = [l for l in lines[1:-1] if l.get('strategies')]
        if dec:
            s = dec[0]['strategies']; c = dec[0]['candidates']
            key = tuple(tuple(sorted((c[k][i]['action'] + (json.dumps(c[k][i]['data']) if c[k][i]['data'] else ''), round(p, 2)) for i, p in enumerate(s[k]) if p > 0.01)) for k in (0, 1))
            t0[key] += 1
            pick0[tuple(c[k][dec[0]['picks'][k]]['action'] for k in (0, 1))] += 1
        ms += [l['ms'] for l in dec]
    n = out['p1'] + out['p2']
    lo, hi = wilson(out['p1'], n)
    print(f'== {d}: {sum(out.values())} games {dict(out)}; p1 share of decisive {out["p1"]/n:.3f} 95% CI [{lo:.3f}, {hi:.3f}]')
    print(f'   turns/game mean {statistics.mean(turns):.1f}; searched-turn ms mean {statistics.mean(ms):.0f} median {statistics.median(ms):.0f} p90 {sorted(ms)[int(.9*len(ms))]}')
    print(f'   distinct turn-0 strategy pairs: {len(t0)}; distinct turn-0 played action pairs: {len(pick0)}')
    for k, v in t0.most_common(4): print('   ', v, 'p1', k[0], '| p2', k[1])
    rounds = []; fb = 0; prev = []
    for f in sorted(glob.glob(d + '/g*.jsonl.gz')):
        for l in gzip.open(f, 'rt'):
            x = json.loads(l)
            if x.get('oracle'): rounds.append(x['oracle']['rounds']); fb += x['oracle'].get('fallback', False); prev.append(x.get('previews', 0))
    print(f'   oracle rounds mean {statistics.mean(rounds):.2f}; previews/turn mean {statistics.mean(prev):.1f}; full-table fallbacks {fb}')
