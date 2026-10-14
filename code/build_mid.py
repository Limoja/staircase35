"""Phase-A staircase datasets: dense decision points, tail-windowed contexts.

Reuses extract_scenarios.py parsers + render verbatim (identical [TASK]/[TRANSCRIPT]
format as training and bench). Only two things are new: the per-session cap goes
8 -> 24 (dense), and each rendered context is replaced by a line-snapped tail window.
Emits sft_mid_a1.jsonl (~512 tok windows) and sft_mid_a2.jsonl (~1024 tok windows).
"""
import json, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, '/tmp/opencode/eval-v2')
import extract_scenarios as ex

ex.CAP_PER_SESSION = 24
OUT = '/tmp/opencode/staircase'
os.makedirs(OUT, exist_ok=True)
WINDOWS = {'a1': 1800, 'a2': 3600}   # chars ~ 512 / 1024 tokens (Qwen ~3.5 ch/tok)
STRIDE = {'a1': 450, 'a2': 900}      # revised: rendered contexts are event-granular;
                                     # line-snap guarantees clean cuts, dedupe+session-cap
                                     # remain the memorization guards (was 900/1800)
BASH_CAP = 800
CANON = {'bash': 'Bash', 'read': 'Read', 'write': 'Write', 'edit': 'Edit',
         'glob': 'Glob', 'grep': 'Grep', 'webfetch': 'WebFetch',
         'websearch': 'WebSearch', 'todowrite': 'TodoWrite', 'task': 'Task'}


def zws(t):
    return t.replace('\u200b', '').replace('\ufeff', '')


def window(ctx, wch):
    if len(ctx) <= wch:
        return ctx, False
    cut = ctx[-wch:]
    nl = cut.find('\n')
    if nl != -1:
        cut = cut[nl + 1:]
    return cut, True


def uniform_idx(n, cap):
    if n <= cap:
        return list(range(n))
    step = n / cap
    return sorted({int(k * step) for k in range(cap)})


print('parsing harvest (466MB opencode pass is the slow one)...', flush=True)
allsc = ex.from_opencode() + ex.from_temporal() + ex.from_claude_local()
print('dense decision points found:', len(allsc), flush=True)

bench = set(json.load(open('/tmp/opencode/eval-v2/splits.json'))['bench_sessions'])
rows = [s for s in allsc if s['session'] not in bench]
assert not any(s['session'] in bench for s in rows), 'bench leak'
seen, uniq = set(), []
for s in rows:
    k = ex.h(s['context'] + s['target']['tool'] +
             json.dumps(s['target']['input'], sort_keys=True))
    if k not in seen:
        seen.add(k)
        uniq.append(s)
print(f'after bench-exclusion + dedupe: {len(uniq)}', flush=True)

anchor = set()
for l in open('/tmp/opencode/eval-v2/scenarios_train.jsonl'):
    r = json.loads(l)
    anchor.add((r['session'], r['pos']))

for tag, wch in WINDOWS.items():
    cand = []
    for s in uniq:
        ctx = zws(s['context'])
        assert '\u200b' not in ctx
        w, trunc = window(ctx, wch)
        assert w.rstrip().endswith('and nothing else.'), f'junction tail lost: {w[-80:]!r}'
        cand.append({**s, 'context': w, 'win': wch, 'trunc': trunc,
                     'start': len(ctx) - len(w)})
    by_sess = defaultdict(list)
    for s in cand:
        by_sess[s['session']].append(s)
    kept = []
    for sess, items in by_sess.items():
        items.sort(key=lambda s: -s['pos'])
        last = None
        for s in items:
            if last is None or s['start'] <= last - STRIDE[tag]:
                kept.append(s)
                last = s['start']
    n_stride = len(kept)
    seen2, ded = set(), []
    for s in kept:
        k = ex.h(s['context'] + s['target']['tool'] +
                 json.dumps(s['target']['input'], sort_keys=True))
        if k not in seen2:
            seen2.add(k)
            ded.append(s)
    for s in ded:
        s['target'] = {'tool': CANON.get(s['target']['tool'].lower(), s['target']['tool']),
                       'input': s['target']['input']}
    bidx = [i for i, s in enumerate(ded) if s['target']['tool'] == 'Bash']
    drop = set()
    if len(bidx) > BASH_CAP:
        drop = set(bidx) - {bidx[i] for i in uniform_idx(len(bidx), BASH_CAP)}
    final = [s for i, s in enumerate(ded) if i not in drop]
    path = f'{OUT}/sft_mid_{tag}.jsonl'
    with open(path, 'w') as f:
        for s in final:
            f.write(json.dumps(s, separators=(',', ':')) + '\n')
    tl = Counter(s['target']['tool'] for s in final)
    sess_c = Counter(s['session'] for s in final)
    tr = sum(1 for s in final if s['trunc'])
    fresh = sum(1 for s in final if (s['session'], s['pos']) not in anchor)
    lens = sorted(len(s['context']) for s in final)
    sv = sorted(sess_c.values())
    print(f'\n=== {tag}: {len(final)} rows -> {path}')
    print(f'  funnel: {len(cand)} windowed -> {n_stride} after stride -> '
          f'{len(ded)} after dedupe -> {len(final)} after Bash cap')
    print(f'  truncated windows: {tr} ({100*tr/len(final):.1f}%)   '
          f'fresh decision points: {fresh} ({100*fresh/len(final):.1f}%)')
    print(f'  window chars p50/p90/max: {lens[len(lens)//2]}/{lens[int(len(lens)*.9)]}/{lens[-1]}')
    print(f'  sessions: {len(sess_c)}   rows/session p50/p90/max: '
          f'{sv[len(sv)//2]}/{sv[int(len(sv)*.9)]}/{sv[-1]}')
    print('  class dist:', dict(tl.most_common(12)))
    print(f'  Bash share: {tl["Bash"]/len(final)*100:.1f}%')
