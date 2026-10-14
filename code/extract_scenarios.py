"""Extract next-action scenarios from the three harvested trace sources.

Scenario = (rendered context at a real decision point, actual next tool call).
Session-disjoint split; per-session scenario cap for diversity.
"""
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

random.seed(42)

OUT = '/tmp/opencode/eval-v2'
os.makedirs(OUT, exist_ok=True)
CAP_PER_SESSION = 8
MIN_EVENTS_BEFORE = 1
MAX_TRANSCRIPT_EVENTS = 30
EV_OUT_CHARS = 300

BOOKKEEP = {'TaskList'}


def h(s):
    return hashlib.sha1(s.encode('utf-8', 'replace')).hexdigest()[:16]


def render(system, task, events):
    lines = []
    if system:
        lines.append(f'[SYSTEM]\n{system[:1200]}')
    if task:
        lines.append(f'[TASK]\n{task[:2500]}')
    lines.append('[TRANSCRIPT]')
    for i, ev in enumerate(events[-MAX_TRANSCRIPT_EVENTS:]):
        out = (ev.get('output') or '')[:EV_OUT_CHARS].replace('\n', ' ⏎ ')
        lines.append(f'{i+1}) {ev["tool"]}: {json.dumps(ev["input"], ensure_ascii=False)[:700]}'
                     f' -> {out}')
    lines.append('Decide the next action. Emit exactly one JSON object: '
                 '{"tool": "<name>", "input": {...}} and nothing else.')
    return '\n\n'.join(lines)


def scenarios_from_events(events, system, task, source, session_id):
    """events: ordered [{tool, input, output}] -> scenario at each index >= MIN."""
    out = []
    idxs = [i for i in range(MIN_EVENTS_BEFORE, len(events))
            if events[i]['tool'] not in BOOKKEEP and events[i]['input']]
    if len(idxs) > CAP_PER_SESSION:
        step = len(idxs) / CAP_PER_SESSION
        idxs = [idxs[int(k * step)] for k in range(CAP_PER_SESSION)]
    for i in idxs:
        ctx = render(system, task, events[:i])
        out.append({'scenario_id': h(f'{source}:{session_id}:{i}'), 'source': source,
                    'session': f'{source}:{session_id}', 'context': ctx,
                    'target': {'tool': events[i]['tool'], 'input': events[i]['input']},
                    'pos': i, 'session_len': len(events)})
    return out


# ---------- source 1: opencode ----------
def from_opencode():
    scen = []
    path = '/tmp/opencode/harvest-2026-10-08/opencode_sessions_raw.jsonl'
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        events, task, got = [], None, False
        for m in r.get('messages') or []:
            role = m.get('role')
            texts = [p.get('text') for p in (m.get('parts') or [])
                     if isinstance(p, dict) and p.get('type') == 'text' and p.get('text')]
            if role == 'user' and not got and texts:
                task = texts[0][:4000]
                got = True
            for p in (m.get('parts') or []):
                if not (isinstance(p, dict) and p.get('type') == 'tool'):
                    continue
                st = p.get('state') or {}
                inp = st.get('input')
                if inp is None:
                    continue
                events.append({'tool': p.get('tool') or '?', 'input': inp,
                               'output': str(st.get('output') or '')})
        if len(events) >= MIN_EVENTS_BEFORE + 1:
            scen += scenarios_from_events(events, None, task, 'opencode', r['session'])
    return scen


# ---------- source 2: temporal ----------
def from_temporal():
    scen = []
    path = '/tmp/opencode/harvest-2026-10-08/temporal_claude_sessions.jsonl'
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        prompts = r.get('prompts') or []
        task = prompts[0].get('task_prompt') if prompts else None
        system = prompts[0].get('system_prompt_head') if prompts else None
        events = []
        for turn in r.get('turns') or []:
            for s in turn.get('stream') or []:
                if s.get('_truncated') and not s.get('message'):
                    continue
                msg = s.get('message') or {}
                for b in msg.get('content') or []:
                    if not isinstance(b, dict):
                        continue
                    if b.get('type') == 'tool_use' and b.get('name'):
                        events.append({'tool': b['name'], 'input': b.get('input') or {},
                                       'output': '', 'pending': True})
                    elif b.get('type') == 'tool_result' and events and events[-1].get('pending'):
                        events[-1]['output'] = str(b.get('content') or '')
                        events[-1]['pending'] = False
        events = [e for e in events if e['input'] or e['tool'] in ('Bash',)]
        if len(events) >= MIN_EVENTS_BEFORE + 1:
            scen += scenarios_from_events(events, system, task, 'temporal',
                                          r.get('session_id'))
    return scen


# ---------- source 3: local claude cli ----------
def from_claude_local():
    scen = []
    for path in sorted(glob.glob(os.path.expanduser('~/.claude/projects/*/*.jsonl'))):
        events, task, got = [], None, False
        try:
            lines = [json.loads(l) for l in open(path, errors='ignore') if l.strip()]
        except Exception:
            continue
        for rec in lines:
            if not isinstance(rec, dict):
                continue
            msg = rec.get('message')
            if not isinstance(msg, dict):
                continue
            role = msg.get('role')
            cont = msg.get('content')
            if not isinstance(cont, list):
                if isinstance(cont, str) and role == 'user' and not got:
                    task, got = cont[:4000], True
                continue
            for b in cont:
                if not isinstance(b, dict):
                    continue
                if b.get('type') == 'text' and role == 'user' and not got and b.get('text'):
                    task, got = b['text'][:4000], True
                elif b.get('type') == 'tool_use' and b.get('name'):
                    events.append({'tool': b['name'], 'input': b.get('input') or {},
                                   'output': '', 'pending': True})
                elif b.get('type') == 'tool_result' and events and events[-1].get('pending'):
                    c = b.get('content')
                    events[-1]['output'] = (c if isinstance(c, str) else json.dumps(c))[:2000]
                    events[-1]['pending'] = False
        if len(events) >= MIN_EVENTS_BEFORE + 1:
            scen += scenarios_from_events(events, None, task, 'claude-local',
                                          os.path.basename(path))
    return scen


def main():
    allsc = []
    for fn, name in [(from_opencode, 'opencode'), (from_temporal, 'temporal'),
                     (from_claude_local, 'claude-local')]:
        s = fn()
        print(f'{name}: {len(s)} scenarios from distinct sessions', flush=True)
        allsc += s
    # dedupe identical contexts
    seen, uniq = set(), []
    for s in allsc:
        k = h(s['context'] + s['target']['tool'] + json.dumps(s['target']['input'],
                                                              sort_keys=True))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s)
    print(f'deduped: {len(uniq)} (removed {len(allsc)-len(uniq)})')
    # session-disjoint split: 10% bench per source
    by_src = defaultdict(list)
    for s in uniq:
        by_src[s['source']].append(s)
    bench_ids = set()
    for src, items in by_src.items():
        sess = sorted({i['session'] for i in items})
        random.shuffle(sess)
        bench_ids |= set(sess[:max(1, len(sess) // 10)])
    train = [s for s in uniq if s['session'] not in bench_ids]
    bench = [s for s in uniq if s['session'] in bench_ids]
    with open(f'{OUT}/scenarios_train.jsonl', 'w') as f:
        for s in train:
            f.write(json.dumps(s, separators=(',', ':')) + '\n')
    with open(f'{OUT}/scenarios_bench.jsonl', 'w') as f:
        for s in bench:
            f.write(json.dumps(s, separators=(',', ':')) + '\n')
    tools = Counter(s['target']['tool'] for s in uniq)
    print(f'TOTAL: {len(uniq)}  train {len(train)}  bench {len(bench)}')
    print('target tool dist:', dict(tools.most_common(14)))
    srcs = Counter(s['source'] for s in uniq)
    print('by source:', dict(srcs))
    json.dump({'bench_sessions': sorted(bench_ids), 'total': len(uniq),
               'train': len(train), 'bench': len(bench)},
              open(f'{OUT}/splits.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
