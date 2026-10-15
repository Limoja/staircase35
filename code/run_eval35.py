"""Run a model against the scenario bench via an OpenAI-compatible endpoint.

Usage: python3 run_eval.py --base http://HOST:PORT/v1 --model NAME \
           --bench scenarios_bench.jsonl --k 4 --out results/
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_v2 import score_one, bucket  # noqa: E402

lock = threading.Lock()
done = [0]


def chat(args, msgs):
    body = {'model': args.model, 'messages': msgs, 'temperature': args.temp,
            'top_p': args.top_p, 'max_tokens': args.max_tokens}
    if args.top_k:
        body['top_k'] = args.top_k
    if args.effort:
        body['chat_template_kwargs'] = {'reasoning_effort': args.effort}
    elif os.environ.get('NO_THINK') == '1':
        body['chat_template_kwargs'] = {'enable_thinking': False}
    body = json.dumps(body).encode()
    req = urllib.request.Request(f'{args.base.rstrip("/")}/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        d = json.load(r)
    m = d['choices'][0]['message']
    text = m.get('content') or ''
    tcs = m.get('tool_calls') or []
    if tcs:
        # structured tool_calls win over raw text: score the function, not the format
        parts = []
        for tc in tcs:
            fn = tc.get('function') or {}
            a = fn.get('arguments')
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    pass
            parts.append('<tool_call>\n' + json.dumps(
                {'name': fn.get('name'), 'arguments': a}, ensure_ascii=False) + '\n</tool_call>')
        text = '\n'.join(parts)
    return text


def run_one(args, row):
    msgs = [{'role': 'user', 'content': row['context']}]
    outs = []
    for k in range(args.k):
        last = None
        for attempt in range(4):
            try:
                outs.append(chat(args, msgs))
                last = None
                break
            except Exception as e:
                last = f'__ERROR__ {type(e).__name__}: {str(e)[:120]}'
                time.sleep(2 * (attempt + 1))
        if last is not None:
            outs.append(last)
    scores = [score_one(o, row['target']) for o in outs]
    with lock:
        done[0] += 1
        if done[0] % 25 == 0:
            print(f'  {done[0]} scenarios', flush=True)
    return {'scenario_id': row['scenario_id'], 'source': row['source'],
            'session': row['session'], 'target_tool': row['target']['tool'],
            'bucket': bucket(row['target']['tool']),
            'samples': [{'text': t[:4000], 'score': s, 'parts': p}
                        for t, (s, p) in zip(outs, scores)],
            'best': max(s for s, _ in scores),
            'first': scores[0][0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--bench', default='/tmp/opencode/eval-v2/scenarios_bench.jsonl')
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--max-tokens', type=int, default=400)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sample', type=int, default=0,
                    help='random N scenarios, seed 42 (prefix subsets are biased — manifest §13)')
    ap.add_argument('--effort', default='',
                    help="reasoning_effort for chat_template_kwargs, e.g. xhigh")
    ap.add_argument('--top-p', type=float, default=0.8)
    ap.add_argument('--top-k', type=int, default=0)
    ap.add_argument('--timeout', type=int, default=1200)
    ap.add_argument('--no-think', action='store_true')
    ap.add_argument('--out', default='/tmp/opencode/eval-v2/results')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(l) for l in open(args.bench) if l.strip()]
    if args.sample:
        import random
        random.seed(42)
        rows = random.sample(rows, min(args.sample, len(rows)))
    elif args.limit:
        rows = rows[:args.limit]
    partial_path = f'{args.out}/partial-{args.model.replace("/", "_")}.jsonl'
    results = []
    if os.path.exists(partial_path):
        done_ids = set()
        with open(partial_path) as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    results.append(r)
                    done_ids.add(r['scenario_id'])
        rows = [r for r in rows if r['scenario_id'] not in done_ids]
        print(f'resuming: {len(done_ids)} done, {len(rows)} remaining', flush=True)
    partial_f = open(partial_path, 'a')
    print(f'eval {args.model} on {len(rows)} scenarios, k={args.k}', flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, args, r) for r in rows]
        for f in cf.as_completed(futs):
            r = f.result()
            results.append(r)
            partial_f.write(json.dumps(r, separators=(',', ':')) + '\n')
            partial_f.flush()
            os.fsync(partial_f.fileno())
    partial_f.close()
    tag = time.strftime('%Y%m%d-%H%M%S')
    path = f'{args.out}/{tag}.jsonl'
    with open(path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    import statistics
    from collections import defaultdict
    best = [r['best'] for r in results]
    first = [r['first'] for r in results]
    byb = defaultdict(list)
    for r in results:
        byb[r['bucket']].append(r['best'])
    bys = defaultdict(list)
    for r in results:
        bys[r['source']].append(r['best'])
    err = sum(1 for r in results for s in r['samples'] if s['text'].startswith('__ERROR__'))
    summ = {
        'model': args.model, 'n': len(results), 'k': args.k, 'effort': args.effort,
        'temp': args.temp, 'top_p': args.top_p, 'top_k': args.top_k,
        'best_of_k': round(sum(best) / max(len(best), 1), 4),
        'pass_at_1': round(sum(first) / max(len(first), 1), 4),
        'solved_best1': sum(1 for b in best if b > 0.9),
        'api_errors': err,
        'secs': round(time.time() - t0, 1),
        'by_bucket': {k: round(sum(v) / len(v), 4) for k, v in sorted(byb.items())},
        'by_source': {k: round(sum(v) / len(v), 4) for k, v in sorted(bys.items())},
    }
    json.dump(summ, open(f'{args.out}/{tag}-summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
