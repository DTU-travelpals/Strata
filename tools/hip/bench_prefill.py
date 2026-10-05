#!/usr/bin/env python3
"""Matched fresh/follow-up prefill probe against an otherwise idle local LLM server.

Does not start/stop services. Use a dedicated server and its engine log, restart
between build/configuration arms, and keep model/template/settings identical.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time
import urllib.request

STRATA_PATTERN = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([0-9.]+) tok/s\), (\d+) generated in (\d+) ms \(([0-9.]+) tok/s\)')
LLAMA_SERVER_PATTERN = re.compile(
    r'prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens .*?([0-9.]+) tokens per second\).*?'
    r'\beval time =\s*([0-9.]+) ms /\s*(\d+) tokens .*?([0-9.]+) tokens per second\)',
    re.DOTALL)
LOG_VALUE_PATTERN = re.compile(r'(?<!\S)([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)')
FIELDS = ('prompt_tokens', 'reused', 'fresh', 'prefill_ms', 'prefill_tps', 'generated', 'decode_ms', 'decode_tps')


def parse_metrics(text, log_format):
    if log_format == 'strata':
        return [dict(zip(FIELDS, map(float, match))) for match in STRATA_PATTERN.findall(text)]
    if log_format == 'llama-server':
        results = []
        for prefill_ms, fresh, prefill_tps, decode_ms, generated, decode_tps in LLAMA_SERVER_PATTERN.findall(text):
            values = (fresh, 0, fresh, prefill_ms, prefill_tps, generated, decode_ms, decode_tps)
            results.append(dict(zip(FIELDS, map(float, values))))
        return results

    results = []
    required = ('prompt_tokens', 'prefill_tokens', 'generated_tokens', 'prefill_tps', 'decode_tps')
    for line in text.splitlines():
        if 'event=completed' not in line or 'path=/v1/chat/completions' not in line or 'status=200' not in line:
            continue
        values = dict(LOG_VALUE_PATTERN.findall(line))
        if not all(field in values for field in required):
            continue
        prompt = float(values['prompt_tokens'])
        fresh = float(values['prefill_tokens'])
        generated = float(values['generated_tokens'])
        prefill_tps = float(values['prefill_tps'])
        decode_tps = float(values['decode_tps'])
        reused = float(values.get('cached_tokens', prompt - fresh))
        prefill_ms = float(values['prefill_ms']) if 'prefill_ms' in values else (
            fresh / prefill_tps * 1000 if prefill_tps else 0)
        decode_ms = float(values['decode_ms']) if 'decode_ms' in values else (
            generated / decode_tps * 1000 if decode_tps else 0)
        results.append(dict(zip(FIELDS, (prompt, reused, fresh, prefill_ms, prefill_tps,
                                         generated, decode_ms, decode_tps))))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--model', required=True)
    p.add_argument('--engine-log', type=Path, required=True)
    p.add_argument('--log-format', choices=('strata', 'gufo', 'llama-server'), default='strata',
                   help='completed-request timing line format (default: strata)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', required=True)
    args = p.parse_args()
    results = []

    def request(messages, kind, trial):
        offset = args.engine_log.stat().st_size
        body = dict(model=args.model, messages=messages, max_tokens=128,
                    temperature=0, top_k=1, top_p=1, min_p=0, seed=42,
                    reasoning_effort='none')
        if args.log_format == 'llama-server':
            # Do not let a nominally fresh case reuse the previous case's short
            # common prefix. Its follow-up keeps llama-server's cache enabled.
            body['cache_prompt'] = kind != 'fresh'
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('STRATA_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['STRATA_API_KEY']
        req = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
                                     data=json.dumps(body).encode(), headers=headers)
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=360) as response:
            data = json.load(response)
        elapsed = time.monotonic() - start
        # The API response may arrive just before stderr's completion line is flushed.
        deadline = time.monotonic() + 2
        while True:
            with args.engine_log.open('rb') as log:
                log.seek(offset)
                matches = parse_metrics(log.read().decode(errors='replace'), args.log_format)
            if matches or time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        if len(matches) != 1:
            raise RuntimeError('Expected exactly one completed engine timing line; check idle server and log path')
        metrics = matches[0]
        if args.log_format == 'llama-server':
            usage = data.get('usage') or {}
            details = usage.get('prompt_tokens_details') or {}
            timings = data.get('timings') or {}
            metrics['prompt_tokens'] = float(usage.get('prompt_tokens',
                                                      metrics['fresh'] + timings.get('cache_n', 0)))
            metrics['reused'] = float(details.get('cached_tokens', timings.get('cache_n', 0)))
        if kind == 'fresh' and metrics['reused'] != 0:
            raise RuntimeError('Fresh prompt reused cached tokens; restart this arm before comparison')
        choice = data['choices'][0]
        results.append(dict(label=args.label, kind=kind, trial=trial, wall_s=elapsed,
                            metrics=metrics, usage=data.get('usage'),
                            finish_reason=choice.get('finish_reason'), message=choice['message']))
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(results[-1]), flush=True)
        return choice['message']

    request([{'role': 'user', 'content': 'Reply READY.'}], 'warmup', 0)
    for trial, n in enumerate((140, 280, 140, 280), 1):
        code = '\n'.join(f'export function rule{i}(x) {{ return x === {i} ? x + {i+1} : x - {i}; }}' for i in range(n))
        messages = [{'role': 'user', 'content': f'CASE {trial}: Review this source and describe its behavior precisely in a paragraph.\n' + code}]
        reply = request(messages, 'fresh', trial)
        messages += [reply, {'role': 'user', 'content': 'Explain the most relevant boundary case and the smallest useful regression test. ' + 'Focus on integer equality, zero, negative input, unexpected types, and caller assumptions. ' * 5}]
        request(messages, 'followup', trial)


if __name__ == '__main__':
    main()
