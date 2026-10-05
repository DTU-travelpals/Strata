# Benchmarks

## Starting llama-server server

```bash
cd /home/daniel/dev/ai/Strata
mkdir -p bench/results/bench/results/2026-10-05-llama-server-UD-Q4_K_XL

read -rsp "llama-server API key: " STRATA_API_KEY
echo
export STRATA_API_KEY

journalctl \
  --unit llama-server.service \
  --follow \
  --lines 0 \
  --output cat \
  | tee bench/results/2026-10-05-llama-server-UD-Q4_K_XL/llama-server-UD-Q4_K_XL-engine.log

```

## Running Qwen3.8-Flash-Next-UD-Q4_K_XL benchmarks

```bash
python tools/hip/bench_prefill.py \
  --url http://127.0.0.1:8033 \
  --model qwen38-flash-ud-q4-k-xl-mtp \
  --engine-log bench/results/2026-10-05-llama-server-UD-Q4_K_XL/llama-server-UD-Q4_K_XL-engine.log \
  --log-format llama-server \
  --output bench/results/2026-10-05-llama-server-UD-Q4_K_XL/llama-server-UD-Q4_K_XL.json \
  --label llama-server-UD-Q4_K_XL
```

## Needles Benchmark

```bash
cd ~/dev/ai/Strata
python tools/needle_bench.py --api-key LLAMA_API_T0msU --url http://127.0.0.1:8033 --lengths 32k,128k --depths 10,50,90 --out bench/results/2026-10-05-llama-server-UD-Q4_K_XL/needles.json
```

## Problems starting model on llama-server

Unfortunately, the model `qwen38-flash-ud-q4-k-xl-mtp` does not reliably load with llama-server, due to memory problems. Please close everything else and use the current mem config in models.ini.

```log
Okt 05 16:41:38 tuxai systemd[1]: Started llama.cpp Model Router.
Okt 05 16:43:01 tuxai systemd[1]: [🡕] llama-server.service: systemd-oomd killed 79 process(es) in this unit.
Okt 05 16:43:01 tuxai systemd[1]: llama-server.service: Main process exited, code=killed, status=9/KILL
Okt 05 16:43:01 tuxai systemd[1]: llama-server.service: Failed with result 'oom-kill'.
Okt 05 16:43:01 tuxai systemd[1]: llama-server.service: Consumed 25.690s CPU time over 1min 23.032s wall clock time, 36.1G memory peak, 10.6G memory swap peak.
Okt 05 16:43:06 tuxai systemd[1]: llama-server.service: Scheduled restart job, restart counter is at 1.
```
