# Benchmarks

## Starting gufo server

```bash
mkdir -p bench/results/2026-10-05-community-gufo-gfx1151


```

## Running Qwen3.8-Flash-Next-UD-Q4_K_XL benchmarks

```bash
python tools/hip/bench_prefill.py \
  --url http://127.0.0.1:8080 \
  --model "Qwen3.8 Flash Next" \
  --engine-log Strata-UD-Q4_K_XL-engine.log \
  --log-format gufo \
  --output bench/results/2026-10-05-Strata-UD-Q4_K_XL/Strata-UD-Q4_K_XL.json \
  --label strata-UD-Q4_K_XL
```

## Needles Benchmark (Precision)

```bash
cd ~/dev/ai/Strata
python tools/needle_bench.py --url http://127.0.0.1:8080 --lengths 32k,128k --depths 10,50,90 --out bench/results/2026-10-05-Strata-UD-Q4_K_XL/needles.json
```

Result see bench/results/2026-10-05-gufo-UD-Q4_K_XL/needles-gufo-UD-Q4_K_XL.json
