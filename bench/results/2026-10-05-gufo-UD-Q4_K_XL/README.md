# Benchmarks

## Starting gufo server

```bash
mkdir -p bench/results/2026-10-05-community-gufo-gfx1151

env LD_LIBRARY_PATH="$ROCM_SDK_ROOT/lib" \
  ./build/release/gufo serve llm \
  --model /opt/qwen38-flash/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp-model /opt/qwen38-flash/models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --mmproj /opt/qwen38-flash/models/mmproj-BF16.gguf \
  --host 127.0.0.1 \
  --port 8081 \
  --sessions 1 \
  --context 131072 \
  --speculative mtp \
  --think off \
  2>&1 | tee /home/daniel/dev/ai/Strata/bench/results/2026-10-05-gufo-UD-Q4_K_XL/gufo-q2_0-engine.log
```

## Running Qwen3.8-Flash-Next-UD-Q4_K_XL benchmarks

```bash
python tools/hip/bench_prefill.py \
  --url http://127.0.0.1:8081 \
  --model "Qwen3.8 Flash Next" \
  --engine-log bench/results/2026-10-05-gufo-UD-Q4_K_XL/gufo-UD-Q4_K_XL-engine.log \
  --log-format gufo \
  --output bench/results/2026-10-05-gufo-UD-Q4_K_XL/gufo-UD-Q4_K_XL.json \
  --label gufo-UD-Q4_K_XL
```

## Needles Benchmark (Precision)

```bash
cd ~/dev/ai/Strata
python tools/needle_bench.py --url http://127.0.0.1:8081 --lengths 32k,128k --depths 10,50,90 --out needles.json
```

Result see bench/results/2026-10-05-gufo-UD-Q4_K_XL/needles-gufo-UD-Q4_K_XL.json
