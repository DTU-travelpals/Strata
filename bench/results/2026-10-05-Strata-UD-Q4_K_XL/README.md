# Strata UD-Q4_K_XL benchmarks

## One-time pack preparation

The four model shards are in `/opt/qwen38-flash/models/UD-Q4_K_XL`. Build the small Strata pack once; the routed
experts continue to be read directly from the GGUF shards.

```bash
cd /home/daniel/dev/ai/Strata

STRATA_GGUF_PY="$PWD/third_party/llama.cpp/gguf-py" \
  .venv/bin/python tools/iq_pack.py \
  --gguf /opt/qwen38-flash/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --out /home/daniel/dev/ai/Strata-data/packs/unsloth-ud-q4_k_xl \
  --compat-bf16
```

## Starting Strata

From the project root, run:

```bash
./run-unsloth-ud-q4_k_xl.sh
```

The launcher applies the Test case E environment from `../2026-10-05-gfx1151/README.md`:

```text
STRATA_SH_STREAM=0
STRATA_DENSE_MMQ=1
STRATA_HIPBLASLT_TUNING=/home/daniel/dev/ai/Strata/tools/hip/gfx1151-hipblaslt-100500.txt
```

The server uses `strata-unsloth-ud-q4_k_xl.json` with these startup parameters:

```text
--pack /home/daniel/dev/ai/Strata-data/packs/unsloth-ud-q4_k_xl
--native /opt/qwen38-flash/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
--expert-profile /home/daniel/dev/ai/Strata/data/expert-profile.bin
--expert-cache auto
--prefill auto
--spec 4
--spec-min-p 0.5
--mtp /home/daniel/dev/ai/Strata-data/mtp/rt
--max-context 131072
--kv int8
--kv-resident 32768
--resident-budget-gib 71
--vision
--vram-reserve-mib 700
```

MTP is enabled by `--mtp` and `--spec 4`. Image input is enabled by `--vision`; the server's vision configuration
uses `/opt/qwen38-flash/models/mmproj-BF16.gguf` on the CPU with 300 image tokens and 16 threads. The server listens
on `127.0.0.1:8080` and writes its engine log to `Strata-UD-Q4_K_XL-engine.log` in this directory.

## Running the prefill benchmark

```bash
cd /home/daniel/dev/ai/Strata

python tools/hip/bench_prefill.py \
  --url http://127.0.0.1:8080 \
  --model qwen3.8-flash-next-unsloth-ud-q4_k_xl \
  --engine-log bench/results/2026-10-05-Strata-UD-Q4_K_XL/Strata-UD-Q4_K_XL-engine.log \
  --log-format strata \
  --output bench/results/2026-10-05-Strata-UD-Q4_K_XL/Strata-UD-Q4_K_XL.json \
  --label strata-UD-Q4_K_XL
```

## Running the needle benchmark

```bash
cd /home/daniel/dev/ai/Strata

python tools/needle_bench.py \
  --url http://127.0.0.1:8080 \
  --lengths 32k,128k \
  --depths 10,50,90 \
  --out bench/results/2026-10-05-Strata-UD-Q4_K_XL/needles.json
```
