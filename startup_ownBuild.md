# Start own Strata Build

## Build with existing models

```bash
cd /home/daniel/dev/ai/Strata

Q2_DIR="$(realpath /opt/qwen38-flash/models/Q2_0)"
env -u HIP_CLANG_PATH -u LD_LIBRARY_PATH \
  ROCM_PATH=/home/daniel/rocm-10.1-venv/lib/python3.14/site-packages/_rocm_sdk_devel \
  HIP_PATH=/home/daniel/rocm-10.1-venv/lib/python3.14/site-packages/_rocm_sdk_devel \
  ./setup.sh --backend hip --family qwen --model Q2_0 \
  --gguf-dir "$Q2_DIR" --no-start --yes
./run-q2_0.sh

env -u HIP_CLANG_PATH -u LD_LIBRARY_PATH \
  ROCM_PATH=/home/daniel/rocm-10.1-venv/lib/python3.14/site-packages/_rocm_sdk_devel \
  HIP_PATH=/home/daniel/rocm-10.1-venv/lib/python3.14/site-packages/_rocm_sdk_devel \
  ./run-unsloth-ud-iq4_xs.sh

env -u HIP_CLANG_PATH -u LD_LIBRARY_PATH \
  ROCM_PATH=/home/daniel/rocm-10.1-venv/lib/python3.14/site-packages/_rocm_sdk_devel \
  HIP_PATH=/home/daniel/rocm-10.1-venv/lib/python3.14/site-packages/_rocm_sdk_devel \
  ./run-unsloth-ud-q4_k_xl.sh

curl -sS http://127.0.0.1:8080/health | python -m json.tool
```

## Test generation

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "strata",
    "messages": [
      {
        "role": "user",
        "content": "Explain in one sentence why the sky is blue."
      }
    ],
    "max_tokens": 100,
    "reasoning_effort": "none"
  }' | python -m json.tool

```

## Benchmark comparison from 2026-10-05

These results were measured on the Radeon 8060S / Ryzen AI Max+ 395 system with 128 GB of unified memory. The
`gfx1151` runs used the smaller GSQ-RCO Q2_0 model; the Gufo, llama-server and second Strata runs used the same
Unsloth UD-Q4_K_XL model. Q2_0 and UD-Q4_K_XL are therefore listed separately where model size affects the result.

The prefill benchmark issued four fresh prompts (4,210, 8,830, 4,210 and 8,830 tokens) and a cached follow-up after
each one. "Weighted prefill" is the 26,080 fresh tokens divided by their combined engine prefill time. "Weighted
decode" is the 1,024 generated tokens across the eight fresh and follow-up requests divided by their combined
decode time. Warm-up is excluded from both figures.

### Prefill: Strata Q2_0 optimization arms

| Arm | Configuration | 4,210 first | 8,830 first | 4,210 second | 8,830 second | Weighted prefill | Weighted decode | Fresh-request wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | Plain hipBLAS, dense MMQ off | 261.6 | 250.9 | 247.5 | 247.7 | **250.91 tok/s** | 48.60 tok/s | 114.50 s |
| B | gfx1151 hipBLASLt tuning table | 549.7 | 525.0 | 526.1 | 524.6 | **528.91 tok/s** | 48.16 tok/s | 59.97 s |
| C | B plus dense MMQ | 554.8 | 526.0 | 528.5 | 523.4 | **529.96 tok/s** | 46.76 tok/s | 60.33 s |
| D | C plus explicit shared-expert stream on | 567.4 | 527.0 | 527.6 | 524.7 | **532.43 tok/s** | 46.77 tok/s | 60.10 s |
| E | C plus shared-expert stream off | 568.9 | 528.1 | 526.0 | 524.2 | **532.58 tok/s** | 45.63 tok/s | 60.32 s |

The hipBLASLt tuning table was the material prefill change: B was 2.11x A, or 110.8% faster. Dense MMQ and the
shared-expert stream changed weighted prefill by less than 1% after tuning. Comparing D and E isolates the stream
setting: enabling it increased weighted decode from 45.63 to 46.77 tok/s, a 2.5% gain. The source files are in
[`bench/results/2026-10-05-gfx1151`](bench/results/2026-10-05-gfx1151/README.md).

### Prefill: UD-Q4_K_XL engine comparison

| Engine/run | 4,210 first | 8,830 first | 4,210 second | 8,830 second | Weighted prefill | Weighted decode | Fresh-request wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gufo | 1,179.6 | 956.1 | 953.6 | 961.7 | **987.84 tok/s** | 33.20 tok/s | 42.06 s |
| Strata | 510.2 | 484.4 | 462.8 | 482.6 | **484.09 tok/s** | **33.79 tok/s** | 69.02 s |
| llama-server, run 1 | 340.3 | 348.3 | 353.7 | 347.6 | **347.60 tok/s** | 21.12 tok/s | 96.27 s |
| llama-server, run 2 | 325.7 | 344.8 | 350.6 | 350.4 | **344.30 tok/s** | 21.14 tok/s | 96.99 s |

For this workload, Gufo's weighted UD-Q4_K_XL prefill rate was 2.04x Strata's. Strata was 1.40x the mean of the
two llama-server runs. Strata and Gufo had similar weighted decode rates (Strata was 1.8% higher), while Strata was
1.60x the llama-server mean. The two llama-server repeats were close: their weighted prefill rates differed by
1.0% and their weighted decode rates by 0.1%.

Strata UD-Q4_K_XL was 9.1% slower at prefill than Strata Q2_0 arm E (484.09 versus 532.58 tok/s). This is useful as
a local size/quantization reference, but it is not an engine-only comparison because the model quantization differs.

Sources: [Gufo results](bench/results/2026-10-05-gufo-UD-Q4_K_XL/gufo-UD-Q4_K_XL.json),
[Strata results](bench/results/2026-10-05-Strata-UD-Q4_K_XL/Strata-UD-Q4_K_XL.json),
[llama-server run 1](bench/results/2026-10-05-llama-server-UD-Q4_K_XL/llama-server-UD-Q4_K_XL_1.json), and
[llama-server run 2](bench/results/2026-10-05-llama-server-UD-Q4_K_XL/llama-server-UD-Q4_K_XL2.json).

### Needle-in-a-haystack recall

| Target length | Depth | gfx1151 Q2_0 | Gufo UD-Q4_K_XL | llama-server UD-Q4_K_XL | Strata UD-Q4_K_XL |
| --- | ---: | --- | --- | --- | --- |
| 32K | 10% | Not run | **Found**, 35.0 s | Request failed: HTTP 400 | No saved result; request logged (66.877 s prefill) |
| 32K | 50% | Not run | **Found**, 36.5 s | Request failed: HTTP 400 | No saved result; request logged (66.186 s prefill) |
| 32K | 90% | Not run | **Found**, 22.6 s | Request failed: HTTP 400 | No saved result; request logged (33.045 s prefill)¹ |
| 128K | 10% | Not run | **Found**, 171.9 s | Request failed: HTTP 400 | No saved result; request logged (282.993 s prefill) |
| 128K | 50% | Not run | **Found**, 159.2 s | Request failed: HTTP 400 | No saved result |
| 128K | 90% | Not run | **Found**, 142.7 s | Request failed: HTTP 400 | No saved result |

Gufo is the only engine in these directories with a complete, scorable needle run: it found all six code words
(6/6), including all three depths near 125.5K prompt tokens. llama-server completed none of the six requests, so
its HTTP 400 results are transport/request failures rather than six recall misses. The gfx1151 Q2_0 directory has
no needle data.

The Strata engine log contains four completed needle-sized requests, but no `needles.json` was saved and the log
does not contain the returned code words. Their recall result therefore cannot be scored. The final two 128K cases
are absent. The listed Strata times are engine prefill times rather than the end-to-end seconds recorded by
`needle_bench.py` and should not be compared directly with Gufo's wall times.

¹ The third Strata 32K request reused 16,384 tokens and read 15,631 fresh tokens; the other listed Strata needle
requests had zero reused tokens.
