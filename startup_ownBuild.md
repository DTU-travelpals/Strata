# Start own Strata Build

## Build with existing models

```bash
Q2_DIR="$(realpath /opt/qwen38-flash/models/Q2_0)"

cd /home/daniel/dev/ai/Strata

env -u HIP_CLANG_PATH -u LD_LIBRARY_PATH \
  ROCM_PATH=/home/daniel/rocm-10.1-venv/lib/python3.14/site-packages/_rocm_sdk_devel \
  HIP_PATH=/home/daniel/rocm-10.1-venv/lib/python3.14/site-packages/_rocm_sdk_devel \
  ./setup.sh --backend hip --family qwen --model Q2_0 \
  --gguf-dir "$Q2_DIR" --no-start --yes

#./build-hip-gfx1151-102/strata

./run-q2_0.sh

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