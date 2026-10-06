#!/bin/sh
cd "/home/daniel/dev/ai/Strata"
export STRATA_SH_STREAM=0
export STRATA_DENSE_MMQ=1
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1151-hipblaslt-100500.txt"
exec "/home/daniel/dev/ai/Strata/.venv/bin/python" "/home/daniel/dev/ai/Strata/serve/server.py" "--engine" "strata" "--config" "/home/daniel/dev/ai/Strata/strata-unsloth-ud-iq4_xs.json" "--port" "8080" "--host" "0.0.0.0" "--api-key" "T0msU" "--open"
