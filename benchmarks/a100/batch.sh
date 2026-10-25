#!/usr/bin/env bash
# batch.sh [--rounds N]: the A100 run's optional arm 4 (DESIGN.md §5.4, P-1's slot S-7), on this host: the scheduler's
# cost at a datacenter batch. vllm bench serve keeps 256 requests in flight on the core's engine, 26 of them (10%) at
# priority 0 and 230 at priority 2 (two clients side by side, each its requests' priority in --extra-body), random
# 1,024-token prompts and 256 output tokens, N rounds (20) of each client's concurrency, on three engines in turn:
# frozen (stock vLLM 0.32.0), make-room with its switches off (its code path, nothing it does), and make-room with
# the PRs' values. Each client's result (vllm bench serve's JSON: request and output throughput, TTFT, TPOT and ITL
# percentiles) goes to ~/leaseholder/runs/a100/batch/ENGINE-pPRIORITY.json; a result already there is kept.
# vllm bench serve speaks TCP and the engine serves on its socket only: the clients reach it through
# deploy/bin/lh-forward, from a loopback port (LH_FORWARD_PORT, 18000) to the socket, on their own cores, the same
# hop for each engine.
set -euo pipefail
rounds=20
while (($#)); do
  case $1 in
    --rounds)
      [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || { echo "batch: --rounds takes a positive number" >&2; exit 2; }
      rounds=$2
      shift
      ;;
    *) echo "usage: batch.sh [--rounds N]" >&2; exit 2 ;;
  esac
  shift
done
root=$HOME/leaseholder
node=${LH_NODE_DIR:-$root/node}
src=$(cd "$(dirname "$0")/../.." && pwd)
port=${LH_FORWARD_PORT:-18000}
engine=http://127.0.0.1:$port # lh-forward's end of the engine's socket
out=$root/runs/a100/batch
mkdir -p "$out"
# shellcheck source=/dev/null
[[ ! -f $node/env.sh ]] || source "$node/env.sh"
sock=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/lh/engine.sock # the engine's own socket (lh-engine.service)
log() { echo "[$(date -u +%FT%TZ)] batch: $*" | tee -a "$out/batch.log"; }
trap 'jobs -p | xargs -r kill 2>/dev/null || true' EXIT # a client left when the other failed
cores=$(sed -n 's/^aiperf_cores = "\(.*\)"$/\1/p' "$HOME/.config/leaseholder/e2-host.toml")
[[ -n $cores ]] || { log "no aiperf_cores in ~/.config/leaseholder/e2-host.toml (deploy/a100/host.py config)"; exit 2; }
taskset -c "$cores" "$root/venvs/bench/bin/python" "$src/deploy/bin/lh-forward" "127.0.0.1:$port=unix:$sock" \
  >"$out/forward.log" 2>&1 &

client() { # client NAME PRIORITY CONCURRENCY SEED: one vllm bench serve, in the background
  local name=$1 priority=$2 concurrency=$3 seed=$4
  HF_HUB_OFFLINE=1 taskset -c "$cores" "$root/venvs/engine/bin/vllm" bench serve --backend vllm \
    --base-url "$engine" --endpoint /v1/completions --model "$model" --tokenizer "$model" \
    --dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos --seed "$seed" \
    --num-prompts $((rounds * concurrency)) --max-concurrency "$concurrency" \
    --extra-body "{\"priority\": $priority}" --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99 \
    --save-result --result-dir "$out" --result-filename "$name-p$priority.partial.json" \
    >"$out/$name-p$priority.log" 2>&1
}

measure() { # measure NAME VARIANT [FLAG...]
  local name=$1 a b
  shift
  if [[ -f $out/$name-p0.json && -f $out/$name-p2.json ]]; then
    log "$name: done before"
    return
  fi
  log "$name: $*"
  bash "$node/variant.sh" use "$@" 2>&1 | tee -a "$out/batch.log"
  model=$(curl -fsS -m 10 --unix-socket "$sock" http://localhost/v1/models | "$root/venvs/engine/bin/python" -c \
    'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])')
  client "$name" 2 230 2 &
  a=$!
  client "$name" 0 26 0 &
  b=$!
  wait "$a" || { log "$name: the priority-2 client failed ($out/$name-p2.log)"; exit 1; }
  wait "$b" || { log "$name: the priority-0 client failed ($out/$name-p0.log)"; exit 1; }
  for priority in 0 2; do
    mv "$out/$name-p$priority.partial.json" "$out/$name-p$priority.json"
  done
  log "$name: $(grep -h -E '^Output token throughput' "$out/$name-p2.log" "$out/$name-p0.log" | tr '\n' ' ')"
}

measure frozen frozen
measure off make-room --additional-config.make_room.max_preemptions_per_step 0 \
  --additional-config.make_room.budget_cap false --additional-config.make_room.reserve_tokens 0
measure on make-room --additional-config.make_room.reserve_tokens 0 --additional-config.make_room.budget_cap_tokens 48
bash "$node/variant.sh" use frozen 2>&1 | tee -a "$out/batch.log"
log "complete"
