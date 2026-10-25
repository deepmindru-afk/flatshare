#!/usr/bin/env bash
# make-room-patches.sh VLLM_CLONE: write make-room's three patches (src/leaseholder_vllm/make-room/*.patch) from the
# backport's branches in a vLLM clone. Each branch is v0.32.0 with commits on top, and each patch is the scheduler.py
# part of one branch's commits, with the commits named in its header:
#   up-make-room-kv-v0.32.0   v0.32.0 + #63350's commits, cherry-picked            -> 63350.patch
#   up-budget-cap-v0.32.0     up-make-room-kv-v0.32.0 + #63351's commits           -> 63351.patch
#   up-make-room-v0.32.0      up-budget-cap-v0.32.0 + the variant's one commit     -> variant.patch
# A change to the PRs is made on their branches (vLLM main), cherry-picked onto these, and the patches rewritten
# here. KV_REF, CAP_REF and VARIANT_REF name other branches.
set -euo pipefail
clone=${1:?usage: make-room-patches.sh VLLM_CLONE}
kv=${KV_REF:-up-make-room-kv-v0.32.0}
cap=${CAP_REF:-up-budget-cap-v0.32.0}
variant=${VARIANT_REF:-up-make-room-v0.32.0}
out=$(cd "$(dirname "$0")/../src/leaseholder_vllm/make-room" && pwd)
scheduler=vllm/v1/core/sched/scheduler.py
git -C "$clone" merge-base --is-ancestor v0.32.0 "$kv"
git -C "$clone" merge-base --is-ancestor "$kv" "$cap"
git -C "$clone" merge-base --is-ancestor "$cap" "$variant"

write() {  # write FILE BASE TIP DESCRIPTION
  local file=$1 base=$2 tip=$3 description=$4 sha
  {
    printf '%s\n\n' "$description"
    printf 'The scheduler.py part of these commits on vLLM v0.32.0 (%s):\n\n' "$(git -C "$clone" rev-parse --short v0.32.0^{commit})"
    for sha in $(git -C "$clone" rev-list --reverse "$base..$tip"); do
      picked=$(git -C "$clone" log -1 --format=%b "$sha" | sed -n 's/^(cherry picked from commit \([0-9a-f]*\))$/\1/p')
      printf '  %s %s%s\n' "$(git -C "$clone" rev-parse --short "$sha")" "$(git -C "$clone" log -1 --format=%s "$sha")" \
        "${picked:+ (cherry-picked from ${picked:0:9})}"
    done
    printf '\n'
    git -C "$clone" diff --no-color "$base" "$tip" -- "$scheduler"
  } >"$out/$file"
}

write 63350.patch v0.32.0 "$kv" "vLLM #63350 (https://github.com/vllm-project/vllm/pull/63350), backported to v0.32.0."
write 63351.patch "$kv" "$cap" "vLLM #63351 (https://github.com/vllm-project/vllm/pull/63351), backported to v0.32.0, on #63350."
write variant.patch "$cap" "$variant" "What only Leaseholder's make-room variant has, on #63350 and #63351: the switches from
--additional-config's make_room table, the reserve, and the counters with their log line."
ls -l "$out"/*.patch
