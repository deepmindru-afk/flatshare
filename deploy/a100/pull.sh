#!/usr/bin/env bash
# pull.sh [--once] [--every SECONDS] HOST [DIR]: copy a single host's measurements to this workstation as they are
# made: its campaigns' results (campaign.sh --local's ~/leaseholder/results), its other runs (~/leaseholder/runs: the
# probes, E0's tiers, the calibration), its node directories (node and node-14b: the host's facts, the rendered
# configs, the preflight) and the warm starts' and JIT check's records (cache/warm.json, jit-check.json), into DIR
# (default ~/leaseholder/hosts/HOST), every 120 s until stopped, or once. Each copy is rsync's: only what changed moves, nothing is deleted on either side, and AIPerf's copies of the
# dataset (aiperf/inputs.json) stay on the host. A copy that fails is logged and tried again at the next round, so a
# dropped link loses nothing. LH_SSH is your ssh command for the host, as for benchmarks/e2/campaign.sh.
set -uo pipefail
usage() { echo "usage: pull.sh [--once] [--every SECONDS] HOST [DIR]" >&2; exit 2; }
once=false every=120
while [[ ${1:-} == --* ]]; do
  case $1 in
    --once) once=true ;;
    --every)
      [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || usage
      every=$2
      shift
      ;;
    *) usage ;;
  esac
  shift
done
(($# == 1 || $# == 2)) || usage
host=$1
dir=${2:-$HOME/leaseholder/hosts/${host%%.*}}
ssh=${LH_SSH:-ssh}
mkdir -p "$dir"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$dir/pull.log"; }
pull() { # pull: one round; the status of the last copy that failed
  local status=0 what
  for what in results runs node node-14b; do
    if $ssh "$host" "test -d leaseholder/$what"; then
      rsync -a -e "$ssh" --exclude aiperf/inputs.json "$host:leaseholder/$what/" "$dir/$what/" ||
        { status=$?; log "pull: $what from $host failed (rsync exit $status); next round"; }
    fi
  done
  for what in warm.json jit-check.json; do
    if $ssh "$host" "test -f leaseholder/cache/$what"; then
      mkdir -p "$dir/cache"
      rsync -a -e "$ssh" "$host:leaseholder/cache/$what" "$dir/cache/" ||
        { status=$?; log "pull: cache/$what from $host failed (rsync exit $status); next round"; }
    fi
  done
  return "$status"
}
trap 'log "pull: stopped"; exit 0' INT TERM
while :; do
  pull
  status=$?
  ((status)) || log "pull: $host up to date in $dir"
  if $once; then exit "$status"; fi
  sleep "$every" &
  wait $!
done
