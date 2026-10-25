# Running Leaseholder on other hardware

Every measurement so far ran on one RTX 5060 Ti 16 GB, the frozen box, with Gray, a second one, for the probes and the two-node points. This page says what the deployment needs from a GPU node, how a node's files come from a hardware profile, what changes on a card or a host that is not the box's, how one host runs both the GPU servers and the control plane, and how the A100 40 GB run is laid out. A number here is measured (on the box or Gray, with its source), modeled (lh-sim's forecast, [`benchmarks/a100/forecast.md`](../benchmarks/a100/forecast.md)) or an estimate, and says which.

## What runs where

What vLLM 0.32.0 runs for the box's checkpoint, Qwen3-4B-Instruct-2507-FP8 (block-scaled FP8 weights), by the card's compute capability, from vLLM's kernel selection order and its attention backends' priorities, and how far each is checked:

| Cards (compute capability) | Linear layers | Attention, KV layout | Status |
|---|---|---|---|
| RTX 5060 Ti (sm120): the box, Gray | FP8 W8A8 (DeepGEMM) | FlashAttention 2, LBHNC | **measured**: E0, E1's calibration, E2, the make-room probes |
| A100 40 GB, SXM4 or PCIe (sm80) | Marlin W8A16: the FP8 weights, BF16 activations, the BF16 tensor rate | FlashAttention 2, LBHNC | **modeled** ([the forecast](../benchmarks/a100/forecast.md)); the run below is prepared, and the Marlin path's [dry run on the box](#the-marlin-dry-run-on-the-box) is its first test on a GPU |
| A10 (sm86), L4, L40S, RTX 4090 (sm89) | Marlin W8A16, as on the A100 (vLLM 0.32.0 sends this checkpoint to Marlin on sm89 too) | FlashAttention 2 | **modeled** (`lh-sim hardware`) |
| H100, H200 (sm90), B200 (sm100) | FP8 W8A8 | FlashAttention 3 (sm90), FlashInfer (sm100) | **modeled** |
| T4 (sm75) | Marlin with FP16 activations | Triton attention | **modeled**; the preflight labels both |
| Older than sm75, CPU-only, ROCm | — | — | not supported by this deployment: vLLM 0.32.0 needs sm75, and the CPU KV tier's platform differs off CUDA |

The KV layout follows the CPU tier's connector, not the card: vLLM's `OffloadingConnector` requires LBHNC, so every CUDA card with the tier runs it. What changes in what the results mean is in [the preflight](#the-preflight): within one card, the frozen engine against make-room compares as on the box; across cards, only in the regime's coordinates and each card's distance to its own SLO.

## The hardware profile

A GPU node's deployment is a function of a few inputs: the card (its memory, compute capability, driver), the host (CPUs by NUMA node, RAM, `/dev/shm`, systemd), the model and the rules below. [`deploy/profiles/`](../deploy/profiles) holds one profile per node type, and [`deploy/profile.py`](../deploy/profile.py) (Python's standard library only, so that a bare host runs it before any venv exists) writes a node's directory from one:

```sh
python3 deploy/profile.py detect --out host.json                # the host's facts: nvidia-smi, /sys, /proc, systemd, toolkits
python3 deploy/profile.py render a100-40gb --host host.json --out ~/leaseholder/node
python3 deploy/profile.py check rtx-5060-ti                     # the box's profile renders deploy/phastform byte for byte
~/leaseholder/venvs/engine/bin/python deploy/profile.py check-vllm ~/leaseholder/node   # vLLM's own parser, no engine
```

A node's directory holds `engine.yaml`, `stt.yaml`, the GPU units (`leaseholder.target`, `lh-engine`, `lh-stt`, `lh-tts`, `lh-mps`), each server on its own Unix socket as the box's, the variants' drop-ins, the measurement drop-in (`measure/lh-engine.conf`, vLLM's development routes, which `variant.sh measure on` links), `variant.sh`, `install.sh`, and `node.json`: every value as resolved, the install path and the core plan, with the warnings. [`deploy/templates/`](../deploy/templates) holds the box's files with their values as placeholders; the comment lines that state a node's own facts are profile fields. The box's profile (`rtx-5060-ti`) holds the box's values only, and a test holds its rendering to `deploy/phastform`, so no measurement changes meaning; a file added to the box's directory has to be rendered, copied or named box-only before the tests pass. The A100's profile (`a100-40gb`) keeps the box's model, pool, step cap and CPU tier ratio and derives the rest from the host ("auto"); `a100-40gb-qwen3-14b` names it as its base and changes the model, the pool and the directory.

## The rules

The profile's "auto" values, in `deploy/profile.py`:

- **R1, the pool.** Like-for-like, the box's 36,400 tokens (5 GiB of Qwen3-4B's KV), so that a card runs the box's comparison; or the card's own pool, what the box keeps beside its pool left to the model's KV in whole blocks (lh-sim's sizing, modeled, checked at the first start).
- **R2, the memory budgets.** vLLM refuses a start whose free memory is below `gpu-memory-utilization` × the card's memory even when `kv-cache-memory` fixes the pool, so the box's budgets port in GiB: 10.83 GiB for the engine and 2.32 GiB for Whisper, 0.28 and 0.06 of a 40 GiB card.
- **R3, the CPU tier.** The box's ratio, four times the pool, bounded by what `/dev/shm` holds less 2 GiB and by the RAM a single host leaves after 25 GiB for its other processes [estimate]; refused below the pool's own size, since a tier no larger than the pool only mirrors it.
- **R4, the step cap.** The core keeps the box's 512/256. The time-equivalent cap is the largest multiple of 256 whose chunk takes no longer than the box's 512-token chunk at the scenario's median context, its long-prefill threshold half of it: 1,536 on the A100, central (2,048 low, 1,280 high) [modeled].
- **R5, the probes.** `probe.py` and `tail.py` size their requests from the engine's pool and context: on the box two KV-case requests of 17,999 tokens, three prefill-case requests of 10,000 and 12 tail agents.
- **R6, the scenario.** E2's corpus, its two callers and the sessions for the box's KV pressure; another load or model is an optional arm.
- **R7, the core plan.** On one host, the engine, the audio servers, the gateway, the voice plane, AIPerf and the callers share the CPUs 8:4:2:8:8:4 (34 give each its want), the engine on the GPU's NUMA node and the audio servers off it; below 17, half the wants, the plan still places every component and warns that the host checks the mechanics, not the numbers.
- **R8, the install path.** The box's lock (vLLM 0.32.0, torch 2.13.0+cu130) where the driver runs CUDA 13.0, natively from R580, or through `cuda-compat-13-0` on a data-center card with R535 or R570 (the units then put its libraries on `LD_LIBRARY_PATH`); the cu129 lock ([`deploy/a100/requirements`](../deploy/a100/requirements): vLLM 0.32.0+cu129 with torch 2.13.0+cu129, every other package at the box's version) on any other driver from R525; refused below. FlashInfer's JIT compiles with a toolkit of the runtime's CUDA version, 13.0 or 12.9: the host's, else the venv's CUDA 13.0 (with `cache/cuda_lib`, which links it: the venv's `nvidia/cu13` has no `lib64` stubs), else CUDA 12.9.1's toolkit, which bootstrap installs into `~/leaseholder/cuda-12.9` from NVIDIA's runfile without root (CUDA 12's wheels carry no `nvcc`).

## The preflight

`deploy/a100/host.py preflight NODE [--gpu]` checks what a campaign depends on and writes `NODE/preflight.json`; `bootstrap.sh` ends each stage with it and fails on a refused check, and `lh-bench e2 run` keeps every check that was not ok in each point's `run.json`. A check that needs the card or the engine's first start is pending until then.

| Check | Degraded, and what it changes | Refused |
|---|---|---|
| Compute capability and precision path | sm75-sm89: W8A16 through Marlin at the BF16 rate; the per-token cost is the BF16 path's, comparisons across cards only in regime coordinates | below sm75 |
| BF16 | sm75: FP16 activations; numerics differ from the box | |
| Attention backend and KV layout | not FlashAttention or FlashInfer on LBHNC (the engine log): attention's cost | |
| Driver and install path | cu130 with the compatibility libraries, or the cu129 lock: another CUDA runtime under the same vLLM and torch | the venv's build is not the one the driver needs |
| torch on the GPU (`--gpu`) | | `torch.cuda.init()` fails with the engine's environment (errors 803, 804) |
| FlashInfer's JIT | | its sampler does not build (the engine would fail at its first start; `VLLM_USE_FLASHINFER_SAMPLER=0` in a drop-in is the degraded way out) |
| Host compilers | | no `cc` or `c++`: Triton builds its launchers with a C compiler, `nvcc` needs a C++ one |
| Compile and JIT caches | cold (each compile configuration's first start compiles: 531.8 s on the box against 88.8 s warm, measured in W8's start drills), on tmpfs, or under 10 GiB free [estimate] | under 2 GiB free |
| CPU tier and `/dev/shm` | below four times the pool: fewer restores, more recompute; the ratio is in every record | (R3 refuses at render) |
| Pinned memory | `cudaHostRegister` failed: a pageable tier, slower restores | |
| Host RAM | the GPU's NUMA node short of the tier: a remote tier (the box measured one within 1% of pinned bandwidth) | |
| Async scheduling, block size | synchronous scheduling; another block size: make-room's token counts round to it | |
| Engine starts | | a start failed (warm.json keeps its log lines) |
| GPU exclusive (`--gpu`) | | another process holds the GPU (`deploy/bin/gpu-guard`) |
| MIG | one instance: a smaller card | MIG on with no instance or several |
| systemd `--user` and lingering | none, or `LH_SUPERVISOR=lh-svc`: `deploy/bin/lh-svc` runs the units; nothing changes in the measurements | |
| User namespaces | the kernel refuses them, under `lh-svc` (no user manager, or `LH_SUPERVISOR=lh-svc`): the voice plane in the host's network namespace, with the host's public addresses named (they must drop inbound UDP). The listener audit cannot attribute listeners to units under `lh-svc` (their processes' cgroups are the session's), so this path's check is the named addresses, not the audit | the kernel refuses them under systemd, where the voice plane's units would listen in the host's namespace, which `smoke.sh`'s listener audit refuses: allow unprivileged user namespaces first (Ubuntu 24.04: `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`), or, with no sudo, set `LH_SUPERVISOR=lh-svc` |
| Listener audit | | no `ss` (iproute2), which `smoke.sh`'s audit ([`deploy/bin/audit-listeners`](../deploy/bin/audit-listeners)) reads; `bootstrap.sh` installs it where sudo and apt work |
| Cores | below 17: the warning of R7 | |
| Control plane, models, corpus and callers, binaries, disk | | down or missing |

## Single-host mode

A host off Penn's network has no cyan beside it, so it runs both: the GPU servers from its node directory and cyan's control plane, on the host's own Unix sockets and loopback, each on its core plan's cores. What differs from the box:

- **No tunnel.** The GPU servers serve on their own sockets, as the box's do (`$XDG_RUNTIME_DIR/lh/engine.sock`, `stt.sock` and `tts.sock`), and the control plane reaches them there: the gateway's upstream is the engine's socket, and a drop-in gives the voice worker the STT's and the TTS's, where cyan has the tunnel's `phastform-stt.sock` and `phastform-tts.sock`. No tunnel unit (`lh-tunnel@`) is linked; the units that want one start without it, as systemd starts them. E2's agents reach the engine through the gateway's TCP listener, as on cyan. AIPerf and `vllm bench serve` speak TCP only, so the calibration point and the batch arm, which go straight to the engine, reach it through [`deploy/bin/lh-forward`](../deploy/bin/lh-forward), from a loopback port to the engine's socket, on the load generator's cores, as the box's calibration points reached their engine through the tunnel's TCP forward.
- **No systemd user manager** (a container, or no lingering), or `LH_SUPERVISOR=lh-svc` where there is one: [`deploy/bin/lh-svc`](../deploy/bin/lh-svc), a supervisor in Python's standard library, runs the same unit files and drop-ins, and `deploy/bin/shims/systemctl` and `journalctl`, first on `PATH` (`~/leaseholder/node/env.sh`, which `deploy/a100/host.py env` writes and which keeps the override for the later stages and every script that sources it; `LH_SUPERVISOR=auto` gives the choice back to the host's facts), answer what `variant.sh`, `campaign.sh` and `lh-bench e2 run` ask. A `Type=notify` unit (the gateway) has started at its `READY=1`, a socket unit (`lh-gateway.socket`) starts and stops as a target does while the gateway binds its own listeners, and a wanted unit that does not exist is left out, as systemd leaves it. A stop returns once every process it stops has exited, a target's units included. Its units' processes carry `LH_SVC_UNIT`, by which `gpu-guard` tells them from a foreign process, as it tells systemd's units by their cgroups.
- **No user namespaces** (Ubuntu 24.04 restricts them): under `lh-svc` the voice plane runs in the host's network namespace (`LH_VOICENET=host`); LiveKit's own candidates stay on 127.0.0.1. Under systemd the preflight refuses it, since the listener audit of `smoke.sh` (block 0) refuses the voice plane's units' listeners outside their namespace: allow unprivileged user namespaces first, or, on a host with no sudo, run the units under `lh-svc` (`LH_SUPERVISOR=lh-svc bootstrap.sh stage`), where the preflight reports the voice plane in the host's namespace as degraded. Under `lh-svc` the audit cannot attribute listeners to units (their processes' cgroups are the session's), so that path's check is the preflight's named public addresses, which must drop inbound UDP, not the audit.
- **The measurement drop-in** ([`deploy/phastform/measure`](../deploy/phastform/measure)): the A100 run reads vLLM's development routes, as the box's measurements do (`/server_info`, `/reset_prefix_cache`): `campaign.sh` turns the drop-in on before each point, and `host.py warm` before its starts.
- **The campaign runs on the host** (`benchmarks/e2/campaign.sh --local`), so a dropped workstation link stops nothing, and [`deploy/a100/pull.sh`](../deploy/a100/pull.sh) copies each point, the probes, the node's records and the warm starts' to the workstation every two minutes.

None of these differs between the frozen engine and make-room on one host, so a single host's comparisons are made within it.

## Running on another GPU, step by step

From the workstation, with the repository at the revision to run and the corpora and the callers' audio (`weka-x32`, `weka-x32-long`, `callers`) in `LH_DATA`:

```sh
deploy/a100/push.sh USER@HOST            # the source tree (with DEPLOYED), the data; LH_GATEWAY_BIN to bring an lh-gateway
```

On the host, from the home directory:

```sh
bash leaseholder/src/deploy/a100/bootstrap.sh stage   # no GPU needed: before the billed clock if the host is up earlier
bash leaseholder/src/deploy/a100/bootstrap.sh gpu     # with the card: the driver, the install path, vLLM's parse, the preflight
bash leaseholder/src/deploy/a100/bootstrap.sh warm    # one start per compile configuration, so the campaign's are warm
nohup setsid bash leaseholder/src/benchmarks/a100/run.sh --until HH:MM \
  >>leaseholder/runs/a100/run.log 2>&1 </dev/null &  # the A100 run's core, block by block
```

On a host whose user manager would run the voice plane outside its namespace (the preflight refuses it) and that has no sudo to allow user namespaces, `LH_SUPERVISOR=lh-svc bash leaseholder/src/deploy/a100/bootstrap.sh stage` runs the units under `lh-svc` instead; `env.sh` keeps the choice for `gpu`, `warm` and the run. `stage` takes uv and Python, renders the node into `~/leaseholder/node` (from the profile's own card when the GPU is not yet visible), builds the engine, TTS, voice and bench venvs (the engine's by R8), fetches Kokoro and the models at their pinned revisions (`LH_MODELS=MODEL@REVISION` adds one, the 14B for C'), installs LiveKit, builds `lh-gateway` when a copied binary does not run, links the units with this host's drop-ins, writes the control plane's configuration (`deploy/a100/host.py config`: new keys, gateway.toml from cyan's example, `e2-host.toml` for `lh-bench e2 run`), starts the control plane, and builds FlashInfer's sampler for the card's architecture into the engine's own workspace (`host.py jit-check`), which checks the compiler before a GPU minute is spent. On the workstation meanwhile: `deploy/a100/pull.sh USER@HOST`, and after: `make reproduce-a100 RECORDS=~/leaseholder/hosts/HOST CORPUS=<weka-x32>`.

## What to ask for at handover

The preflight finds the host's facts, but these decide the plan, and some need the provider:

1. The card and its form factor (A100 40 GB SXM4 or PCIe), how many, and whether MIG is off.
2. The driver version (R580 or newer runs the box's lock as it is; R535 or R570 on a data-center card with `cuda-compat-13-0` installed, or sudo to install it; else the cu129 lock), and any CUDA toolkit installed (13.0 or 12.9 saves the toolkit download).
3. Whether the host is reachable before the billed clock, for `stage` and `warm` (block 0 is 40 minutes without them, about 15 fewer with them [estimate]).
4. Whether the home directory is on a volume that outlives the host's restarts: the compile caches and every record live there until `pull.sh` copies them.
5. The OS image and whether there is sudo (build-essential, `cuda-compat-13-0`, lingering), a container (its `/dev/shm`: Docker's default 64 MB holds no CPU tier; `--shm-size` or `--ipc=host`), systemd with lingering, and whether unprivileged user namespaces are allowed: under systemd the voice plane needs its own namespace (Ubuntu 24.04 restricts them unless sudo lifts it); without sudo, `LH_SUPERVISOR=lh-svc` runs the units under `lh-svc` instead, degraded.
6. CPUs and their NUMA nodes (34 give the core plan its wants, 17 is the floor), RAM (the tier is 20 GiB for the core, 71 GiB for C' by R3), and free disk: the engine venv takes 6.1 GB on the box's lock and 12 GB on the cu129 lock, the CUDA 12.9 toolkit 9.4 GB (measured on a CPU host), the models about 7 GB and 16 GB more for C' [estimate].
7. Egress to PyPI, download.pytorch.org, GitHub, Hugging Face and NVIDIA's download site, for staging.
8. SSH access (user, key) and whether the host has a public address and a firewall that drops inbound UDP (the voice plane without its own namespace).
9. The billed window's length: the core takes about 4.6 hours of it [estimate].

## The A100 run

The A100 40 GB run asks whether make-room's result carries to a datacenter card. Its core is like-for-like with the box: the box's model at its revision, its 5 GiB pool, 512/256, the 20 GiB CPU tier, 24 sessions of `weka-x32-long`, two callers, W5's three arm files and five rounds at seeds 11 to 15; only the card, its precision path and the host change. So a difference within the A100 is the engine change's, and a difference from the box is the card's. [The forecast](../benchmarks/a100/forecast.md) gives every term the run measures, modeled: the KV case's stock wait 1,636–1,688 ms, make-room's first token 134–158 ms, a running share of 13% in the core scenario against the box's 17%, 76% busy against the box's 100%.

### The core, block by block

[`benchmarks/a100/run.sh`](../benchmarks/a100/run.sh) runs it on the host; each block is complete when it ends and is skipped once done, so a stop anywhere leaves whole comparisons and a rerun resumes. Durations are the box's measured ones where they exist and estimates otherwise:

| Block | Work | Minutes (estimate) | Complete after it |
|---|---|---:|---|
| 0 | `bootstrap.sh gpu` and `warm` (the first start with empty caches, about 9 minutes on the box's kernels, measured; each other compile configuration 2 to 4 minutes, estimate), the box's `smoke.sh` from the source tree (its listener audit included), one 2-turn call through LiveKit | 40 | the deployment, every compile configuration warm |
| 1 | the frozen engine: E0 (`lh-bench tiers --reps 3`), the KV case x3, the prefill case x3, the calls' tail x2 with whole 40-token replies ([`probes.sh`](../benchmarks/a100/probes.sh) 1) | 25 | the frozen side of the probes, E0's cells |
| 2 | stock at vLLM's default, 2,048/0: KV x3, tail x2; stock at the time-equivalent cap (`--cap`, 1,536): KV x1, tail x2 | 26 | the stock step-cap settings |
| 3 | the PRs' values (make-room, no reserve, a 48-token window): KV x3, prefill x3, tail x2 | 17 | the probe comparison on the A100 |
| 4 | E2 round 1 ([`a100-r1.campaign`](../benchmarks/a100/a100-r1.campaign)): the PRs' values, voice alone, frozen | 29 | one E2 round |
| 5 | the calibration point ([`calibration.sh`](../benchmarks/a100/calibration.sh)): AIPerf's replay of the whole corpus at 16 sessions straight to a fresh frozen engine (through `lh-forward`'s loopback port), 600 s + 60 s | 17 | the step model's data |
| 6-9 | rounds 2 to 5, the engine groups alternating so that no switch falls between rounds | 4 x 28 | three rounds by about 3.5 h, five by about 4.4 h |
| 10 | the GPU units stopped; the last pull | 10 | the core, about 4.6 h |

During every agent point `lh-bench e2 run` samples the engine's `/metrics` every 0.5 s, from which each point's summary gives the measured running share (the time free KV is under two calls' 2,048 tokens).

### The optional arms

In order, each complete in itself, after the core (`run.sh NAME`):

1. **C'** ([`cprime.sh`](../benchmarks/a100/cprime.sh)): Qwen3-14B-FP8 at the card's own pool, 117,344 tokens (modeled), at 40 sessions, on its own node (`a100-40gb-qwen3-14b`), the agent's requests naming it (the `qwen3-14b` worker variant): its first starts, the pool checked, a smoke reply without `<think>`, E0 with one rep, its own voice alone, then three rounds of tuned stock (the stock step cap with the lowest calls' first-token p95 in blocks 1 and 2, [`probe_results.py`](../benchmarks/a100/probe_results.py) `pick-stock`) against the PRs' values. About 95 minutes.
2. **The PRs' own code** ([`branches.sh`](../benchmarks/a100/branches.sh)): vLLM at the PRs' tip with their switches on against vLLM at their base, two Python-only installs over precompiled binaries staged beforehand (`branches.sh stage BASE HEAD`), each through the probes: KV x3, prefill x2, tail x2. About 45 minutes.
3. **Preempt-resubmit** on the stock engine (`a100-resubmit-r3` to `-r5`): three points, seeds 13 to 15. About 36 minutes.
4. **The scheduler at a datacenter batch** ([`batch.sh`](../benchmarks/a100/batch.sh)): `vllm bench serve` keeping 256 requests in flight, 10% at priority 0, on frozen and on make-room with its switches off and on. About 20 minutes.
5. **A'** (`a100-aprime-r1`, `-r2`): the box's pool at 32 sessions, where the forecast places the A100 closest to the box's regime (a running share of 18%). About 80 minutes.

MIG (a 3g.20gb voice instance beside a 4g.20gb agent instance) needs administrator rights and a GPU reset and is not scripted.

### The analysis

`make reproduce-a100 RECORDS=<the pulled host directory> CORPUS=<weka-x32>` writes `build/reproduce-a100/a100.md` ([`analysis.py`](../benchmarks/a100/analysis.py)), each section from what the records hold: the step model fitted on the calibration point with the card's own E0, each term beside the forecast as a ratio; lh-sim with those costs against the measured point, as the box's calibration table; the probes beside the forecast and the box; `lh-bench e2 report` over the A100's points alone, with each configuration's measured running share; and the two cards in regime coordinates only. `make check-a100-calibration EXPORTS=<the v0.3 calibration exports> CORPUS=<weka-x32>` runs its calibration code on the box's three W2 points with the box's E0, and checks that it gives the box's own table rows.

## The Marlin dry run on the box

Two sessions on Phastform test the A100's code path before the A100: the FP8 checkpoint through `MarlinFP8ScaledMMLinearKernel` (W8A16), FlashAttention 2 on LBHNC, the CPU tier, make-room and the harness, on a real GPU. They measure function, not the A100's magnitudes: the box runs Marlin and BF16 at 47.4 TFLOPS with FP32 accumulation, so its linear layers on Marlin take at least about 153 µs a token against DeepGEMM's fitted 48 [derived].

- [`marlin-session1.sh`](../benchmarks/a100/marlin-session1.sh), about 25 minutes of GPU time: E0 on the frozen engine (DeepGEMM), then on Marlin (`--linear-backend marlin`, checked by the engine log and `/server_info`, with `smoke.sh`), then make-room's KV and prefill cases and the tail on Marlin, and the frozen engine back. `--fallback` selects Marlin by disabling the four block-FP8 kernels ahead of it (`VLLM_DISABLED_KERNELS`) instead of the flag.
- [`marlin-session2.sh`](../benchmarks/a100/marlin-session2.sh), about 30 minutes: [`bench_marlin.py`](../benchmarks/a100/bench_marlin.py) (Marlin, BF16 and DeepGEMM at Qwen3-4B's four linear shapes, M 1 to 1,024: Marlin's factor over BF16 on the box, one data point beside PR #61224's 1.42 on an L4), the probes at a 3 GiB pool and a 16,384-token context (which size themselves from it), and one E2 point on Marlin ([`marlin-dry.campaign`](../benchmarks/a100/marlin-dry.campaign)), without the KV sampler, as W5's points it is read against ran.

Both run from the workstation with `LH_SSH`, `LH_GPU` and `LH_CYAN` as for `campaign.sh`, keep each step's outputs on its host, skip a step done before, and copy the outputs to `LH_OUT` (`~/leaseholder/hosts/marlin-dry`).
