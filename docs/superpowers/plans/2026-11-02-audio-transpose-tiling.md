# Audio Transpose Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved generic phase-tiled audio transpose cost model without measured-time fitting.

**Architecture:** A cached, pure shape model computes useful work, padded work, panel payload and critical-core wave work. A decode subclass preserves physical boundaries and activates the model only for Ascend910B3; a minimal optional Kernel scheduling hook keeps runtime and CSV compute timing consistent.

**Tech Stack:** Python, dataclasses, integer interval arithmetic, existing RoofLang DES, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-02-audio-transpose-tiling-design.md` (approved by user).

## Global Constraints

- Fixed tiles: Mt=16, Nt=16, Kt=8. No measured-time inputs, fitted constants, new efficiencies or serial spatial restriction.
- Preserve original reference-FLOP convention, 4590 leaf boundaries, tensor layouts and non-transpose paths.
- Report actual executed work separately from critical-path wave work; repeated traffic never changes allocation.
- Initially activate only for Ascend910B3. Read core count from its existing hardware metadata.
- Keep old reports and experiment outputs intact; compare at compute=0.65 and operator HBM=0.60.

## Review Focus

- Padding/dilation/stride with empty phases: no invalid reads; output and bias still accounted (task 1).
- Partial M/N tiles across core waves: use max actual bias work in each wave (task 1).
- Very long lengths: closed interval sums and periodic run processing, not per-element expansion (task 1).
- Repeated filter/bias reads: preserve weight/activation capacity proxies (task 2).
- Ordinary kernels and unsupported hardware: unchanged costs, clear fallback metadata (task 2).

### Task 1: Pure geometry and execution model

**Files:** Create `src/rooflang/programs/models/minimax_h3/transpose_tiling.py` and `tests/test_minimax_h3_transpose_tiling.py`.

**Interfaces:** Frozen `TransposeGeometry(batch, cin, cout, length, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True)`; `transpose_cost(geometry, core_count)` returns an immutable cost with phase records, work/bytes, `compute_time_us(tflops)` and `to_dict()`.

- [x] Write independent scalar/scatter and panel-enumeration tests for valid contributions, work and bytes; core-count boundaries 19/20/21; empty phases, cropped edges and invalid geometry.
- [x] Run `.venv/bin/python -m pytest tests/test_minimax_h3_transpose_tiling.py -q`; require missing-module failure before implementation.
- [x] Implement phase geometry, fixed tile counts, clipped interval sums for payload and compressed bias-wave scheduling. Preserve exact integers until conversion to time.
- [x] Verify oracle tests and long-length tests pass; document index order `(batch, group, M tile, N tile)` and tap-major/channel-minor K flattening.

### Task 2: Decode/runtime integration

**Files:** Modify `audio_decode.py`, `bridge.py`, `language/kernels/kernel.py`, `runtime/simulator.py`; add `audio_transpose.py` and integration tests in `tests/test_minimax_h3_transpose_tiling.py`.

**Interfaces:** `AudioTransposeOperator` extends `DecodeOperator`, retains reference work before/after `enable_tiling(core_count)`. `Kernel.scheduled_compute_time_us(device)` returns `None` by default; H3Kernel uses the active cost schedule. Simulator validates optional results and otherwise follows its current FLOP path.

- [x] Add failing tests for B3 activation, unsupported-hardware fallback, capacity/traffic separation, bias Vector FLOPs, CSV timing and ordinary-kernel behavior.
- [x] Add the subclass and select it only for transpose calls; bind hardware in the bridge, preserve measured tensor descriptors and export audit metadata.
- [x] Keep memory payload in zero-storage traffic descriptors, physical weight residency unchanged, and preserve original activation-peak proxy.
- [x] Run all H3/decode, Ascend preset and runtime tests. Run a wider suite once integration checks pass.

### Task 3: Reproducible fixed-config comparison and review

**Files:** Add a narrow comparison script and new output directory; update design status and this plan.

- [x] Read the report manifest and pinned experiment scenario. Build audio-only new costs with the original runtime configuration and compare the pinned old CSV; retain baseline source checksums.
- [x] Export new audio CSV, per-shape transpose deltas and a short Markdown result explaining executed work/traffic and remaining error. Do not update old reports or infer passing accuracy thresholds.
- [x] Verify the 4456 non-transpose rows are identical to the pinned baseline in relevant work/traffic/time fields and all 4590 physical boundaries/counts are preserved.
- [x] Run a fresh independent code review, resolve findings, perform final verification and deliver the scoped local commit on the current feature branch. No push.

## Execution record

- User approved implementation in the current chat; execute inline without requesting another plan approval.
- Existing clean feature branch: `codex/multimodal-sim`, design commit `ea41292`. Work in place; no additional worktree or branch is necessary for this scoped task.
- The builder may select equivalent closed-form counting algorithms and names, but must preserve the approved semantics and keep data evidence separate from prediction.
- Task 1: missing-module RED observed; independent scatter/panel oracle suite passed after implementation. Interval sums avoid output enumeration; cycle skipping handles long bias schedules.
- Task 2: missing integration/hook RED observed, then 58 combined transpose/audio/decode tests passed. Prediction succeeds with both builtins.open and Path.open blocked. CSV timing uses simulated trace entries from the common runtime hook.
- Task 3: 404 targeted tests passed, 1 skipped; first full suite 857 passed, 1 skipped. A further large-bias regression was added before final verification.
- Fixed-config result: audio 94.82140691442427 -> 101.28338557724551 ms; transpose 4.204111570512707 -> 10.666090233333604 ms; 4456 other rows unchanged. Artifacts: `outputs/audio_transpose_tiling_20261102_verified/`.
- Evidence integrity: all 11 original experiment files and measured stage summary match `reports/manifest.json` SHA256 values. Reports/measured files were not modified.
- Scope ruling: the generic model still underestimates the real transpose kernel sum (867.372689 ms) and overestimates some dense shapes; retain both results without fitting. It is not a CANN reconstruction or NPU validation.
- Final review: fresh reviewer found no important issue; 400 independent JS scatter/panel cases and 250 larger bias-wave cases matched. Minor issue fixed: comparison metadata now describes the actual 0.65 compute configuration instead of inheriting preset text saying 0.70; no prediction change.
- Final verification: `.venv/bin/python -m pytest -q` -> 858 passed, 1 skipped; `git diff --check` clean. Preserve the current feature branch and workspace; no merge, push or PR is part of this task.
