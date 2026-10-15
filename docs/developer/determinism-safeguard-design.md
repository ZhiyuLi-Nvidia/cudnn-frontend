# Determinism safeguard for FE kernels (design)

Status: proposal, aligned with a Codex review (3 rounds). Modeled on Megatron-LM's
`tests/unit_tests/determinism/` and `tools/check_kernel_determinism_coverage.py`.

## Goal

1. Every FE GPU kernel family has a bit-exact run-to-run test, or a reasoned exemption.
2. A PR that adds or changes a kernel must touch that kernel's own test.
3. Long term: burn down the untested-kernel count and grow the set declared deterministic.

```
PR diff -> check.py (CPU) -> manifest.py <- registered tests (GPU, marker: determinism)
              |                                   |
              +-- fail: unregistered / baseline   +-- harness: run N x, compare bytes
                  touched / test not updated
```

## Pieces

| Piece | Path | Notes |
|---|---|---|
| Manifest | `tools/determinism/manifest.py` | stdlib only; never loads conftest, `cudnn` or CUDA |
| Checker | `tools/determinism/check.py` | CPU job; `--selftest` covers every rule |
| Harness | `test/python/determinism_harness.py` | `assert_replays_bit_exact(run, replays=5, contend=True)` |
| Tests | next to each op's tests | marker `determinism` and `L0`; inherit the op's conftest |
| CI | `.github/workflows/determinism-gate.yml` | `pull_request`, `fetch-depth: 0`, reruns on label changes |
| Docs | PR template, `python/cudnn/AGENTS.md` rule, `test/AGENTS.md` how-to | label `determinism-exempt` |

### Manifest entry

`KernelEntry(name, sources, tests, guarantee, scope, status, exempt_reason)`

- `status`: `covered` (has tests) | `baseline` (exists today, untriaged) | `exempt` (reason required).
- `guarantee`: `deterministic` | `flag` (needs e.g. `use_deterministic_algorithm=True`) | `nondeterministic`; `null` only while `baseline`.
- `scope`: engine, arch, flag, dtype the guarantee holds for.
- One entry per kernel family. A test counts only for the entry that lists it.
- Baseline is seeded once by a script (about 280 files match `@cute.kernel`), grouped by directory. New baseline entries are rejected.
- Existing bitwise tests are registered on day one, e.g. `test_two_launches_are_bitwise_and_race_free` in `test/python/sdpa/frost/test_sdpa_bwd_d256_sm100.py`.

### Checker rules (files changed vs merge-base)

1. A kernel-bearing file not matched by any entry fails. Detector: kernel directories, `@cute.kernel`, `@cute.jit`, `@ct.kernel`, atomics, `__global__`, and an explicit list of code generators (`gemm/frost/sm100|sm120` `epilogue_codegen.py`, `kernel_templates/`). Comments and strings are ignored.
2. Touching a `baseline` file fails unless the PR splits out its family as `covered` or `exempt`. Whole-directory promotion is rejected.
3. Touching a `covered` file requires one of that entry's tests to change, unless the PR has `determinism-exempt`.
4. Integrity: unique names, paths exist, valid field combinations, and no per-source-file coverage downgrade vs base (globs expanded).
5. Test IDs: CPU checks the function name by AST; a GPU step diffs `pytest --collect-only -m determinism` against the manifest.

### Harness contract

- `run()` executes the real FE path and returns tensors; callers recreate RNG and state inside it.
- Synchronize the launch device or stream first, then clone, then compare bits (signed zero and NaN payload count). Empty results are rejected.
- Self-tests: a counter-driven control that must fail, and a run on an explicit non-current stream. An atomic float kernel is stress evidence only, never a required failure.

## Rollout

1. PR1: everything above, baseline seeded, existing bitwise tests registered. No kernel behavior change.
2. PR2: SDPA backward (dQ accumulation) with `use_deterministic_algorithm=True`. The `False` path is documented as `flag`; no promise it differs.
3. Then per-op burn-down. Metric: count of `status="baseline"`.

Maintainers make the gate a required status check in branch protection.

## Phase 2 (optional): shared helpers, by example

Deterministic kernels repeat a few mechanics. Share plain functions for them; no framework, no base class. Kernels import the helpers and the helpers never import kernels.

```
partials -> private buffer -> ordered handoff / reduce
(layout)                      (ordering)
```

| Mechanic | Example helper |
|---|---|
| Ordering: tiles update a shared output in a fixed order | `turn_wait(sem, idx, turn)` / `turn_release(sem, idx, turn)` |
| Independent buffers: partial results need their own space | declare regions `(name, dtype, shape, align, zero_init)`; get total bytes and views back |
| Eligibility: when a deterministic path is allowed | one predicate per rule, shared by the engine selector and the kernel API |

Each helper is adopted only where two or more kernels already duplicate it.

## Accepted risks

- Pattern detectors miss unusual kernel forms; review and the manifest self-checks are the backstop.
- FE cannot prove determinism of closed-source cuDNN backend kernels. It plumbs the flag and tests run-to-run on one GPU, never across lanes or SM counts.
- Bit-exact tests need a GPU lane; the checker does not.
