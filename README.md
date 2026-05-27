<!-- README STUB — all quantitative claims are written ONLY at S6/S7 from bench_results/. Placeholder marker: MEASURED@S6 -->

# vlatrust

**Calibration-under-shift trust harness for Vision-Language-Action (VLA) policies.**

A VLA policy can score ~97% on its in-distribution benchmark and then drop to
near-0% the moment the scene, lighting, instruction phrasing, or initial state
shifts — *while remaining just as confident*. `vlatrust` measures exactly that
failure mode: **does a policy's confidence degrade in step with its competence
as input distribution shift rises?**

It does this over **recorded action traces** (no GPU, no simulator required for
the core), emitting:

- a **conformal-guaranteed abstention gate** (distribution-free finite-sample coverage),
- a **Reliability-Shift / collapse curve** (success rate vs. perturbation intensity, per modality),
- a single **Trust-Shift score** that is high only when the policy *flags its own* degradation.

> Status: **pre-alpha (v0.1.0a1)**. See "Scope" below for exactly what is and is
> not claimed.

## The one claim (falsifiable)

> For VLA policies that expose token-level confidence (Tier-A, e.g. OpenVLA),
> a well-calibrated policy earns a **high** Trust-Shift score and a
> confidently-wrong policy earns a **low** one.

Falsification test (run in CI): a known 90%→0% "confidently collapsing" policy
fixture **must** receive a low Trust-Shift score, and an abstention-enabled
variant **must** outscore its abstention-disabled twin. If not, the metric is
falsified.

## Why calibration, not just perturbation

Perturbation-robustness benchmarking for VLAs already exists (e.g. RobustVLA /
LIBERO-plus). `vlatrust`'s contribution is the **cross-model calibration-under-shift**
layer on top: conformal coverage, a reliability gap (sim-vs-real, calibrated-vs-actual),
and a fail-closed out-of-distribution action gate — bundled as one released harness.
<!-- prior-art differentiation expanded at S7 (G2/G3/G5) -->

## Tiers (what confidence means per policy family)

| Tier | Policy family | Confidence source | Trust-Shift CLAIM |
|------|---------------|-------------------|-------------------|
| A | token/autoregressive (OpenVLA) | token entropy (native) | full claim |
| B | flow-matching (π0, SmolVLA, GR00T) | sampling variance (opt-in, GPU) | NON-claim (v0.1.1) |
| — | no exposable confidence | `ConfidenceSource.NONE` | abstention axis returns `N/A` (fail-closed) |

## Install

```bash
pip install vlatrust                 # core: recorded-trace path, numpy only
pip install "vlatrust[openvla]"      # Tier-A token-confidence backend (falsification fixture)
pip install "vlatrust[lerobot]"      # ingest LeRobot datasets
```

## Quickstart

<!-- Quickstart commands are validated at S5; any numeric output shown here is added at S7 and marked `# illustrative` unless it is a measured bench_results value. -->

```bash
vlatrust doctor                      # report which backends are live vs. mock
vlatrust score   <trace.json>        # full scorecard (JSON + self-contained HTML)
vlatrust calibrate <trace.json>      # calibration report only
```

## Scope

- **In (v0.1.0a1):** TraceSet core; 14 post-hoc perturbation dims (own injector);
  `ConfidenceSource` enum + OpenVLA Tier-A adapter; spike-preserving sequence-level
  conformal (+ Mondrian, + weighted); reliability gap (Δ_succ, Δ_cov); fail-closed
  OOD action gate; collapse curve + fragility; PAVA / inverse-Brier / multiplicative
  gate / bootstrap CI; self-contained HTML scorecard; CPU-only tests.
- **Deferred (v0.1.1):** renderer-heavy 3 perturbation dims (GPU); flow-matching
  sampling-variance adapter (Tier-B); live π0 / GR00T inference; live sim integration.
- **Deferred (v0.2):** sim→real gap on real-robot traces; GR00T backend;
  evolutionary score-hardening.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). vlatrust bundles no
third-party model code or weights, and does **not** depend on LIBERO-plus
(which ships no license).
