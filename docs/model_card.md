# Model Card — BioTwin.X CMCL Risk Engine

*Version 0.1.0 · October 2026 · Team BioTwin.X, VIT Vellore*

This model card follows the structure proposed by Mitchell et al. (2019) and
is written to be read alongside `README.md` and `data/artifacts/`.
It describes **what the model does, what it was trained on, how it was
evaluated, and where it must not be used.**

---

## 1. Model details

| Field | Value |
| --- | --- |
| Name | BioTwin.X Cross-Modal Contrastive (CMCL) risk engine |
| Version | 0.1.0 |
| Type | Dual-encoder transformer + FiLM fusion + supervised heads |
| Task | 12-hour-ahead prediction of silent ischaemia, glycaemic crash, autonomic arrhythmia; ordinal triage score |
| Inputs | 12-channel, 5-minute telemetry grid (12 h look-back) + 44-dimensional static clinical/genomic vector |
| Outputs | 3 calibrated event probabilities, any-event probability, time-to-event estimate (hours), Clinical Risk Index 0–100 |
| Framework | PyTorch 2.x (CPU), scikit-learn for the surrogate and metrics |
| License | MIT |
| Owner | Team BioTwin.X (Anika Kumar) |
| Contact | anikak206@gmail.com |

### Architecture in one paragraph

A PatchTST-style temporal encoder (patch 12 bins / stride 6, 3 transformer
layers, 8 heads, d=128) reads the 12-channel telemetry grid. A low-capacity
static encoder (LayerNorm, d=128, archetype dropout 0.15) reads age, sex, BMI,
lipids, HbA1c, blood pressure, medication flags, 16 SNP dosages, PRS z-scores
and derived clinical interactions. Fusion is **FiLM-first**:
`h = γ(z_s) ⊙ (z_t · gate) + β(z_s)`, followed by cross-attention (temporal
queries attending to the archetype) and a conjunction term
`[h, z_s ⊙ z_t, |z_s − z_t|]`. Three NT-Xent contrastive heads (fused,
static-only, temporal-only) align the modalities by patient identity; the
supervised heads sit on the fused representation.

### Why FiLM rather than concatenation

With ten to a hundred patients, a concatenated static vector is a free
patient-identity channel: the head can memorise "patient #37 had an event"
without learning any physiology. This was observed directly during
development (training AUC 1.000 by epoch 4 while validation collapsed). Under
FiLM the static branch can only *scale and shift* the temporal code, so every
path to the output still carries window-level physiology. `archetype_dropout`
randomly replaces the archetype with a learned null vector during training,
which further prevents the static branch from becoming a lookup table.

---

## 2. Intended use

**Intended**
* Research on multi-modal cardiometabolic early-warning systems.
* Demonstration of cross-modal contrastive fusion for asynchronous wearable
  streams.
* A clinician-facing triage console where the model *prioritises* patients for
  review and explains its evidence.

**Not intended**
* Diagnosis, treatment selection, or any autonomous clinical action.
* Triage in an emergency department or any setting where a missed event is
  immediately life-threatening.
* Use on real patient data without external validation, regulatory review and
  a prospective study.
* Inferring risk for populations not represented in the synthetic cohort
  (the generator's demographics are deliberately narrow).

**Human oversight.** Every alert is accompanied by an evidence chain and a
recommended next step, but the clinician remains the decision-maker. The
console never blocks, overrides or auto-escalates care.

---

## 3. Training data

All data is **synthetic**, produced by `synthetic_generator.py`.

| Property | Value |
| --- | --- |
| Cohort profiles | `demo` 20×7 d · `standard` 100×14 d · `research` 240×30 d · `smoke` 6×2 d |
| Headline evaluation | `standard` (100 patients, 104 events: 42 ischaemia, 34 crash, 28 arrhythmia) |
| Telemetry modalities | ECG R-R, PPG HR, PPG RMSSD, CGM glucose, SpO₂, skin temperature, sleep stage |
| Cadence | 1 s / 60 s / 300 s / 30 s depending on modality (deliberately asynchronous) |
| Static inputs | demographics, vitals, lipids, glycaemia, medications, family history, 16-locus SNP panel |
| Labels | Event onset timestamps; a window is positive if an event falls within the next 12 h |
| Generator model | Circadian + ultradian physiology, HRV spectral synthesis, glucose–insulin dynamics, event precursor signatures (autonomic withdrawal, glucose drift, ectopy), recurrent events with per-type minimum spacing, contextual sensor noise and dropouts |
| Splits | **Patient-level** k-fold. Windows from one patient never appear in both train and validation. |

The generator writes `data/artifacts/dataset_manifest.json` with event counts
and per-patient paths. The cohort is reproducible from the seed alone.

### Known data limitations

* The generator encodes our own hypotheses about prodromal physiology. A model
  that learns them is learning *our assumptions*, not human biology.
* Event prevalence (~1–2 % of windows) is far higher than in a real screened
  population, which flatters precision and F1.
* Sensor artefacts are simulated with simple statistical models, not from
  device recordings.
* No comorbidity interactions beyond those explicitly parameterised.

---

## 4. Evaluation

### Protocol

* Patient-level k-fold cross-validation (k=3 for `demo`, k=5 for `standard`).
* Metrics: per-class ROC-AUC, average precision (AP), F1 at the
  Youden-optimal threshold, any-event AUC, ECE (10 bins) and Brier score.
* Calibration fitted on out-of-fold predictions only.
* Latency measured end-to-end (window assembly + forward pass + explanation +
  action engine), not just the tensor forward pass.

### Measured results

The authoritative, machine-readable numbers are written by the pipeline to
`data/artifacts/training_report_<profile>.json`. The README §6.5 table is
generated from the same file.

> **Do not quote a metric that is not in that file.** The challenge targets
> (ROC-AUC ≥ 0.94, F1 ≥ 0.91, p95 latency < 120 ms) are *targets*, not
> results. Where the measured value falls short, the README says so.

### Ablations

The pipeline runs, with an identical training budget:

1. full CMCL fusion;
2. temporal encoder only;
3. static/genomic branch only (telemetry-derived context zeroed);
4. XGBoost over static features only (EHR-equivalent baseline);
5. XGBoost over static + engineered window summaries.

The ablations exist to answer one question: **does the fusion add anything
over the best tabular baseline?** If it does not, that is the honest finding
and it is reported.

---

## 5. Limitations and failure modes

| Failure mode | Mechanism | Mitigation |
| --- | --- | --- |
| Patient-identity memorisation | Static vector constant per patient | FiLM fusion, archetype dropout, patient-level splits |
| Uncalibrated confidence | Class-balanced oversampling inflates scores | Temperature scaling on out-of-fold predictions |
| Modality dropout | A device stops streaming | Explicit staleness channels + coverage attenuation of the risk index |
| Distribution shift | Real physiology ≠ generator | Not solved. Requires external validation. |
| Label ambiguity | Silent events are, by definition, often unrecorded | Labels come from generator ground truth; real silent events may never be captured |
| Small cohorts | 20-patient demo cannot support any AUC claim | `standard` is the minimum reported cohort; `demo` is for UI only |
| Threshold brittleness | F1 depends on the operating point | Report the threshold alongside F1 |
| Fairness | Generator demographics are narrow | Not evaluated; must be before any deployment |

---

## 6. Ethical and regulatory considerations

* **Not a medical device.** No CE mark, no FDA clearance, no CDSCO approval.
* **No real patient data** was used, so no consent or DPDP Act obligations
  attach to this version; any real deployment would require a DPIA, consent
  management and audit logging.
* **Explainability is partial.** Attributions are analytic/rule-based
  approximations of a black-box fusion model, useful for communication but not
  a proof of causation.
* **Automation bias** is a real risk in triage UIs. The console deliberately
  separates "evidence" from "recommendation" and keeps a human-readable
  rationale on every card.
* **Genetic data** is sensitive personal data. The demo uses simulated
  genotypes; a real system must encrypt at rest, restrict access and provide
  genetic counselling pathways.

---

## 7. Reproduction

```bash
python synthetic_generator.py --profile standard      # cohort + telemetry
python inference_engine.py bootstrap --profile standard
python scripts/train_pipeline.py --profile standard --folds 5 --epochs 24
```

Artifacts:

* `data/models/biotwin_cmcl_<profile>.pt` — weights, scalers, calibrators,
  metrics, config.
* `data/artifacts/training_report_<profile>.json` — metrics, ablations,
  baselines.
* `data/features/` — memoised grids and window tensors.

---

## 8. Citation

```bibtex
@misc{biotwinx2026,
  title  = {BioTwin.X: Cross-Modal Contrastive Digital Twin for Predictive
            Cardiometabolic Prognosis},
  author = {Kumar, Anika and Team BioTwin.X},
  year   = {2026},
  note   = {Happiest Health Digital Twin Challenge 2026. Research prototype,
            synthetic data only.},
  url    = {https://github.com/anikak206/BioTwin.X}
}
```
