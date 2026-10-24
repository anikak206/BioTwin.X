"""
BioTwin.X — Genomic Layer
=========================
The static "genomic architecture" half of the digital twin.

This module owns:

* the curated cardiometabolic SNP panel (the *ground truth* effect weights used
  by the synthetic generator),
* polygenic risk score (PRS) computation with proper allele-frequency
  standardisation (z-scores),
* the allele-frequency normalisation and per-trait decomposition used by the
  CMCL static encoder.

Everything is deterministic and side-effect free so that the same patient
genotype always yields the same vulnerability vector.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Final, Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# SNP Panel
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SNPLocus:
    """A single genetic variant with its true (simulated) causal effect."""

    rsid: str
    gene: str
    chrom: str
    position: int
    risk_allele: str
    effect_weight: float   # log-odds contribution per risk allele
    trait: str             # "T2D" | "CVD" | "DYS" | "AUT"
    population_freq: float  # alternate (risk) allele frequency in the cohort

    def as_row(self) -> dict:
        return asdict(self)


#: Curated 16-locus cardiometabolic panel. Effect weights are calibrated so the
#: synthetic cohort spans roughly 0.6σ – 3.4σ of polygenic risk, which is the
#: dynamic range the cross-modal fusion module needs to be discriminative.
SNP_PANEL: Final[tuple[SNPLocus, ...]] = (
    SNPLocus("rs7903146",  "TCF7L2",  "10",  114758349, "T", 0.42, "T2D", 0.39),
    SNPLocus("rs9939609",  "FTO",    "16",   87378664, "A", 0.35, "T2D", 0.42),
    SNPLocus("rs2231142",  "GCK",    "7",   150634306, "C", 0.18, "T2D", 0.47),
    SNPLocus("rs1800574",  "HNF1A",  "12",   120977640, "C", 0.55, "T2D", 0.36),
    SNPLocus("rs757110",  "ABCC8",  "11",   2151748,   "C", 0.30, "T2D", 0.41),
    SNPLocus("rs1801282", "PPARG",  "3",    41361847,  "C", 0.22, "T2D", 0.45),
    SNPLocus("rs4922068", "KCNQ1",  "11",   2449730,   "C", 0.26, "T2D", 0.44),
    SNPLocus("rs1799854", "SLC30A8","8",    117172099, "C", 0.20, "T2D", 0.40),
    SNPLocus("rs1799983",  "NOS3",   "7",   150994267, "G", 0.24, "CVD", 0.38),
    SNPLocus("rs1800794",  "CETP",   "16",   56929912,  "C", 0.19, "DYS", 0.43),
    SNPLocus("rs4988235",  "PCSK9",  "1",    55039474,  "C", 0.33, "CVD", 0.31),
    SNPLocus("rs56266445", "LPA",    "6",    160541860, "T", 0.48, "CVD", 0.09),
    SNPLocus("rs12785878", "SH2B3",  "12",   111450442, "G", 0.16, "T2D", 0.37),
    SNPLocus("rs1260326",  "TCF7L2", "10",  114758349, "T", 0.21, "CVD", 0.36),
    SNPLocus("rs6496665",  "ABCC10", "4",    9475216,   "C", 0.14, "T2D", 0.30),
    SNPLocus("rs1110559",  "TLE4",   "19",   19187312,  "T", 0.17, "AUT", 0.28),
)

PANEL_BY_RSID: Final[dict[str, SNPLocus]] = {s.rsid: s for s in SNP_PANEL}
PANEL_GENES: Final[tuple[str, ...]] = tuple(dict.fromkeys(s.gene for s in SNP_PANEL))
TRAITS: Final[tuple[str, ...]] = ("T2D", "CVD", "DYS", "AUT")


# --------------------------------------------------------------------------- #
# Genotype simulation
# --------------------------------------------------------------------------- #
def simulate_genotypes(n_patients: int, seed: int) -> pd.DataFrame:
    """Draw Hardy-Weinberg genotypes for the full panel.

    Each locus dosage (0/1/2 risk alleles) is drawn from a binomial with the
    locus-specific alternate allele frequency, which preserves realistic
    heterozygosity and yields the correct polygenic architecture.
    """
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {}
    for locus in SNP_PANEL:
        p = float(np.clip(locus.population_freq, 1e-3, 1 - 1e-3))
        dosage = rng.binomial(2, p, size=n_patients).astype(np.int8)
        cols[f"{locus.rsid}"] = dosage
    frame = pd.DataFrame(cols)
    frame.insert(0, "genotype_seed", np.full(n_patients, seed, dtype=np.int64))
    return frame


def dosage_matrix(genotypes: pd.DataFrame) -> np.ndarray:
    """(n_patients, n_loci) float32 matrix in the canonical panel order."""
    return genotypes[[s.rsid for s in SNP_PANEL]].to_numpy(dtype=np.float32)


def trait_weight_matrix() -> np.ndarray:
    """(n_loci, n_traits) matrix of true per-allele effect weights."""
    w = np.zeros((len(SNP_PANEL), len(TRAITS)), dtype=np.float32)
    for i, locus in enumerate(SNP_PANEL):
        w[i, TRAITS.index(locus.trait)] = locus.effect_weight
    return w


# --------------------------------------------------------------------------- #
# Polygenic Risk Score
# --------------------------------------------------------------------------- #
def compute_prs(genotypes: pd.DataFrame, cohort_dosage: np.ndarray | None = None) -> pd.DataFrame:
    """Compute standardised polygenic risk scores.

    PRS_raw[i] = sum_j dosage[i, j] * beta[j]

    The raw score is then standardised against the *cohort* mean/SD (or a
    reference distribution supplied by the caller) to yield a z-score. The
    per-trait decomposition (T2D-PRS, CVD-PRS, ...) is what the cross-modal
    fusion module consumes — it allows the model to reason about *which*
    physiological system a given patient's vulnerability lies in.
    """
    dosage = dosage_matrix(genotypes)
    w = trait_weight_matrix()

    raw_total = dosage @ w.sum(axis=1)

    traits = {}
    for k, trait in enumerate(TRAITS):
        traits[f"prs_{trait.lower()}"] = dosage @ w[:, k]

    out = pd.DataFrame(traits)
    out.insert(0, "prs_raw", raw_total.astype(np.float32))

    # Standardisation reference: use cohort statistics when provided so that
    # train/inference remain consistent.
    for col in ["prs_raw"] + list(traits.keys()):
        vals = out[col].to_numpy(dtype=np.float64)
        if cohort_dosage is not None:
            ref = cohort_dosage @ (w.sum(axis=1) if col == "prs_raw" else w[:, TRAITS.index(col[4:].upper())])
        else:
            ref = vals
        mu, sd = float(np.mean(ref)), float(np.std(ref))
        sd = sd if sd > 1e-8 else 1.0
        out[f"{col}_z"] = ((vals - mu) / sd).astype(np.float32)
    return out


def vulnerability_vector(genotypes: pd.DataFrame, prs: pd.DataFrame) -> np.ndarray:
    """Compact per-patient genomic vulnerability vector fed to the static encoder.

    Layout:
        [0]      prs_total_z
        [1:5]    prs_{t2d,cvd,dys,aut}_z
        [5:5+L]  dosage of each locus (standardised)
        [last]   max per-locus z contribution (dominant driver)
    """
    dosage = dosage_matrix(genotypes).astype(np.float32)
    zdosage = (dosage - dosage.mean(axis=0, keepdims=True)) / (
        dosage.std(axis=0, keepdims=True) + 1e-6
    )
    w = trait_weight_matrix()
    # ``max`` over loci is the dominant-driver term described in the layout:
    # the single variant contributing the most standardised risk to this
    # patient. (A mean would dilute exactly the signal the encoder needs.)
    per_locus_z = (zdosage * w.sum(axis=1)[None, :]).max(axis=1, keepdims=True)

    parts = [
        prs[["prs_raw_z"]].to_numpy(np.float32),
        prs[[f"prs_{t.lower()}_z" for t in TRAITS]].to_numpy(np.float32),
        zdosage.astype(np.float32),
        per_locus_z.astype(np.float32),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def dominant_driver(genotypes: pd.DataFrame) -> list[str]:
    """Per-patient rsid contributing the largest share of polygenic risk.

    Returns one rsid per row of ``genotypes`` (aligned to its index), so the
    dashboard can say *which* variant is driving a given patient's risk rather
    than reporting a cohort-level average.
    """
    dosage = dosage_matrix(genotypes).astype(np.float64)
    w = trait_weight_matrix().sum(axis=1)
    contrib = dosage * w[None, :]
    idx = np.argmax(contrib, axis=1)
    return [SNP_PANEL[int(i)].rsid for i in idx]


# --------------------------------------------------------------------------- #
# Panel persistence (shipped as a machine-readable artifact)
# --------------------------------------------------------------------------- #
def panel_frame() -> pd.DataFrame:
    return pd.DataFrame([s.as_row() for s in SNP_PANEL])


def dump_panel(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel_frame().to_csv(path, index=False)
    meta = {
        "n_loci": len(SNP_PANEL),
        "traits": list(TRAITS),
        "genes": list(PANEL_GENES),
        "note": "Effect weights are simulated ground truth for the challenge dataset.",
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_panel(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    print(panel_frame().to_string(index=False))
