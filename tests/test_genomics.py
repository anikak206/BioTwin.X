"""Genomic layer: panel integrity, HWE genotypes, PRS maths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biotwin.genomics import (
    PANEL_BY_RSID,
    SNP_PANEL,
    compute_prs,
    dominant_driver,
    dosage_matrix,
    panel_frame,
    simulate_genotypes,
    vulnerability_vector,
)


def test_panel_is_internally_consistent():
    assert len(SNP_PANEL) >= 16
    rsids = [locus.rsid for locus in SNP_PANEL]
    assert len(rsids) == len(set(rsids)), "duplicate rsIDs in the panel"
    for locus in SNP_PANEL:
        assert locus.rsid.startswith("rs")
        assert 0.0 < locus.population_freq < 1.0
        assert locus.gene
        assert locus.trait in {"T2D", "CVD", "DYS", "AUT"}
        assert np.isfinite(locus.effect_weight)
    assert set(PANEL_BY_RSID) == set(rsids)


def test_simulate_genotypes_are_in_hwe_range():
    n = 4000
    gt = simulate_genotypes(n, seed=7)
    assert len(gt) == n
    for locus in SNP_PANEL:
        dosage = gt[locus.rsid].to_numpy()
        assert set(np.unique(dosage)).issubset({0, 1, 2})
        # Under HWE the mean dosage is 2p and the variance is 2p(1-p).
        p = locus.population_freq
        assert abs(dosage.mean() - 2 * p) < 0.08
        assert abs(dosage.var() - 2 * p * (1 - p)) < 0.08


def test_prs_is_standardised_and_rank_preserving():
    gt = simulate_genotypes(600, seed=11)
    prs = compute_prs(gt)
    for trait in ("raw", "cvd", "t2d", "dys", "aut"):
        col = prs[f"prs_{trait}_z"]
        assert np.isfinite(col).all()
        assert abs(col.mean()) < 0.2
        assert 0.7 < col.std() < 1.3
    # raw z must be a strictly monotone function of the raw score
    if "prs_raw" in prs:
        order = np.argsort(prs["prs_raw"].to_numpy())
        z = prs["prs_raw_z"].to_numpy()[order]
        assert np.all(np.diff(z) >= -1e-9)


def test_vulnerability_vector_shape_and_dominance():
    gt = simulate_genotypes(40, seed=3)
    prs = compute_prs(gt)
    vec = vulnerability_vector(gt, prs)
    assert vec.shape[0] == 40
    assert np.isfinite(vec).all()
    drivers = dominant_driver(gt)
    assert len(drivers) == 40
    assert all(d in PANEL_BY_RSID for d in set(drivers))


def test_panel_frame_roundtrip(tmp_path):
    from biotwin.genomics import dump_panel, load_panel

    frame = panel_frame()
    assert isinstance(frame, pd.DataFrame)
    assert {"rsid", "gene", "trait", "effect_weight"} <= set(frame.columns)
    path = dump_panel(tmp_path / "panel.csv")
    again = load_panel(path)
    assert len(again) == len(frame)
    assert set(again["rsid"]) == set(frame["rsid"])
