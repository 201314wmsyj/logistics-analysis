
"""Tests for statistical_tests module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

import src.statistical_tests as st_mod

@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        'Disruption_Occurred': rng.choice([0, 1], size=n, p=[0.4, 0.6]),
        'Lead_Time_Days': np.concatenate([
            rng.normal(10, 5, 80),
            rng.normal(30, 20, 120),
        ]),
        'Transport_Mode': rng.choice(['Air', 'Rail', 'Road', 'Sea'], size=n),
        'Product_Category': rng.choice(['Electronics', 'Textiles', 'Automotive', 'Perishables', 'Pharmaceuticals'], size=n),
        'Weather_Condition': rng.choice(['Clear', 'Fog', 'Rain', 'Storm', 'Hurricane'], size=n),
        'Risk_Level': rng.choice(['Low', 'Medium', 'High'], size=n),
    })

# ---- _cohens_d ----

def test_cohens_d_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert abs(st_mod._cohens_d(a, a)) < 1e-10

def test_cohens_d_known():
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([2.0, 3.0, 4.0])
    d = st_mod._cohens_d(a, b)
    assert d <= -2.0

def test_cohens_d_zero_std():
    a = np.array([5.0, 5.0, 5.0])
    b = np.array([5.0, 5.0, 5.0])
    assert st_mod._cohens_d(a, b) == 0.0

# ---- _cramers_v ----

def test_cramers_v_independent():
    ct = pd.DataFrame([[50, 50], [50, 50]])
    v = st_mod._cramers_v(ct)
    assert v < 0.05

def test_cramers_v_perfect():
    ct = pd.DataFrame([[100, 0], [0, 100]])
    v = st_mod._cramers_v(ct)
    assert abs(v - 1.0) < 0.02

def test_cramers_v_single_row():
    ct = pd.DataFrame([[50, 50]])
    assert st_mod._cramers_v(ct) == 0.0

# ---- _bonferroni_corrected_significance ----

def test_bonferroni_rejects():
    _p_adj, sig = st_mod._bonferroni_corrected_significance(0.001, 6)
    assert sig

def test_bonferroni_accepts():
    _p_adj, sig = st_mod._bonferroni_corrected_significance(0.05, 10)
    assert not sig

def test_bonferroni_clamps():
    p_adj, sig = st_mod._bonferroni_corrected_significance(0.5, 10)
    assert p_adj == 1.0
    assert not sig

# ---- test functions return expected keys ----

def test_ttest_returns_keys(sample_df):
    result = st_mod.test_lead_time_by_disruption(sample_df)
    assert 'p_value' in result
    assert 'effect_size_cohens_d' in result
    assert 'effect_magnitude' in result

def test_anova_returns_keys(sample_df):
    result = st_mod.test_lead_time_by_mode(sample_df)
    assert 'p_value' in result
    assert 'effect_size_eta_squared' in result
    assert 'tukey_posthoc' in result

def test_chi2_independence_returns_keys(sample_df):
    result = st_mod.test_disruption_independence(sample_df)
    assert 'Transport_Mode' in result
    assert 'Product_Category' in result

def test_weather_chi2_returns_keys(sample_df):
    result = st_mod.test_weather_disruption(sample_df)
    assert 'cramers_v' in result

def test_risk_chi2_returns_keys(sample_df):
    result = st_mod.test_risk_level_disruption(sample_df)
    assert 'cramers_v' in result

# ---- run_statistical_tests integration ----

def test_run_statistical_tests(sample_df):
    results = st_mod.run_statistical_tests(sample_df)
    assert len(results) == 5
    assert 'ttest_lt_disruption' in results
    assert 'anova_lt_mode' in results
