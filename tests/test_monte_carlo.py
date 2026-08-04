
"""Tests for monte_carlo module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.monte_carlo import (
    MonteCarloEngine, _bootstrap_means, _weighted_bootstrap_means,
    run_monte_carlo_analysis,
)

@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame({
        'Disruption_Occurred': rng.choice([0, 1], size=n, p=[0.4, 0.6]),
        'Lead_Time_Days': np.concatenate([
            rng.normal(10, 5, 120),
            rng.normal(30, 20, 180),
        ]),
        'Transport_Mode': rng.choice(['Air', 'Rail', 'Road', 'Sea'], size=n),
        'Geopolitical_Risk_Score': rng.uniform(0, 10, size=n),
        'Carrier_Reliability_Score': rng.uniform(0.5, 1.0, size=n),
        'Weather_Condition': rng.choice(['Clear', 'Fog', 'Rain', 'Storm', 'Hurricane'], size=n),
        'Distance_km': rng.uniform(500, 15000, size=n),
        'Weight_MT': rng.uniform(10, 500, size=n),
        'Origin_Port': rng.choice(['Shanghai', 'Singapore', 'Rotterdam', 'Busan'], size=n),
        'Destination_Port': rng.choice(['Los Angeles', 'Shanghai', 'Hamburg', 'Busan'], size=n),
        'Route': ['Shanghai -> Los Angeles'] * 75 + ['Singapore -> Shanghai'] * 75 +
                 ['Rotterdam -> Hamburg'] * 75 + ['Busan -> Busan'] * 75,
        'Shipment_ID': [f'SC-{i:05d}' for i in range(n)],
        'Fuel_Price_Index': rng.uniform(1, 5, size=n),
        'Product_Category': rng.choice(['Electronics', 'Textiles'], size=n),
    })

@pytest.fixture
def engine(sample_df):
    return MonteCarloEngine(sample_df, random_seed=42)

# ---- _bootstrap_means ----

def test_bootstrap_shape():
    rng = np.random.default_rng(0)
    vals = np.array([0, 1, 0, 1, 0])
    result = _bootstrap_means(rng, vals, n_samples=100)
    assert len(result) == 100
    assert 0 <= result.min() <= 1
    assert 0 <= result.max() <= 1

def test_bootstrap_reproducible():
    vals = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    r1 = _bootstrap_means(rng1, vals, 500)
    r2 = _bootstrap_means(rng2, vals, 500)
    assert np.allclose(r1, r2)

def test_weighted_bootstrap():
    rng = np.random.default_rng(0)
    vals = np.array([1, 1, 0, 0, 0])
    weights = np.array([10, 10, 1, 1, 1])  # 1's much more likely
    result = _weighted_bootstrap_means(rng, vals, weights, 1000)
    assert np.mean(result) > 0.6  # should be biased toward 1

# ---- MonteCarloEngine ----

def test_simulate_disruption_rate(engine):
    result = engine.simulate_disruption_rate(n_samples=500)
    assert 'observed' in result
    assert 'ci_95_lower' in result
    assert result['ci_95_lower'] <= result['observed'] <= result['ci_95_upper']

def test_simulate_disruption_by_mode(engine):
    result = engine.simulate_disruption_by_mode(n_samples=200)
    assert len(result) > 0
    assert 'CI_95_Lower' in result.columns
    assert 'CI_95_Upper' in result.columns

def test_stress_test_geopolitical(engine):
    result = engine.stress_test_geopolitical_risk(risk_shift=2.0, n_samples=200)
    assert 'pct_change' in result
    assert 'prob_worse' in result

def test_stress_test_weather(engine):
    result = engine.stress_test_weather(n_samples=200)
    assert 'pct_change' in result

def test_route_var(engine):
    result = engine.route_var_simulation(top_n=5, n_samples=200)
    assert 'VaR_95' in result.columns
    assert 'CVaR_95' in result.columns
    assert (result['VaR_95'] >= result['Observed_Mean_LT']).all()

def test_total_cost(engine):
    result = engine.simulate_total_cost(n_samples=200)
    assert result['mean_cost'] > 0
    assert result['VaR_95_cost'] >= result['mean_cost']

def test_roi_has_model_auc(engine):
    result = engine.simulate_reliability_improvement(improvement_pct=0.10, n_samples=200)
    assert 'model_auc_test' in result
    assert 'method' in result
    assert result['method'] == 'associational (not causal)'

# ---- Edge cases ----

def test_empty_disruption_column(engine):
    """Engine should handle all-zero disruption gracefully."""
    df_zero = engine.df.copy()
    df_zero['Disruption_Occurred'] = 0
    eng = MonteCarloEngine(df_zero, random_seed=42)
    result = eng.simulate_disruption_rate(n_samples=100)
    assert abs(result['observed']) < 0.01

def test_run_monte_carlo_analysis(engine):
    results = run_monte_carlo_analysis(engine, n_samples=100)
    assert 'disruption_rate' in results
    assert 'route_var' in results
    assert 'total_cost' in results
    assert 'roi' in results
