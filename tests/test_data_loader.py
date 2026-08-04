
"""Tests for data_loader module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.data_loader import clean_data, engineer_features, validate_data

# ---- Fixtures ----

@pytest.fixture
def valid_sample_df():
    return pd.DataFrame({
        'Shipment_ID': ['SC-001', 'SC-002', 'SC-003', 'SC-004', 'SC-005'],
        'Date': pd.to_datetime(['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05', '2024-05-12']),
        'Origin_Port': ['Shanghai', 'Singapore', 'Rotterdam', 'Busan', 'Shanghai'],
        'Destination_Port': ['Los Angeles', 'Shanghai', 'Los Angeles', 'Hamburg', 'Busan'],
        'Transport_Mode': ['Sea', 'Rail', 'Rail', 'Road', 'Air'],
        'Product_Category': ['Electronics', 'Textiles', 'Automotive', 'Perishables', 'Pharmaceuticals'],
        'Distance_km': [12000.0, 5000.0, 11000.0, 9000.0, 800.0],
        'Weight_MT': [200.0, 150.0, 300.0, 100.0, 50.0],
        'Fuel_Price_Index': [2.5, 2.3, 1.8, 3.2, 2.1],
        'Geopolitical_Risk_Score': [3.0, 5.0, 5.6, 0.8, 2.0],
        'Weather_Condition': ['Clear', 'Fog', 'Rain', 'Storm', 'Clear'],
        'Carrier_Reliability_Score': [0.85, 0.60, 0.67, 0.83, 0.92],
        'Lead_Time_Days': [30.0, 40.0, 11.5, 8.0, 1.5],
        'Disruption_Occurred': [0, 1, 0, 1, 0],
    })

# ---- validate_data ----

def test_validate_passes_valid_data(valid_sample_df):
    result = validate_data(valid_sample_df.copy())
    assert result is not None

def test_validate_raises_on_missing_column(valid_sample_df):
    df = valid_sample_df.drop(columns=['Distance_km'])
    with pytest.raises(ValueError, match='Distance_km'):
        validate_data(df)

def test_validate_warns_on_out_of_range(valid_sample_df, capsys):
    df = valid_sample_df.copy()
    df.loc[0, 'Lead_Time_Days'] = -5
    _ = validate_data(df)
    captured = capsys.readouterr()
    assert '时效' in captured.out or 'Lead_Time_Days' in captured.out

def test_validate_accepts_all_zero_disruption(valid_sample_df):
    df = valid_sample_df.copy()
    df['Disruption_Occurred'] = 0
    result = validate_data(df)
    assert result is not None

# ---- clean_data ----

def test_clean_handles_nulls(valid_sample_df):
    df = valid_sample_df.copy()
    df.loc[0, 'Distance_km'] = np.nan
    df.loc[1, 'Weather_Condition'] = None
    result = clean_data(df)
    assert not result['Distance_km'].isnull().any()
    assert not result['Weather_Condition'].isnull().any()

def test_clean_preserves_row_count(valid_sample_df):
    result = clean_data(valid_sample_df.copy())
    assert len(result) == 5

# ---- engineer_features ----

def test_engineer_adds_expected_columns(valid_sample_df):
    result = engineer_features(valid_sample_df.copy())
    expected_new = ['Year', 'Month', 'Quarter', 'Speed_km_per_day', 'Ton_KM', 'Route', 'Risk_Level', 'Lead_Time_Category']
    for col in expected_new:
        assert col in result.columns, f'Missing {col}'

def test_engineer_route_format(valid_sample_df):
    result = engineer_features(valid_sample_df.copy())
    assert result['Route'].iloc[0] == 'Shanghai -> Los Angeles'

def test_engineer_speed_calculation(valid_sample_df):
    result = engineer_features(valid_sample_df.copy())
    expected_speed = 12000.0 / 30.0
    assert abs(result['Speed_km_per_day'].iloc[0] - expected_speed) < 0.01

# ---- End-to-end ----

def test_prepare_data_pipeline(valid_sample_df, tmp_path):
    csv_path = tmp_path / 'test.csv'
    valid_sample_df.to_csv(csv_path, index=False)
    from src.data_loader import prepare_data
    result = prepare_data(str(csv_path))
    assert result.shape[0] == 5
    assert 'Route' in result.columns
    assert 'Risk_Level' in result.columns
