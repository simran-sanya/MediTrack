"""
MediTrack — Shared Preprocessing Module
========================================
Provides canonical data loading, feature engineering, train/val/test split,
and one-hot encoding for all MediTrack notebooks.

Purpose:
  - Guarantee identical eligible populations across LR, RF, GB, NB
  - Guarantee identical final-test encounter IDs
  - Guarantee identical feature definitions for classical models (LR/RF/GB)
  - Prevent preprocessing leakage (OHE fitted on train only)

This module is imported by each notebook but does NOT replace the notebook's
own code, analysis, or explanations.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
EXCLUDE_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]

# ─────────────────────────────────────────────────
# 1. Canonical Feature Column Definitions
# ─────────────────────────────────────────────────
FEATURES = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'age', 'gender',
    'primary_diagnosis_group', 'num_medication_changes',
]

OHE_COLUMNS = ['age', 'gender', 'primary_diagnosis_group']

MEDICATION_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
]


# ─────────────────────────────────────────────────
# 2. Data Loading
# ─────────────────────────────────────────────────
def load_clean_data(data_path='diabetic_data.csv'):
    """Load the UCI Diabetes dataset and apply deceased/hospice exclusions."""
    if not os.path.exists(data_path):
        for alt in ['../diabetic_data.csv', '../../diabetic_data.csv',
                     os.path.join(os.path.dirname(__file__), 'diabetic_data.csv'),
                     os.path.join(os.path.dirname(__file__), '..', 'diabetic_data.csv')]:
            if os.path.exists(alt):
                data_path = alt
                break

    df = pd.read_csv(data_path)
    assert df.shape[0] == 101766, f"Expected 101,766 raw rows, got {df.shape[0]}"

    df_clean = df.replace('?', np.nan)
    df_clean = df_clean[~df_clean['discharge_disposition_id'].isin(EXCLUDE_DISCHARGE_IDS)].copy()
    df_clean['risk_target'] = (df_clean['readmitted'] == '<30').astype(int)

    return df_clean


# ─────────────────────────────────────────────────
# 3. ICD-9 Grouping
# ─────────────────────────────────────────────────
def group_diagnosis(code):
    """Map a raw ICD-9 code to one of 9 broad clinical categories."""
    if pd.isna(code):
        return 'Missing'
    if isinstance(code, str) and (code.startswith('V') or code.startswith('E')):
        return 'Other'
    try:
        code_num = float(code)
    except ValueError:
        return 'Other'
    if 390 <= code_num <= 459 or code_num == 785:
        return 'Circulatory'
    elif 460 <= code_num <= 519 or code_num == 786:
        return 'Respiratory'
    elif 520 <= code_num <= 579 or code_num == 787:
        return 'Digestive'
    elif code_num == 250:
        return 'Diabetes'
    elif 800 <= code_num <= 999:
        return 'Injury'
    elif 710 <= code_num <= 739:
        return 'Musculoskeletal'
    elif 580 <= code_num <= 629 or code_num == 788:
        return 'Genitourinary'
    elif 140 <= code_num <= 239:
        return 'Neoplasms'
    else:
        return 'Other'


def apply_feature_engineering(df_clean):
    """Apply canonical feature engineering: ICD-9 grouping + medication changes."""
    df_clean.drop(columns=['weight', 'payer_code', 'medical_specialty'],
                  errors='ignore', inplace=True)
    df_clean['primary_diagnosis_group'] = df_clean['diag_1'].apply(group_diagnosis)
    df_clean['num_medication_changes'] = df_clean[MEDICATION_COLS].apply(
        lambda row: ((row == 'Up') | (row == 'Down')).sum(), axis=1
    )
    return df_clean


# ─────────────────────────────────────────────────
# 4. Canonical Split
# ─────────────────────────────────────────────────
def get_outer_test_split(index, target, test_size=0.20, random_state=RANDOM_SEED):
    """Return (dev_idx, test_idx) — the canonical 80/20 outer split.
    The test_idx is IDENTICAL across all four models (LR/RF/GB/NB)."""
    dev_idx, test_idx = train_test_split(
        index, test_size=test_size, stratify=target, random_state=random_state
    )
    return dev_idx, test_idx


def get_classical_split(index, target, test_size=0.20, val_frac=0.20,
                        random_state=RANDOM_SEED):
    """Return (train_idx, val_idx, test_idx) for classical models.
    Outer 80/20 dev/test -> then dev split 80/20 -> giving 64/16/20 overall."""
    dev_idx, test_idx = get_outer_test_split(index, target, test_size, random_state)
    y_dev = target.loc[dev_idx]
    train_idx, val_idx = train_test_split(
        dev_idx, test_size=val_frac, stratify=y_dev, random_state=random_state
    )
    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────
# 5. Feature Encoding (OHE fitted on train only)
# ─────────────────────────────────────────────────
def prepare_classical_features(df_clean, train_idx, val_idx, test_idx):
    """Create X_train, X_val, X_test with OHE fitted on TRAINING data only.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test"""
    X_all = df_clean[FEATURES].copy()

    num_features = [c for c in FEATURES if c not in OHE_COLUMNS]
    for c in num_features:
        X_all[c] = pd.to_numeric(X_all[c], errors='coerce').fillna(0)
    for c in OHE_COLUMNS:
        X_all[c] = X_all[c].astype(str).replace('nan', 'Missing').fillna('Missing')

    X_train_raw = X_all.loc[train_idx]
    X_val_raw = X_all.loc[val_idx]
    X_test_raw = X_all.loc[test_idx]

    # OHE on train only to learn vocabulary
    X_train = pd.get_dummies(X_train_raw, columns=OHE_COLUMNS, drop_first=True)
    train_columns = X_train.columns

    X_val = pd.get_dummies(X_val_raw, columns=OHE_COLUMNS, drop_first=True)
    X_val = X_val.reindex(columns=train_columns, fill_value=0)

    X_test = pd.get_dummies(X_test_raw, columns=OHE_COLUMNS, drop_first=True)
    X_test = X_test.reindex(columns=train_columns, fill_value=0)

    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    X_val = X_val.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

    y_train = df_clean.loc[train_idx, 'risk_target']
    y_val = df_clean.loc[val_idx, 'risk_target']
    y_test = df_clean.loc[test_idx, 'risk_target']

    return X_train, X_val, X_test, y_train, y_val, y_test
