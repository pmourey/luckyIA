import os
import numpy as np
import pandas as pd
import pytest

from web_app import app as webapp


def make_dummy_model_input_dim(n_features):
    # create a tiny PyTorch nn.Module with a single Linear layer of in_features=n_features
    try:
        import torch
        import torch.nn as nn
    except Exception:
        pytest.skip('torch not available in environment for this test')
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(n_features, 1)
        def forward(self, x):
            return self.linear(x)
    return M()


def test_prepare_tensor_removes_risque_and_matches_dim(tmp_path):
    # Build a DataFrame with 6 numeric columns including 'risque'
    df = pd.DataFrame({
        'age': np.random.randint(18,80,size=10),
        'revenu': np.random.randint(15000,200000,size=10),
        'anciennete': np.random.randint(0,30,size=10),
        'score_credit': np.random.randint(300,850,size=10),
        'depenses_mensuelles': np.random.randint(500,8000,size=10),
        'risque': np.random.randint(0,2,size=10)
    })

    # Create a dummy model expecting 5 features
    model = make_dummy_model_input_dim(5)

    tensor, err_msg, warning = webapp._prepare_tensor_for_model(df, model)
    assert err_msg is None, f"_prepare_tensor_for_model returned error: {err_msg}"
    assert tensor is not None
    # tensor should have shape (10, 5)
    assert tensor.shape[0] == 10
    assert tensor.shape[1] == 5
    # If a warning exists, it should mention the removed column 'risque'
    if warning:
        assert 'risque' in warning or 'Colonnes supprimées' in warning


def test_prepare_tensor_without_risque_works(tmp_path):
    df = pd.DataFrame({
        'age': np.random.randint(18,80,size=8),
        'revenu': np.random.randint(15000,200000,size=8),
        'anciennete': np.random.randint(0,30,size=8),
        'score_credit': np.random.randint(300,850,size=8),
        'depenses_mensuelles': np.random.randint(500,8000,size=8)
    })
    model = make_dummy_model_input_dim(5)
    tensor, err_msg, warning = webapp._prepare_tensor_for_model(df, model)
    assert err_msg is None
    assert tensor is not None
    assert tensor.shape == (8,5)

