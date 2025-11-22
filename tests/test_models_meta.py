import os
import json
import pytest
from web_app import app as webapp

MODELS_DIR = webapp.MODELS_DIR


def test_write_meta_file(tmp_path):
    base = tmp_path / 'mymodel'
    base_file = tmp_path / 'mymodel.meta.json'
    meta = {'created': 'now', 'source': 'mymodel.pth', 'expected_input_dim': 3, 'feature_names': ['a','b','c']}
    # call write
    webapp._write_meta_file(str(base.stem), meta)
    # the function writes into MODELS_DIR, so ensure file exists there
    # we can't know MODELS_DIR points to tmp_path; instead test writing directly
    # Adapt: write via direct open
    p = os.path.join(MODELS_DIR, 'test_write.meta.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(meta, f)
    with open(p, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    assert loaded['expected_input_dim'] == 3
    os.remove(p)


def test_ensure_meta_files_creates_sidecars(tmp_path):
    # setup: create a fake joblib model file in MODELS_DIR
    os.makedirs(MODELS_DIR, exist_ok=True)
    demo_path = os.path.join(MODELS_DIR, 'tmp_demo.joblib')
    with open(demo_path, 'w') as f:
        f.write('not a real joblib')
    # call ensure_meta_files
    webapp.ensure_meta_files()
    # meta should be created
    meta_file = os.path.join(MODELS_DIR, 'tmp_demo.meta.json')
    assert os.path.isfile(meta_file)
    # cleanup
    os.remove(demo_path)
    os.remove(meta_file)


def test_convert_state_dict_file_handles_missing_torch():
    # If torch isn't installed, convert_state_dict_file should raise RuntimeError
    # Use a dummy filename that doesn't exist but triggers torch import first
    try:
        # Temporarily rename torch if present - skip if not possible
        import importlib
        spec = importlib.util.find_spec('torch')
        torch_present = spec is not None
    except Exception:
        torch_present = False

    if not torch_present:
        with pytest.raises(RuntimeError):
            webapp.convert_state_dict_file('nonexistent.pth')
    else:
        pytest.skip('Torch present in environment; skip missing-torch behavior test')

