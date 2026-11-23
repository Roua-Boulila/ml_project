"""Exemple de test unitaire pour le pipeline ML."""

import os

from model_pipeline import load_model, save_model


def test_save_and_load_model(tmp_path):
    """Un objet sauvegardé puis rechargé doit rester identique."""
    obj = {"name": "demo", "value": 42}
    path = os.path.join(tmp_path, "demo.joblib")
    save_model(obj, path)
    assert os.path.exists(path)
    assert load_model(path) == obj
