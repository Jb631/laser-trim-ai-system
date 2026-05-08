import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import laser_trim_analyzer.ml.predictor as predictor_module
from laser_trim_analyzer.ml.predictor import ModelPredictor


class PassthroughScaler:
    def transform(self, values):
        return values


class SingleClassClassifier:
    def __init__(self, only_class):
        self.classes_ = np.array([only_class])


def test_predict_with_confidence_treats_string_zero_class_as_pass():
    predictor = ModelPredictor("test-model")
    predictor.is_trained = True
    predictor.scaler = PassthroughScaler()
    predictor.classifier = SingleClassClassifier("0")
    predictor.feature_importance = {"sigma_gradient": 1.0}
    predictor.feature_means = {"sigma_gradient": 0.0}

    probability, lower, upper = predictor.predict_with_confidence({"sigma_gradient": 0.012})

    assert probability == 0.0
    assert lower == 0.0
    assert upper == 0.0


def test_save_failure_does_not_leave_partial_model_file(tmp_path, monkeypatch):
    predictor = ModelPredictor("test-model")
    predictor.is_trained = True
    predictor.classifier = SingleClassClassifier("0")
    predictor.scaler = PassthroughScaler()
    predictor.feature_importance = {"sigma_gradient": 1.0}
    predictor.feature_means = {"sigma_gradient": 0.0}
    predictor.feature_stds = {"sigma_gradient": 1.0}

    def failing_dump(*args, **kwargs):
        raise RuntimeError("simulated pickle write failure")

    monkeypatch.setattr(predictor_module.pickle, "dump", failing_dump)

    model_path = tmp_path / "model.pkl"

    assert predictor.save(model_path) is False
    assert not model_path.exists()
    assert not model_path.with_suffix(".hash").exists()
    assert not list(tmp_path.glob("*.tmp"))
