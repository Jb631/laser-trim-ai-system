import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
