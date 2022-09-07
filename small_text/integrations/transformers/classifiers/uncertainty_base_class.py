import abc

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.integrations.transformers.classifiers.classification import (
    TransformerBasedClassification,
)
from small_text.utils.classification import empty_result

import numpy as np

from functools import partial


try:
    import torch
    import torch.nn.functional as F  # noqa: N812
    from small_text.integrations.pytorch.utils.data import dataloader
except ImportError:
    raise PytorchNotFoundError("Could not import pytorch")


class UncertaintyBaseClass(TransformerBasedClassification):
    @abc.abstractmethod
    def predict_proba(self, test_set):
        raise NotImplementedError


class SoftmaxUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        if len(test_set) == 0:
            return empty_result(
                self.multi_label,
                self.num_classes,
                return_prediction=False,
                return_proba=True,
            )

        self.model.eval()
        test_iter = dataloader(
            test_set.data, self.mini_batch_size, self._create_collate_fn(), train=False
        )

        predictions = []
        logits_transform = (
            torch.sigmoid if self.multi_label else partial(F.softmax, dim=1)
        )

        with torch.no_grad():
            for text, masks, *_ in test_iter:
                text, masks = text.to(self.device), masks.to(self.device)
                outputs = self.model(text, attention_mask=masks)

                predictions += logits_transform(outputs.logits).to("cpu").tolist()
                del text, masks

        return np.array(predictions)


class TemperatureScalingUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError


class LabelSmoothingUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError


class MonteCarloDropoutUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError


class InhibitedSoftmaxUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError


class EvidentialDeepLearning1UncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError


class EvidentialDeepLearning2UncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError


class BayesianUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError


class EnsemblesUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError


class TrustScoreUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError


class ModelCalibrationUncertaintyClassifier(UncertaintyBaseClass):
    def predict_proba(self, test_set):
        raise NotImplementedError
