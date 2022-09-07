from classification import TransformerBasedClassification
import abc

from sklearn.preprocessing import MultiLabelBinarizer

from small_text.classifiers.classification import EmbeddingMixin
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.utils.classification import empty_result, get_splits
from small_text.utils.context import build_pbar_context
from small_text.utils.data import check_training_data, list_length
from small_text.utils.datetime import format_timedelta
from small_text.utils.labels import csr_to_list, get_num_labels
from small_text.utils.logging import verbosity_logger, VERBOSITY_MORE_VERBOSE
from small_text.utils.system import get_tmp_dir_base

try:
    import torch
    import torch.nn.functional as F  # noqa: N812

    from torch.optim import AdamW
    from transformers import logging as transformers_logging
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    from small_text.integrations.pytorch.classifiers.base import (
        check_optimizer_and_scheduler_config,
        PytorchClassifier,
    )
    from small_text.integrations.pytorch.model_selection import (
        Metric,
        PytorchModelSelection,
    )
    from small_text.integrations.pytorch.utils.data import dataloader
    from small_text.integrations.pytorch.utils.misc import (
        early_stopping_deprecation_warning,
    )
    from small_text.integrations.transformers.datasets import TransformersDataset
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
