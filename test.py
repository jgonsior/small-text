"""Example of a binary active learning text classification.
"""
import random
import numpy as np
from sklearn.metrics import accuracy_score
import torch

from small_text.active_learner import PoolBasedActiveLearner
from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.classifiers.factories import SklearnClassifierFactory
from small_text.query_strategies import PoolExhaustedException, EmptyPoolException
from small_text.query_strategies import RandomSampling


from small_text.active_learner import PoolBasedActiveLearner

from small_text.initialization import random_initialization_balanced
from small_text.integrations.transformers import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import (
    TransformerBasedClassificationFactory,
)
from small_text.query_strategies import (
    # DeepEnsemble,
    # TrustScore2,
    # TrustScore,
    # EvidentialConfidence2,
    # BT_Temp,
    # TemperatureScaling,
    # BreakingTies,
    RandomSampling,
    # PredictionEntropy,
    # FalkenbergConfidence2,
    # FalkenbergConfidence,
    LeastConfidence,
)

from small_text.integrations.transformers import TransformerModelArguments


from dataset_loader import load_my_dataset


def main(
    num_iterations: int,
    batch_size: int,
    dataset: str,
    transformer_model_name: str,
    initially_labeled_samples: int,
):
    train, test, num_classes = load_my_dataset(dataset, transformer_model_name)

    cpu_cuda = "cpu"
    if torch.cuda.is_available():
        cpu_cuda = "cuda"
        print("cuda available")

    transformer_model = TransformerModelArguments(transformer_model_name)
    clf_factory = TransformerBasedClassificationFactory(
        transformer_model,
        num_classes,
        kwargs=dict(
            {
                "device": cpu_cuda,
                "mini_batch_size": 64,
                "class_weight": "balanced",
            }
        ),
    )
    query_strategy = LeastConfidence()

    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
    labeled_indices = initialize_active_learner(
        active_learner, train.y, initially_labeled_samples
    )

    try:
        perform_active_learning(
            active_learner, train, labeled_indices, test, num_iterations, batch_size
        )
    except PoolExhaustedException:
        print("Error! Not enough samples left to handle the query.")
    except EmptyPoolException:
        print("Error! No more samples left. (Unlabeled pool is empty)")


def _evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    test_acc = accuracy_score(y_pred_test, test.y)
    train_acc = accuracy_score(y_pred, train.y)
    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")

    print("---")
    return (train_acc, test_acc)


def perform_active_learning(
    active_learner, train, indices_labeled, test, num_iterations, batch_size
):
    results = []

    for i in range(num_iterations):
        indices_queried = active_learner.query(num_samples=batch_size)

        y = train.y[indices_queried]

        active_learner.update(y)

        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        print("Iteration #{:d} ({} samples)".format(i, len(indices_labeled)))
        results.append(_evaluate(active_learner, train[indices_labeled], test))


def initialize_active_learner(active_learner, y_train, initially_labeled_samples: int):
    indices_initial = random_initialization_balanced(
        y_train, n_samples=initially_labeled_samples
    )
    active_learner.initialize_data(indices_initial, y_train[indices_initial])

    return indices_initial


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="An example that shows active learning "
        "for binary text classification."
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="number of active learning iterations",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--initially_labeled_samples",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["20newsgroups", "ag_news", "trec6", "subj", "rotten", "imdb"],
        default="20newsgroups",
    )
    parser.add_argument(
        "--transformer_model_name",
        type=str,
        default="bert-base-uncased",
    )
    args = parser.parse_args()

    # set random seed
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    main(
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        dataset=args.dataset,
        transformer_model_name=args.transformer_model_name,
        initially_labeled_samples=args.initially_labeled_samples,
    )