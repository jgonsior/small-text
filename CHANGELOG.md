# Changelog

## Version 1.1.0 - unreleased

### Added

- General:
  - Small-Text package is now available via [conda-forge](https://anaconda.org/conda-forge/small-text). 
  
- Classification:
  - All classifiers now support weighting of training samples.
  - [Early stopping](https://small-text.readthedocs.io/en/v1.1.0/components/classification.html) has been reworked, improved, and documented ([#18](https://github.com/webis-de/small-text/issues/18)).
  - **[!]** `KimCNNClassifier.__init()__`: The default value of the (now deprecated) keyword argument `early_stopping_acc` has been changed from `0.98` to `-1` in order to match `TransformerBasedClassification`.

- Query Strategies:
  - New multi-label strategy: [CategoryVectorInconsistencyAndRanking](https://github.com/webis-de/small-text/blob/v1.1.0/small_text/query_strategies/multi_label.py)

### Deprecated

- `small_text.integrations.pytorch.utils.misc.default_tensor_type()` is deprecated without replacement ([#2](https://github.com/webis-de/small-text/issues/2)).
- `TransformerBasedClassification` and `KimCNNClassifier`:
  The keyword arguments for early stopping (early_stopping / early_stopping_no_improvement, early_stopping_acc) that are passed to `__init__()` are now deprecated. Use the `early_stopping`
  keyword argument in the `fit()` method instead ([#18](https://github.com/webis-de/small-text/issues/18)). 


### Fixed
- Classification:
  - `KimCNNClassifier.fit()` and `TransformerBasedClassification.fit()` now correctly
    process the `scheduler` keyword argument ([#16](https://github.com/webis-de/small-text/issues/16)).

## Version 1.0.0 - 2022-06-14

First stable release.

### Changed

- Datasets:
  - `SklearnDataset` now checks if the dimensions of the features and labels match.
- Query Strategies:
  - [ExpectedGradientLengthMaxWord](https://github.com/webis-de/small-text/blob/main/small_text/integrations/pytorch/query_strategies/strategies.py): Cleaned up code and added checks to detect invalid configurations.
- Documentation:
  - The documentation is now available in full width.
- Repository:
  - Versions in this can now be referenced using the respective [Zenodo DOI](https://zenodo.org/record/6641063).

## [1.0.0b4] - 2022-05-04

### Added

- General:
  - We now have a concept for [optional dependencies](https://small-text.readthedocs.io/en/v1.0.0b4/install.html#optional-dependencies) which 
    allows components to rely on soft dependencies, i.e. python dependencies which can be installed on demand
    (and only when certain functionality is needed).
- Datasets:
  - The `Dataset` interface now has a `clone()` method 
    that creates an identical copy of the respective dataset.
- Query Strategies:
  - New strategies: [DiscriminativeActiveLearning](https://github.com/webis-de/small-text/blob/v1.0.0b4/small_text/query_strategies/strategies.py) 
    and [SEALS](https://github.com/webis-de/small-text/blob/v1.0.0b4/small_text/query_strategies/strategies.py).

### Changed

- Datasets:
  - Separated the previous `DatasetView` implementation into interface (`DatasetView`) 
    and implementation (`SklearnDatasetView`).
  - Added `clone()` method which creates an identical copy of the dataset.
- Query Strategies:
  - `EmbeddingBasedQueryStrategy` now only embeds instances that are either in the label
    or in the unlabeled pool (and no longer the entire dataset).
- Code examples:
  - Code structure was  unified.
  - Number of iterations can now be passed via an cli argument.
- `small_text.integrations.pytorch.utils.data`:
  - Method `get_class_weights()` now scales the resulting multi-class weights so that the smallest
    class weight is equal to `1.0`.

## [1.0.0b3] - 2022-03-06

### Added

- New query strategy: [ContrastiveActiveLearning](https://github.com/webis-de/small-text/blob/v1.0.0b3/small_text/query_strategies/strategies.py).
- Added [Reproducibility Notes](https://small-text.readthedocs.io/en/v1.0.0b3/reproducibility_notes.html).

### Changed

- Cleaned up and unified argument naming: The naming of variables related to datasets and 
  indices has been improved and unified. The naming of datasets had been inconsistent, 
  and the previous `x_` notation for indices was a relict of earlier versions of this library and 
  did not reflect the underlying object anymore.
  - `PoolBasedActiveLearner`:
    - attribute `x_indices_labeled` was renamed to `indices_labeled`
    - attribute `x_indices_ignored` was unified to `indices_ignored`
    - attribute `queried_indices` was unified to `indices_queried`
    - attribute `_x_index_to_position` was named to `_index_to_position`
    - arguments `x_indices_initial`, `x_indices_ignored`, and `x_indices_validation` were
      renamed to `indices_initial`, `indices_ignored`, and `indices_validation`. This affects most 
      methods of the `PoolBasedActiveLearner`.
    
  - `QueryStrategy`
    - old: `query(self, clf, x, x_indices_unlabeled, x_indices_labeled, y, n=10)`
    - new: `query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10)`
    
  - `StoppingCriterion`
    - old: `stop(self, active_learner=None, predictions=None, proba=None, x_indices_stopping=None)`
    - new: `stop(self, active_learner=None, predictions=None, proba=None, indices_stopping=None)`

- Renamed environment variable which sets the small-text temp folder from `ALL_TMP` to `SMALL_TEXT_TEMP`


## [1.0.0b2] - 2022-02-22

Bugfix release.

### Fixed

- Fix links to the documentation in README.md and notebooks.


## [1.0.0b1] - 2022-02-22

First beta release with multi-label functionality and stopping criteria.

### Added

- Added a changelog.
- All provided classifiers are now capable of multi-label classification.

### Changed

- Documentation has been overhauled considerably.
- `PoolBasedActiveLearner`: Renamed `incremental_training` kwarg to `reuse_model`.
- `SklearnClassifier`: Changed `__init__(clf)` to `__init__(model, num_classes, multi_Label=False)`
- `SklearnClassifierFactory`: `__init__(clf_template, kwargs={})` to `__init__(base_estimator, num_classes, kwargs={})`.
- Refactored `KimCNNClassifier` and `TransformerBasedClassification`.

### Removed

- Removed `device` kwarg from `PytorchDataset.__init__()`, 
`PytorchTextClassificationDataset.__init__()` and `TransformersDataset.__init__()`.
