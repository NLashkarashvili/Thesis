import numpy as np
import datasets
from datasets import Metric  
from sklearn.metrics import mean_squared_error


_CITATION = """\
@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}
"""

_DESCRIPTION = """\
Mean Squared Error(MSE) is the average of the square of difference between the predicted
and actual values.
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    references: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    sample_weight: array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput: {"raw_values", "uniform_average"} or array-like of shape (n_outputs,), default="uniform_average"
        Defines aggregating of multiple output values. Array-like value defines weights used to average errors.
        "raw_values" : Returns a full set of errors in case of multioutput input.
        "uniform_average" : Errors of all outputs are averaged with uniform weight.
    squared : bool, default=True
        If True returns MSE value, if False returns RMSE (Root Mean Squared Error) value.
Returns:
    mse : mean squared error.
Examples:
    >>> mse_metric = datasets.load_metric("mse")
    >>> predictions = [2.5, 0.0, 2, 8]
    >>> references = [3, -0.5, 2, 7]
    >>> results = mse_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'mse': 0.375}
    >>> rmse_result = mse_metric.compute(predictions=predictions, references=references, squared=False)
    >>> print(rmse_result)
    {'mse': 0.6123724356957945}
    If you're using multi-dimensional lists, then set the config as follows :
    >>> mse_metric = datasets.load_metric("mse", "multilist")
    >>> predictions = [[0.5, 1], [-1, 1], [7, -6]]
    >>> references = [[0, 2], [-1, 2], [8, -5]]
    >>> results = mse_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'mse': 0.7083333333333334}
    >>> results = mse_metric.compute(predictions=predictions, references=references, multioutput='raw_values')
    >>> print(results) # doctest: +NORMALIZE_WHITESPACE
    {'mse': array([0.41666667, 1. ])}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class RMSEMetric(Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(self._get_feature_types()),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html"
            ],
        )

    def _get_feature_types(self):
        # if self.config_name == "multilist":
            return {
                "predictions": datasets.Sequence(datasets.Value("float")),
                "references": datasets.Sequence(datasets.Value("float")),
            }
        # else:
        #     return {
        #         "predictions": datasets.Value("float"),
        #         "references": datasets.Value("float"),
        #     }

    def _compute(self, predictions, references, sample_weight=None, multioutput="uniform_average", squared=True):
        predictions = np.array(predictions)
        references = np.array(references)

        predictions_v, references_v = predictions[:, 0], references[:, 0]
        predictions_a, references_a = predictions[:, 1], references[:, 1]
        predictions_d, references_d = predictions[:, 2], references[:, 2]
        rmse_v = np.sqrt(mean_squared_error(
            references_v, predictions_v, sample_weight=sample_weight, multioutput=multioutput, squared=squared
        ))
        rmse_a = np.sqrt(mean_squared_error(
            references_a, predictions_a, sample_weight=sample_weight, multioutput=multioutput, squared=squared
        ))

        rmse_d = np.sqrt(mean_squared_error(
            references_d, predictions_d, sample_weight=sample_weight, multioutput=multioutput, squared=squared
        ))


        return {"rmse_valence": rmse_v,
                "rmse_arousal": rmse_a,
                "rmse_dominance": rmse_d
                }
    

    