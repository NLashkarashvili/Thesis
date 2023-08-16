import numpy as np
import datasets
from datasets import Metric

_DESCRIPTION = """"""
_CITATION = """"""
_KWARGS_DESCRIPTION = """"""



class CCCMetric(Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float32")),
                    "references": datasets.Sequence(datasets.Value("float32")),
                }
            ),
            reference_urls=["https://github.com/stylianos-kampakis/supervisedPCA-Python/blob/master/Untitled.py"],
        )
    def _compute(self, predictions, references):
        predictions = np.array(predictions)
        references = np.array(references)
        predictions_v, references_v = predictions[:, 0], references[:, 0]
        predictions_a, references_a = predictions[:, 1], references[:, 1]
        predictions_d, references_d = predictions[:, 2], references[:, 2]
        # print(predictions_a.shape)
        
        return {"concordance_correlation_coefficient_valence": float(concordance_correlation_coefficient(predictions_v, references_v)),
                "concordance_correlation_coefficient_arousal": float(concordance_correlation_coefficient(predictions_a, references_a)),
                "concordance_correlation_coefficient_dominance": float(concordance_correlation_coefficient(predictions_d, references_d)),
                }
    
    
    

def concordance_correlation_coefficient(y_pred, y_true):
    """Concordance correlation coefficient.
    Modified from https://github.com/stylianos-kampakis/supervisedPCA-Python/blob/master/Untitled.py
    """

    cor=np.corrcoef(y_true,y_pred)[0][1]
    
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)
    
    var_true=np.var(y_true)
    var_pred=np.var(y_pred)
    
    sd_true=np.std(y_true)
    sd_pred=np.std(y_pred)
    
    numerator=2*cor*sd_true*sd_pred
    
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return numerator/denominator

    