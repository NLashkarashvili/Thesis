import numpy as np
from datasets import load_metric

root_dir = "/home/nl438/rds/hpc-work/PROJECT/custom_pipeline/"
def compute_metrics_acc(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1) 
    metric = load_metric("accuracy")                         
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def compute_metrics_regression(eval_pred):
    predictions = eval_pred.predictions
    references = eval_pred.label_ids
    metric1 = load_metric(f"{root_dir}ccc.py")
    metric2 = load_metric(f"{root_dir}rmse.py")

    return {"ccc": metric1.compute(predictions=predictions, references=references),
            "rmse": metric2.compute(predictions=predictions, references=references)}

def compute_metrics_multi(eval_pred):
    
    predictions_reg = eval_pred.predictions[0]
    predictions_cl = eval_pred.predictions[1]

    print(predictions_reg)
    print(predictions_cl)

    references_reg = eval_pred.label_ids[:, :3]
    references_cl = eval_pred.label_ids[:, 3]
    predictions_cl = np.argmax(predictions_cl, axis=1)   

    
    metric1 = load_metric(f"{root_dir}ccc.py")
    metric2 = load_metric(f"{root_dir}rmse.py")
    metric3 = load_metric("accuracy")

    return {"ccc": metric1.compute(predictions=predictions_reg, references=references_reg),
            "rmse": metric2.compute(predictions=predictions_reg, references=references_reg),
            "accuracy": metric3.compute(predictions=predictions_cl, references=references_cl)}    
    
    
    

# def compute_metrics_rmse(eval_pred):
#     predictions = eval_pred.predictions
#     references = eval_pred.label_ids
#     metric = load_metric("rmse.py")
#     return metric.compute(predictions=predictions, references=references)