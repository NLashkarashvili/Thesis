import torch
import torch.nn as nn

class CCCLoss(nn.Module):
    def __init__(self, ):
        super(CCCLoss, self).__init__()
        

    def forward(self, predictions, targets):
        # Modified from https://github.com/stylianos-kampakis/supervisedPCA-Python/blob/master/Untitled.py
        tmp = torch.stack((targets,predictions),dim=0)
        # cor=torch.corrcoef(tmp)[0][1]
        cor = torch.cov(tmp)[0][1]
        
        mean_true=torch.mean(targets)
        mean_pred=torch.mean(predictions)
        
        var_true=torch.var(targets)
        var_pred=torch.var(predictions)
        
        # sd_true=torch.std(targets)
        # sd_pred=torch.std(predictions)
        
        numerator=2*cor#*sd_true*sd_pred
        denominator=var_true+var_pred+(mean_true-mean_pred)**2

        # print(tmp)
        # print(predictions)
        # # print(var_pred)
        # # print(mean_pred)
        # # print(sd_pred)
        # # print(denominator)
        # # print(numerator)
        return 1 - numerator/(denominator + 1e-7)