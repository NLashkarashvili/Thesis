import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from data_prep.EMOVO.dataset import EMOVODataset


def emovo_load_splits(iemocap_only : bool = True):
                    #  set = "train",
                #  iemocap_only = True
        data = EMOVODataset(iemocap_only=iemocap_only,
                                   set="train")
        torch.manual_seed(0)
        np.random.seed(0)

        train_idx, validation_idx = train_test_split(np.arange(len(data)),
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=data.glabels)

        train_dataset = Subset(data, train_idx)
        dev_dataset = Subset(data, validation_idx)
        test_dataset = EMOVODataset(iemocap_only=iemocap_only,
                                   set="test")
 

        return train_dataset, dev_dataset, test_dataset