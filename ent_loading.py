import os, torch
from data_prep.ENTERFACE.dataset import ENTDataset
import numpy as np


def ent_load_splits(iemocap_only : bool = True,
                    all_esd : bool = False,
                    sets : bool = False):
    if sets:
        train_dataset = ENTDataset(iemocap_only=iemocap_only,
                                   all_esd=all_esd,
                                   sets=True,
                                   train=True)
        
        dev_dataset = ENTDataset(iemocap_only=iemocap_only,
                                all_esd=all_esd,
                                sets=True,
                                val=True)

        test_dataset = ENTDataset(iemocap_only=iemocap_only,
                                all_esd=all_esd,
                                sets=True,
                                test=True)     

        return train_dataset, dev_dataset, test_dataset
    else:
        test_dataset = ENTDataset(iemocap_only=iemocap_only,
                                  all_esd=all_esd
                                  )
        return test_dataset