import os, torch
from data_prep.MELD.dataset import MELDDataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def meld_load_splits(get_class_weights=False):
    meld_train_path = "/home/nl438/rds/hpc-work/PROJECT/data/MELD/MELD.Raw/train_audios"
    meld_test_path = "/home/nl438/rds/hpc-work/PROJECT/data/MELD/MELD.Raw/test_audios"
    meld_valid_path = "/home/nl438/rds/hpc-work/PROJECT/data/MELD/MELD.Raw/dev_audios"

    train_df_path = "/home/nl438/rds/hpc-work/PROJECT/data/MELD/MELD.Raw/train_sent_emo.csv"
    dev_df_path = "/home/nl438/rds/hpc-work/PROJECT/data/MELD/MELD.Raw/dev_sent_emo.csv"
    test_df_path = "/home/nl438/rds/hpc-work/PROJECT/data/MELD/MELD.Raw/test_sent_emo.csv"
    
    train_dataset = MELDDataset(meld_train_path, train_df_path)
    dev_dataset = MELDDataset(meld_valid_path, dev_df_path)
    test_dataset = MELDDataset(meld_test_path, test_df_path)

    class_weights=list(compute_class_weight(class_weight='balanced',
                                    classes=np.unique([0, 1, 2, 3]),
                                    y=np.array([train_dataset[i][-1] for i in range(len(train_dataset))])))
    
    if get_class_weights:
        return train_dataset, dev_dataset, test_dataset, class_weights
    
    return train_dataset, dev_dataset, test_dataset