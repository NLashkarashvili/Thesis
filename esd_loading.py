import os, torch
from data_prep.ESD.dataset import ESDDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def esd_load_splits(only_english : bool = True,
                     only_mandarin : bool = False,
                     four_way : bool = True,
                     downsample: bool = False):
    data_path_dict = { "english_4way" : { "train" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/esd_english_train.pt",
                                          "test" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/esd_english_test.pt",
                                          "evaluation" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/esd_english_eval.pt" 
                                        },
                        "mandarin_4way":{ "train" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/esd_mandarin_train.pt",
                                          "test" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/esd_mandarin_test.pt",
                                          "evaluation" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/esd_mandarin_eval.pt" 
                                        },
                        "all_4way" :    { "train" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/esd_all_train.pt",
                                          "test" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/esd_all_test.pt",
                                          "evaluation" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/esd_all_eval.pt" 
                                        },
                        "english_5way" : { "train" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/5way_esd_english_train.pt",
                                          "test" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/5way_esd_english_test.pt",
                                          "evaluation" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/5way_esd_english_eval.pt" 
                                        },
                        "mandarin_5way":{ "train" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/5way_esd_mandarin_train.pt",
                                          "test" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/5way_esd_mandarin_test.pt",
                                          "evaluation" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/5way_esd_mandarin_eval.pt" 
                                        },
                        "all_5way" :    { "train" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/5way_esd_all_train.pt",
                                          "test" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/5way_esd_all_test.pt",
                                          "evaluation" : "/home/nl438/rds/hpc-work/PROJECT/data/ESD/5way_esd_all_eval.pt" 
                                        }
                                    
                      }
    if four_way:
        s = "4"
    else:
        s = "5"
    if only_english:

        train_dataset = torch.load(data_path_dict[f"english_{s}way"]["train"])
        dev_dataset = torch.load(data_path_dict[f"english_{s}way"]["evaluation"])
        test_dataset = torch.load(data_path_dict[f"english_{s}way"]["test"])

    elif only_mandarin:

        train_dataset = torch.load(data_path_dict[f"mandarin_{s}way"]["train"])
        dev_dataset = torch.load(data_path_dict[f"mandarin_{s}way"]["evaluation"])
        test_dataset = torch.load(data_path_dict[f"mandarin_{s}way"]["test"])

    else:

        train_dataset = torch.load(data_path_dict[f"all_{s}way"]["train"])
        dev_dataset = torch.load(data_path_dict[f"all_{s}way"]["evaluation"])
        test_dataset = torch.load(data_path_dict[f"all_{s}way"]["test"])
        

    if downsample:
      torch.manual_seed(0)
      np.random.seed(0)

      train_idx, validation_idx = train_test_split(np.arange(len(train_dataset)),
                                                  test_size=0.3,
                                                  shuffle=True,
                                                  stratify=train_dataset.glabels)

      train_dataset = Subset(train_dataset, validation_idx)

    return train_dataset, dev_dataset, test_dataset