import torch
import numpy as np
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer
from data_prep.IEMOCAP.dataset import IEMOCAPCollator
from data_prep.MELD.dataset import MELDCollator
from data_prep.ESD.dataset import ESDCollator
from data_prep.ENTERFACE.dataset import ENTCollator
from data_prep.EMODB.dataset import EMODBCollator
from data_prep.EMOVO.dataset import EMOVOCollator
from transformers.trainer_pt_utils import (nested_concat, 
                                           DistributedTensorGatherer,
                                           SequentialDistributedSampler)
from transformers.trainer_utils import (has_length, 
                                        EvalLoopOutput,
                                        EvalPrediction,
                                        denumpify_detensorize)
from typing import List, Union
from transformers.deepspeed import deepspeed_init
from transformers.utils import is_torch_tpu_available, logging
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
from torch.utils.data import WeightedRandomSampler




train_batch_size = 4
num_workers = 6
dev_batch_size  = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.get_logger(__name__)
collator_dict = {
                 "iemocap": IEMOCAPCollator,
                 "meld": MELDCollator,
                 "esd" : ESDCollator,
                 "ent" : ENTCollator,
                 "emovo": EMOVOCollator,
                 "emodb": EMODBCollator
                }



class CustomTrainer(Trainer):
    def __init__(self,
                 *args,
                 path,
                 data_name,
                 train_batch_size = 4,
                 weighted_sampler = False,
                 do_normalize=True,
                 continuous = False,
                 multi = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.data_name = data_name
        self.weighted_sampler = weighted_sampler
        self.train_batch_size = train_batch_size

        if self.data_name == "iemocap":
            self.collate_fn = IEMOCAPCollator(path, 
                                    do_normalize=do_normalize,
                                    continuous=continuous,
                                    multi= multi)
        elif self.data_name == "meld":
            self.collate_fn = MELDCollator(path)
        elif self.data_name == "esd":
            self.collate_fn = ESDCollator(path)
        elif self.data_name == "ent":
            self.collate_fn = ENTCollator(path)
        elif self.data_name == "emovo":
            self.collate_fn = EMOVOCollator(path)
        elif self.data_name == "emodb":
            self.collate_fn = EMODBCollator(path)
        
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        sampler = None
        if self.data_name == "meld" and self.weighted_sampler:
            meld_df = train_dataset.df
            counts = meld_df[meld_df["Emotion"].isin(["neutral", 
                                    "joy",
                                    "anger",
                                    "sadness"])]["Emotion"].value_counts()
            weight = 1. / counts.values
            samples_weight = torch.tensor([weight[train_dataset[i][-1]] for i in range(len(train_dataset))])
            sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(train_dataset), replacement=True)    
        if sampler:
            return DataLoader(
                    train_dataset, 
                    sampler=sampler,
                    batch_size=self.train_batch_size,
                    num_workers=num_workers,
                    collate_fn=self.collate_fn
                )
        return DataLoader(
                                train_dataset, 
                                batch_size=self.train_batch_size,
                                shuffle=True, 
                                num_workers=num_workers,
                                collate_fn=self.collate_fn
                            )

    def get_eval_dataloader(self, eval_dataset) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return  DataLoader(
                                eval_dataset, 
                                batch_size=dev_batch_size,
                                shuffle=True, 
                                num_workers=num_workers,
                                collate_fn=self.collate_fn
                            )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        if test_dataset is None and self.test_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        test_dataset = test_dataset if test_dataset is not None else self.test_dataset
        return  DataLoader(
                                test_dataset, 
                                batch_size=dev_batch_size,
                                shuffle=False, 
                                num_workers=num_workers,
                                collate_fn=self.collate_fn
                            )