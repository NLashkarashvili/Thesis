import os, torch
from data_prep.IEMOCAP.dataset import IEMOCAPDataset
from torch.utils.data import random_split
def iemocap_load_splits(args,
                iemocap_dir): 
    torch.manual_seed(0)
    five_way = args.nclasses == 5
    fold = f"Fold{args.fold}" #"Session1"
    meta_dir = os.path.join(iemocap_dir, args.meta_dir)
    valid_ratio = args.valid_ratio
    train_path = os.path.join(meta_dir, fold, 'train.json')
    test_path = os.path.join(meta_dir, fold, 'test.json')
    print(f'Training path: {train_path}, Testing Path: {test_path}')
    multi=False
    continuous = False
    if args.task == "multi":
        multi=True
    if args.task == "regression":
        continuous = True
    improvised = args.improvised
    scripted = args.scripted
    improvised2scripted = args.improvised2scripted
    scripted2improvised = args.scripted2improvised
    torch.manual_seed(0)
    dataset = IEMOCAPDataset(train_path, 
                            True, 
                            five_way=five_way, 
                            continuous=continuous, 
                            multi=multi,
                            scripted=(scripted or improvised2scripted),
                            improvised=(improvised or scripted2improvised))
    trainlen = int((1 - valid_ratio) * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    test_dataset = IEMOCAPDataset(test_path, 
                                True, 
                                five_way=five_way, 
                                continuous=continuous, 
                                multi=multi,
                                scripted=(scripted or improvised2scripted),
                                improvised=(improvised or scripted2improvised))
    train_dataset, dev_dataset = random_split(dataset, lengths)
    
    return train_dataset, dev_dataset, test_dataset