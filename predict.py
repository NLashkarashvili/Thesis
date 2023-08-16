import os, torch, copy
import argparse
import numpy as np
from torch.utils.data import random_split
from transformers import TrainingArguments
from transformers import HubertModel, HubertConfig, Wav2Vec2FeatureExtractor
from classifier_models.foundation_models.custom_wav2vec2 import AdaWav2Vec2Config
from custom_trainer import CustomTrainer
from data_prep.IEMOCAP.dataset import IEMOCAPDataset
import data_prep.IEMOCAP.dataset as dst
import json
from data_prep.ESD.dataset import ESDDataset
from custom_metrics import compute_metrics_acc, compute_metrics_regression, compute_metrics_multi
from classifier_models.foundation_models.hubert_custom import HubertForSequenceClassification
from classifier_models.foundation_models.custom_wav2vec2 import (Wav2Vec2ForSER)
from classifier_models.foundation_models.custom_wavlm import WavLMForSequenceClassification, AdaWavLMConfig
from meld_loading import meld_load_splits
from esd_loading import esd_load_splits
from ent_loading import ent_load_splits
from emovo_loading import emovo_load_splits
from emodb_loading import emodb_load_splits



path_dict = {"wavlm_base": "microsoft/wavlm-base",
             "wav2vec2_base": "facebook/wav2vec2-base",
             "wavlm_base_plus": "microsoft/wavlm-base-plus",
             "wav2vec2_base_xlsr": "facebook/wav2vec2-large-xlsr-53",
}

config_dict = {
                "wavlm_base" : AdaWavLMConfig,
                "wav2vec2_base": AdaWav2Vec2Config,
                "wavlm_base_plus" : AdaWavLMConfig,
                "wav2vec2_base_xlsr": AdaWav2Vec2Config
              }


model_dict = {"wavlm": WavLMForSequenceClassification,
              "wav2vec2_base": Wav2Vec2ForSER,
              "wav2vec2_base_xlsr": Wav2Vec2ForSER
} 


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="wav2vec2_base")
    parser.add_argument("--adapter_name", type=str, default="superb")
    parser.add_argument("--scripted", type=bool, default=False)
    parser.add_argument("--improvised", type=bool, default=False)
    parser.add_argument("--label_cat", type=str, default="4way")
    parser.add_argument("--fold_id", type=int, default=1)
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--data_name", type=str, default='iemocap')
    parser.add_argument("--only_english", type=str, default=True)
    parser.add_argument("--only_mandarin", type=str, default=False)
    parser.add_argument("--nclasses", type=int, default=4)
    parser.add_argument("--sets", type=bool, default=False)

    
    
    args = parser.parse_args()
    return args


args = get_arguments()
label_cat = args.label_cat
model_name = args.model_name
adapter_name = args.adapter_name
ix = args.fold_id
data_name = args.data_name
# learning_rate = 0.0005 if adapter_name != "fine_tune" else 0.0001
# if adapter_name in ["transformers", "last_block"]:
#     learning_rate=0.001
# if adapter_name == "fine_tune":
#     learning_rate=5e-5
# if "sure_lora" in adapter_name:
    # learning_rate = 1e-3
path = path_dict[model_name]
# do_normalize=True


if "2proj" in label_cat:
    data_dir = f"/home/nl438/rds/hpc-work/PROJECT/custom_pipeline/data_prep/custom_split/multi_4way"
if "cont_4way" in label_cat:
    data_dir = f"/home/nl438/rds/hpc-work/PROJECT/custom_pipeline/data_prep/custom_split/cont_4way"
if "scripted_all" in label_cat:
    data_dir = f"/home/nl438/rds/hpc-work/PROJECT/custom_pipeline/data_prep/custom_split/4way"
if "improvised" in label_cat:
    data_dir = f"/home/nl438/rds/hpc-work/PROJECT/custom_pipeline/data_prep/custom_split/4way"
data_dir = f"/home/nl438/rds/hpc-work/PROJECT/custom_pipeline/data_prep/custom_split/cont_label_multi"


fold = f"Fold{ix}"
test_path = os.path.join(data_dir, fold, 'test.json')
# test_path = os.path.join(data_dir, fold, 'test.json')
five_way = "5way" in label_cat 
scripted = args.scripted
improvised = args.improvised

if label_cat=="cont_label_multi":
    label_cat = "multi_5way"

data_path = args.data_path
# scripted = True

# if scripted:
    # data_path = 'scripted'
# elif improvised:
    # data_path = 'improvised'
def scr_or_imp():
    if scripted:
        return "_scripted"
    elif improvised:
        return  "_improvised"
    else:
        return ''


#file paths
model_path = f"/home/nl438/rds/hpc-work/PROJECT/{label_cat}_logs/{data_path}/{model_name}/{adapter_name}/\
fold{ix}/logs_transformers/pytorch_model.bin"
training_args_path = f"/home/nl438/rds/hpc-work/PROJECT/{label_cat}_logs/{data_path}/{model_name}/{adapter_name}/\
fold{ix}/logs_transformers/training_args.bin"
config_path = f"/home/nl438/rds/hpc-work/PROJECT/{label_cat}_logs/{data_path}/{model_name}/{adapter_name}/\
fold{ix}/logs_transformers/config.json"
predictions_path = os.path.dirname(config_path) + f"/predictions{scr_or_imp()}{data_name}_cont.json"



#load model
continuous = False
multi = False
model_weights = torch.load(model_path)
training_args = torch.load(training_args_path)
config = config_dict[model_name].from_json_file(config_path)
if "cont" in label_cat:
    continuous = True
    compute_metrics = compute_metrics_regression
elif 'multi' in label_cat:
    multi = True
    compute_metrics = compute_metrics_multi

else:
    compute_metrics = compute_metrics_acc
sequence_model = model_dict[model_name].from_pretrained(path, config=config)
if data_name == "iemocap":
    if label_cat!="4way":
        test_path = test_path.replace(label_cat, "4way")
    test_dataset = IEMOCAPDataset(test_path, 
                                True, 
                                five_way=five_way,
                                continuous=continuous,
                                scripted=scripted,
                                improvised=improvised,
                                multi=multi)
elif data_name == "meld":
    _, _, test_dataset = meld_load_splits()
elif data_name == "esd":
    _, _, test_dataset = esd_load_splits(four_way=args.nclasses!=5,
                                         only_english=args.only_english,
                                         only_mandarin=args.only_mandarin)
elif data_name == "ent":
    if not args.sets:
        test_dataset = ent_load_splits()
    else:
        _, _, test_dataset = ent_load_splits(sets=args.sets,
                                             iemocap_only=False
                                             )
        print(len(test_dataset))

elif data_name == "emovo":
    _, _, test_dataset = emovo_load_splits()

elif data_name == "emodb":
    _, _, test_dataset = emodb_load_splits()



path = config._name_or_path
# sequence_model = Wav2Vec2ForSequenceClassification.from_pretrained(path, config=config)
for key in list(model_weights.keys()):
    if key == "regressor.weight":
        model_weights[key.replace('regressor.weight', 'regression.weight')] = model_weights.pop(key)
    if key == "regressor.bias":
        model_weights[key.replace('regressor.bias', 'regression.bias')] = model_weights.pop(key)

sequence_model.load_state_dict(model_weights)
print(compute_metrics)
trainer = CustomTrainer(sequence_model,
                        data_name=args.data_name,
                        path=path,
                        do_normalize=True,
                        continuous=continuous,
                        multi=multi,
                        args=training_args,
                        compute_metrics=compute_metrics)
# print(test_path)
# print(len(test_dataset))
predictions = trainer.predict(test_dataset)
print(predictions.metrics)
results = {}
# try:
#     id2label = test_dataset.idx2emotion
# except:
id2label = {v:k for k, v in test_dataset.class_dict.items()}
print(test_dataset.class_dict)


if multi: 
    results["predictions_regression"] = [i.tolist() for i in predictions.predictions[0]]
    results["predictions_classifier"] = [id2label[ix] for ix in np.argmax(predictions.predictions[1], axis=1).tolist()]
    results["labels_regression"] = [i.tolist() for i in predictions.label_ids[:, :3]]
    results["labels_classification"] = [id2label[ix] for ix in predictions.label_ids[:, 3].tolist()]
elif continuous:
    results["predictions"] = [i.tolist() for i in predictions.predictions]
    results["labels"] = [i.tolist() for i in predictions.label_ids]
else:
    results["predictions"] = [id2label[ix] for ix in np.argmax(predictions.predictions, axis=1).tolist()]
    results["labels"] = [id2label[ix] for ix in predictions.label_ids.tolist()]

for k, v in predictions.metrics.items():
    results[k] = v
print(predictions_path)
with open(predictions_path, "w", encoding="utf8") as writer:
    json.dump(results, writer, indent=4)
