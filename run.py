import os, torch, copy
import ast
from data_prep.ESD.dataset import ESDDataset
from esd_loading import esd_load_splits
from iemocap_loading import iemocap_load_splits
from meld_loading import meld_load_splits
from ent_loading import ent_load_splits
from emovo_loading import emovo_load_splits
from emodb_loading import emodb_load_splits
from torch.utils.data import random_split
from transformers import TrainingArguments
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from custom_trainer import CustomTrainer
from data_prep.IEMOCAP.dataset import IEMOCAPDataset
from models import UpstreamDownstreamModel
from custom_metrics import (compute_metrics_acc, 
                            compute_metrics_regression,
                            compute_metrics_multi
                            )
from classifier_models.foundation_models.custom_wavlm import (AdaWavLMConfig,
                                                              WavLMForSequenceClassification)
from classifier_models.foundation_models.custom_wav2vec2 import (Wav2Vec2ForSER,
                                                                 AdaWav2Vec2Config)
import argparse
import data_prep.IEMOCAP.dataset as dst # TO-DO save feature extractor after training
from transformers import Wav2Vec2Model
from adapter_name_generator import get_adapter_name
from torch.utils.data import ConcatDataset


iemocap_dir = "/home/nl438/rds/hpc-work/PROJECT/custom_pipeline/data_prep/custom_split/"
meld_dir = "/home/nl438/rds/hpc-work/PROJECT/data/MELD/MELD.Raw"


path = "facebook/hubert-base-ls960"

path_dict = {"wavlm_base": "microsoft/wavlm-base",
             "wavlm_base_plus": "microsoft/wavlm-base-plus",
             "wav2vec2_base": "facebook/wav2vec2-base",
             "wav2vec2_base_xlsr": "facebook/wav2vec2-large-xlsr-53",
             }


config_dict = {
                "wavlm" : AdaWavLMConfig,
                "wav2vec2": AdaWav2Vec2Config,
              }


model_dict = {"wavlm": WavLMForSequenceClassification,
                   "wav2vec2": Wav2Vec2ForSER
}     

feature_dict = {
                "wavlm": Wav2Vec2FeatureExtractor,
                "wav2vec2": Wav2Vec2FeatureExtractor
               }


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="iemocap", help="dataset name")
    parser.add_argument("--task", type=str, default="classification", 
                        choices=["classification", "regression","multi"], 
                        help="type of model classification/regression/multi-task")    
    parser.add_argument("--cnn_multiplier", type=bool, default=False)
    parser.add_argument("--trans_multiplier", type=bool, default=False)
    parser.add_argument("--encoder_tuning", type=bool, default=False, 
                        help="whether to tune encoder or not")
    parser.add_argument("--use_vert_layer_sum", type=bool, default=False, 
                        help="whether to use vert_layer_sum")
    parser.add_argument("--nclasses", type=int, default=4, help="number of classes")
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--classifier_proj_size", type=int, default=256)
    parser.add_argument("--adapter_dim", type=int, default=64)
    parser.add_argument("--every_hidden_state", type = bool, default = False)
    parser.add_argument("--adapter_name",  type=str, default='none', 
                        help="adapter category")
    parser.add_argument("--add_adapter", type=bool, default=False, 
                        help="include/exclude adapter")
    parser.add_argument("--prefix_tuning", type=bool, default=False)
    parser.add_argument("--lora_adapter", type=bool, default=False, 
                        help="use lora as adapter")
    parser.add_argument("--sure_lora", type=bool, default=False, 
                        help="use SURE benchmark settings for LoRA")
    parser.add_argument("--mh_adapter", type=bool, default=False, 
                        help="use adapter in multi-head attention")
    parser.add_argument("--output_adapter", type=bool, 
                        default=False, help="bottleneck (houlsby adapter) after feedforward")
    parser.add_argument("--continuous", type=bool, default=False,
                        help="valence-arousal-dominance prediction or not")
    parser.add_argument("--bitfit", type=bool, default=False,
                        help="bias tuning")
    parser.add_argument("--foundation_model_name", type=str, default="wav2vec2_base",
                        # choices=["wav2vec2_base", "wavlm_base"]
                        )
    parser.add_argument("--meta_dir", type=str, default="4way",
                        help="data split directory")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--dev_batch_size", type=int, default=4)    
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--valid_ratio", type=float, default=.2,
                        help="validation ratio from IEMOCAP training data")
    parser.add_argument("--improvised", type=bool, default=False)
    parser.add_argument("--scripted", type=bool, default=False)
    parser.add_argument("--improvised2scripted", type=bool, default=False)
    parser.add_argument("--scripted2improvised", type=bool, default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_accumulation_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--compute_metrics", type=str, default="accuracy")
    parser.add_argument("--do_predict", type=bool, default=True)
    parser.add_argument("--use_weighted_layer_sum", type=bool, default=False)
    parser.add_argument("--tune_last_block", type=bool, default=False)
    parser.add_argument("--fine_tune", type=bool, default=False)
    parser.add_argument("--do_normalize", type=bool, default=True)
    parser.add_argument("--num_predict", type=int, default=3,
                        help="regression model prediction values")
    parser.add_argument("--conv_lora", type=bool, default=False,
                        help="add conv_lora to feature encoder")
    parser.add_argument("--bottleneck_cnn", type=bool, default=False,
                        help="add bottleneck cnn to feature encoder")
    parser.add_argument("--depthwise_cnn", type=bool, default=False,
                        help="add depthwise cnn to feature encoder")
    parser.add_argument("--regression_proj_size", type=int, default=256)
    parser.add_argument("--multi_proj_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--adapter_layer_list", type=str, default=None)
    parser.add_argument("--adapter_enc_layer_list", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False)
    parser.add_argument("--freeze_lora_A", type=bool, default=False)
    parser.add_argument("--freeze_lora_B", type=bool, default=False)
    parser.add_argument("--freeze_adapter", type=bool, default=False)
    parser.add_argument("--freeze_gate", type=bool, default=False)
    parser.add_argument("--last_3_enc", type=bool, default=False)
    parser.add_argument("--n_enc_adapters", type=int, default=3)
    parser.add_argument("--conv_r", type=int, default=8)
    parser.add_argument("--bconv_d", type=int, default=8)
    parser.add_argument("--weighted_sampler", type=bool, default=False)
    parser.add_argument("--use_different_projectors", type=bool, default=False)
    parser.add_argument("--use_cat_in_vad", type=bool, default=False)
    parser.add_argument("--use_vad_in_cat", type=bool, default=False)
    parser.add_argument("--get_class_weights", type=bool, default=False)
    parser.add_argument("--only_english", type=bool, default=True)
    parser.add_argument("--only_mandarin", type=bool, default=False)
    parser.add_argument("--language", type=str, default='english')
    parser.add_argument("--iemocap2esd", type=bool, default=False)
    parser.add_argument("--esd2iemocap", type=bool, default=False)
    parser.add_argument("--reg_loss", type=str, default='rmse')
    parser.add_argument("--scripted_all", type=bool, default=False)
    parser.add_argument("--improvised_all", type=bool, default=False)
    parser.add_argument("--enterface_in_train", type=bool, default=False)
    parser.add_argument("--esd_classes", type=int, default=4)
    parser.add_argument("--scripted2improvised_all", type=bool, default=False)
    parser.add_argument("--iemocap2cl", type=bool, default=False)
    parser.add_argument("--all2cl", type=bool, default=False)
    parser.add_argument("--downsample", type=bool, default=False)
    parser.add_argument("--downsample_full2cl", type=bool, default=False)
    parser.add_argument("--only_iemocap", type=bool, default=False)
    parser.add_argument("--simple2complex", type=bool, default=False)



    args = parser.parse_args()
    return args

args = get_arguments()


data_name = args.data_name
multi=False
continuous = False
if args.task == "multi":
    multi=True
if args.task == "regression":
    continuous = True

# args.only_english = ast.literal_eval(args.only_english)
# args.only_mandarin = ast.literal_eval(args.only_mandarin)
if args.language == "english":
    args.only_english = True
    lang = "english_"
elif args.language == "mandarin":
    args.only_english = False
    args.only_mandarin = True
    lang = "mandarin_"
else:
    args.only_english = False
    args.only_mandarin = False
    lang = ""

    
#improvised / scripted arguments

improvised = args.improvised
scripted = args.scripted
improvised2scripted = args.improvised2scripted
scripted2improvised = args.scripted2improvised
freeze_lora_A = args.freeze_lora_A
freeze_lora_B = args.freeze_lora_B
freeze_adapter = args.freeze_adapter
freeze_gate = args.freeze_gate
class_weights = None   


if data_name == "iemocap":
    torch.manual_seed(0)
    data_dir = iemocap_dir
    train_dataset, dev_dataset, test_dataset = iemocap_load_splits(args,
                                                                   data_dir)
    meta_dir = args.meta_dir
elif data_name == "meld":
    get_class_weights = args.get_class_weights
    if get_class_weights:
        train_dataset, dev_dataset, test_dataset, class_weights = meld_load_splits(get_class_weights=get_class_weights)
    else:
        train_dataset, dev_dataset, test_dataset = meld_load_splits(get_class_weights=get_class_weights)     
    meta_dir = f"meld_{args.nclasses}way"
elif data_name == "esd":
    four_way = args.meta_dir == "4way"
    only_english = args.only_english
    only_mandarin = args.only_mandarin
    
    train_dataset, dev_dataset, test_dataset = esd_load_splits(only_english=only_english,
                                                               only_mandarin=only_mandarin,
                                                               four_way=four_way,
                                                               downsample=args.downsample)     
    if only_english:
        meta_dir = f"esd_english_{args.nclasses}way"
    elif only_mandarin:
        meta_dir = f"esd_mandarin_{args.nclasses}way"
    else:
        meta_dir = f"esd_{args.nclasses}way"

elif data_name == "emovo":
    train_dataset, dev_dataset, test_dataset = emovo_load_splits()
    meta_dir = f"emovo_4way"

elif data_name == "emodb":
    train_dataset, dev_dataset, test_dataset = emodb_load_splits()
    meta_dir = f"emodb_4way"

if data_name == "ent":
    train_dataset, dev_dataset, test_dataset = ent_load_splits(iemocap_only=args.only_iemocap,
                                                                 sets=True)
    meta_dir = f"ent_6way"
    args.nclasses = 6
    

if args.scripted_all:
    four_way = args.nclasses == 4
    only_english = args.only_english
    only_mandarin = args.only_mandarin
    train_esd, dev_esd, test_esd = esd_load_splits(only_english=only_english,
                                                   only_mandarin=only_mandarin,
                                                   four_way=four_way,
                                                   downsample=args.downsample)
    meta_dir = "scripted_all"
    train_dataset = ConcatDataset([train_dataset, train_esd])
    dev_dataset = ConcatDataset([dev_dataset, dev_esd])

elif args.improvised_all:
    meta_dir = "improvised_iem"
    if args.enterface_in_train:
        train_ent, dev_ent, test_ent = ent_load_splits(sets=True)
        train_dataset = ConcatDataset([train_dataset, train_ent])
        dev_dataset = ConcatDataset([dev_dataset, dev_ent])
        meta_dir = "improvised_all"








device = "cuda" if torch.cuda.is_available() else "cpu"
if args.adapter_layer_list:
    args.adapter_layer_list = [int(i) for i in args.adapter_layer_list.split(',')]
if args.adapter_enc_layer_list:
    args.adapter_enc_layer_list = [int(i) for i in args.adapter_enc_layer_list.split(',')]
print(args.adapter_layer_list)


#model
path = path_dict[args.foundation_model_name]
# meta_dir = data_dir + "/" + args.meta_dir
model_name = args.foundation_model_name.split('_')[0]
config_class = config_dict[model_name]


config = config_class(
                    _name_or_path = path,
                    class_weights = class_weights,
                    use_weighted_layer_sum = args.use_weighted_layer_sum,
                    add_adapter = args.add_adapter,
                    use_different_projectors = args.use_different_projectors,
                    use_cat_in_vad = args.use_cat_in_vad,
                    use_vad_in_cat = args.use_vad_in_cat,
                    prefix_tuning = args.prefix_tuning,
                    lora_adapter = args.lora_adapter,
                    sure_lora = args.sure_lora,
                    mh_adapter=args.mh_adapter,
                    adapter_name = args.adapter_name,
                    output_adapter=args.output_adapter,
                    adapter_layer_list=args.adapter_layer_list,
                    lora_rank=args.lora_rank,
                    adapter_enc_layer_list = args.adapter_enc_layer_list,
                    keys_to_ignore_at_inference = ["hidden_states", "attentions"],
                    num_labels = args.nclasses,
                    classifier_proj_size = args.classifier_proj_size,
                    label2id = test_dataset.class_dict,
                    regression_proj_size = args.regression_proj_size,
                    multi_proj_size = args.multi_proj_size,
                    num_predict = args.num_predict,
                    task = args.task,
                    adapter_dim = args.adapter_dim,
                    cnn_multiplier = args.cnn_multiplier,
                    use_vert_layer_sum = args.use_vert_layer_sum,
                    trans_multiplier = args.trans_multiplier,
                    conv_lora = args.conv_lora,
                    bottleneck_cnn = args.bottleneck_cnn,
                    depthwise_cnn = args.depthwise_cnn,
                    last_3_enc = args.last_3_enc,
                    every_hidden_state = args.every_hidden_state,
                    n_enc_adapters = args.n_enc_adapters,
                    conv_r = args.conv_r,
                    bconv_d = args.bconv_d,
                    reg_loss = args.reg_loss,
                    input_dim = args.input_dim
                           )


feature_extractor = feature_dict[model_name].from_pretrained(path)
if "xlsr" in path:
    tmp_config = Wav2Vec2Model.from_pretrained(path).config.to_dict()
    config_dict = config.to_dict()
    for key in tmp_config:
        if key in config_dict:
            config_dict[key] = tmp_config[key]
    config = AdaWav2Vec2Config(**config_dict)
    config.num_labels = args.nclasses
    print(config)
sequence_model = model_dict[model_name].from_pretrained(path, config=config).to(device)

adapter_name, sequence_model = get_adapter_name(args, sequence_model)
print(adapter_name)
print(args)
print(config.label2id)
#trainer
if args.sure_lora:
    add_str = "sure_lora"
else:
    add_str = ''
    
resume_from_checkpoint = args.resume_from_checkpoint
data_path = ''
if scripted:
    data_path = 'scripted'
elif improvised:
    data_path = 'improvised'
elif improvised2scripted:
    data_path = 'improvised2scripted'
elif scripted2improvised:
    data_path = 'scripted2improvised'   
elif args.iemocap2esd:
    data_path = "iemocap2esd"
elif args.esd2iemocap:
    data_path = "esd2iemocap"
if args.scripted2improvised_all:
    data_path = "scripted_all2improvised"
if args.iemocap2cl:
    data_path = f"iemocap2{args.data_name}"
if args.downsample:
    data_path = "downsample"
if args.downsample and not (args.scripted or args.improvised):
    data_path = "downsample_full"
if args.downsample_full2cl:
    data_path = "downsample_full"

freeze_str = ''
if freeze_lora_A:
    freeze_str += "_fLA"
if freeze_lora_B:
    freeze_str += "_fLB"
if freeze_adapter:
    freeze_str += "_fAd"
if freeze_gate:
    freeze_str += "_fG"
if args.data_name == "meld" and args.weighted_sampler:
    meta_dir += "_" + "sampler"
if args.use_different_projectors:
    meta_dir += "_2proj"
if args.use_cat_in_vad:
    meta_dir += "_cat"
if args.use_vad_in_cat:
    meta_dir += "_reg"

if data_name =="ent" and not args.simple2complex:
    meta_dir += "_ns"

if args.task in ["regression", "multi"]:
    output_dir = f"/home/nl438/rds/hpc-work/PROJECT/{meta_dir}_{args.reg_loss}_logs/{data_path}/{args.foundation_model_name}/{adapter_name+freeze_str}/\
{add_str}{args.lora_rank}/{args.learning_rate}/{args.do_normalize}/fold{args.fold}/logs_transformers"
else:
    output_dir = f"/home/nl438/rds/hpc-work/PROJECT/{meta_dir}_logs/{data_path}/{args.foundation_model_name}/{adapter_name+freeze_str}/\
{add_str}{args.lora_rank}/{args.learning_rate}/{args.do_normalize}/fold{args.fold}/logs_transformers"

print(output_dir)



if improvised2scripted:
    model_path = output_dir.replace("improvised2scripted", "improvised")
    model_path = model_path.replace(adapter_name + freeze_str, adapter_name)
    model_path = f"{model_path}/pytorch_model.bin"
    model_weights = torch.load(model_path)
    sequence_model.load_state_dict(model_weights)
    sequence_model.freeze_exclude_prompt(freeze_lora_A=freeze_lora_A,
                                         freeze_lora_B=freeze_lora_B,
                                         freeze_adapter=freeze_adapter,
                                         freeze_gate = freeze_gate)
if scripted2improvised and args.improvised_all:
    model_path = output_dir.replace("improvised_all", "scripted_all")
    model_path = model_path.replace(adapter_name + freeze_str, adapter_name)
    model_path = f"{model_path}/pytorch_model.bin"
    model_weights = torch.load(model_path)
    sequence_model.load_state_dict(model_weights)
    sequence_model.freeze_exclude_prompt(freeze_lora_A=freeze_lora_A,
                                         freeze_lora_B=freeze_lora_B,
                                         freeze_adapter=freeze_adapter,
                                         freeze_gate = freeze_gate)
elif scripted2improvised:
    model_path = output_dir.replace("scripted2improvised", "scripted")
    model_path = model_path.replace(adapter_name + freeze_str, adapter_name)
    model_path = f"{model_path}/pytorch_model.bin"
    model_weights = torch.load(model_path)
    sequence_model.load_state_dict(model_weights)
    sequence_model.freeze_exclude_prompt(freeze_lora_A=freeze_lora_A,
                                         freeze_lora_B=freeze_lora_B,
                                         freeze_adapter=freeze_adapter,
                                         freeze_gate = freeze_gate)
if args.scripted2improvised_all:
    model_path = output_dir.replace("improvised_all", "scripted_all")
    model_path = model_path.replace(adapter_name + freeze_str, adapter_name)
    model_path = f"{model_path}/pytorch_model.bin"
    model_weights = torch.load(model_path)
    sequence_model.load_state_dict(model_weights)
    sequence_model.freeze_exclude_prompt(freeze_lora_A=freeze_lora_A,
                                         freeze_lora_B=freeze_lora_B,
                                         freeze_adapter=freeze_adapter,
                                         freeze_gate = freeze_gate)


    
if args.iemocap2esd:
    model_path = output_dir.replace(meta_dir, "4way")
    model_path = model_path.replace(data_path, "")
    model_path = model_path.replace(adapter_name + freeze_str, adapter_name)
    model_path = f"{model_path}/pytorch_model.bin"
    model_weights = torch.load(model_path)
    # sequence_model.load_state_dict(model_weights)
    if args.nclasses == 5:
        model_weights = {k:v for k, v in model_weights.items() if k not in ["classifier.weight", "classifier.bias"]}
    sequence_model.load_state_dict(model_weights,
                                strict=False)
    sequence_model.freeze_exclude_prompt(freeze_lora_A=freeze_lora_A,
                                         freeze_lora_B=freeze_lora_B,
                                         freeze_adapter=freeze_adapter,
                                         freeze_gate = freeze_gate)
if args.esd2iemocap:
    if args.only_english:
        model_path = output_dir.replace(meta_dir, f"esd_{lang}{args.esd_classes}way")
    model_path = model_path.replace(data_path, "")
    model_path = model_path.replace(adapter_name + freeze_str, adapter_name)
    model_path = f"{model_path}/pytorch_model.bin"
    model_weights = torch.load(model_path)    
    sequence_model.freeze_exclude_prompt(freeze_lora_A=freeze_lora_A,
                                         freeze_lora_B=freeze_lora_B,
                                         freeze_adapter=freeze_adapter,
                                         freeze_gate = freeze_gate)
if args.iemocap2cl:
    model_path = output_dir.replace(meta_dir, "4way")
    model_path = model_path.replace(data_path, "")
    model_path = model_path.replace(adapter_name + freeze_str, adapter_name)
    model_path = f"{model_path}/pytorch_model.bin"
    model_weights = torch.load(model_path)
    sequence_model.freeze_exclude_prompt(freeze_lora_A=freeze_lora_A,
                                         freeze_lora_B=freeze_lora_B,
                                         freeze_adapter=freeze_adapter,
                                         freeze_gate = freeze_gate)
    
if args.downsample_full2cl:
    model_path = output_dir.replace(meta_dir, "scripted_all")
    # model_path = model_path.replace(data_path, "")
    model_path = model_path.replace(adapter_name + freeze_str, adapter_name)
    model_path = f"{model_path}/pytorch_model.bin"
    model_weights = torch.load(model_path)
    sequence_model.freeze_exclude_prompt(freeze_lora_A=freeze_lora_A,
                                         freeze_lora_B=freeze_lora_B,
                                         freeze_adapter=freeze_adapter,
                                         freeze_gate = freeze_gate)

if args.simple2complex:
    model_path = output_dir.replace(meta_dir, "4way")
    # model_path = model_path.replace(data_path, "")
    model_path = model_path.replace(adapter_name + freeze_str, adapter_name)
    model_path = f"{model_path}/pytorch_model.bin"
    model_weights = torch.load(model_path)
    if args.nclasses == 6:
        model_weights = {k:v for k, v in model_weights.items() if k not in ["classifier.weight", "classifier.bias"]}
    sequence_model.load_state_dict(model_weights,
                                   strict=False)
    sequence_model.freeze_exclude_prompt(freeze_lora_A=freeze_lora_A,
                                         freeze_lora_B=freeze_lora_B,
                                         freeze_adapter=freeze_adapter,
                                         freeze_gate = freeze_gate)


    
    
    
training_args = TrainingArguments(  output_dir=output_dir,
                                    learning_rate = args.learning_rate,
                                    max_steps=  args.max_steps,
                                    logging_steps=args.logging_steps,
                                    eval_steps=args.eval_steps,
                                    save_steps=args.save_steps,
                                    max_grad_norm=args.max_grad_norm,
                                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                                    eval_accumulation_steps=args.eval_accumulation_steps,
                                    save_strategy=args.save_strategy,
                                    evaluation_strategy=args.evaluation_strategy,
                                    do_predict=args.do_predict, 
                                    load_best_model_at_end = True)

training_args.set_optimizer(name="adamw_torch",
                            learning_rate=args.learning_rate,
                            weight_decay=args.weight_decay)


if args.task == "classification":
    compute_metrics = compute_metrics_acc
elif args.task == "regression":
    compute_metrics = compute_metrics_regression
elif args.task == "multi":
    compute_metrics = compute_metrics_multi



num_params = sum(p.numel() for p in sequence_model.parameters() if p.requires_grad)
print(f"Number of Trainable Parameters {num_params}")
print(path)
trainer = CustomTrainer(sequence_model,
                        weighted_sampler=args.weighted_sampler,
                        data_name=data_name,
                        path=path,
                        do_normalize=args.do_normalize,
                        continuous = continuous,
                        multi=multi,
                        train_batch_size=args.train_batch_size,
                        args = training_args,
                        train_dataset=train_dataset,
                        eval_dataset=dev_dataset,
                        compute_metrics=compute_metrics
                        )
print("Training Started")
trainer.train(resume_from_checkpoint=resume_from_checkpoint)
trainer.save_model()
print("Model Saved")