# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import json, torch
from pathlib import Path
from os.path import join as path_join
from transformers import Wav2Vec2FeatureExtractor
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample




class IEMOCAPDataset(Dataset):
    def __init__(self, 
                 meta_path, 
                 pre_load=True,
                 five_way=False,
                 continuous=False,
                 multi=False,
                 scripted=False,
                 improvised=False):
        self.meta_path = meta_path
        self.pre_load = pre_load
        self.multi = multi
        self.continuous = continuous
        self.scripted = scripted
        self.improvised = improvised
        with open(meta_path, 'r') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        if five_way:
            self.class_dict = {"neu": 0, "hap": 1, 
                               "ang": 2, "sad": 3, "oth": 4}
        else:
            self.class_dict = {"neu": 0, "hap": 1, "ang": 2, "sad": 3}
        self.idx2emotion = {value: key for key, value in self.class_dict.items()}
        self.class_num = len(self.class_dict)         
        key0 = self.keys[0]
        _, origin_sr = torchaudio.load(
            path_join(self.data[key0]['wav']))
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        if self.pre_load:
            self.wavs = self._load_all()

    def _load_wav(self, path):
        wav, _ = torchaudio.load(path)
        wav = self.resampler(wav).squeeze(0)
        return wav

    def _load_all(self):
        wavforms = []
        new_keys = []
        for key in self.keys:
            wav_path = self.data[key]["wav"]
            if self.improvised and not "impro" in wav_path:
                continue
            elif self.scripted and not "script" in wav_path:
                continue
            wav = self._load_wav(wav_path)
            wavforms.append(wav)
            new_keys.append(key)
        self.keys = new_keys
        return wavforms

    def __getitem__(self, 
                    idx):
        key = self.keys[idx]
        if self.multi:
            v, a, d = float(self.data[key]["v"]), float(self.data[key]["a"]), float(self.data[key]["d"])
            label = self.data[key]["emo"]
            label = self.class_dict[label]
            label = v, a, d, label
        elif self.continuous:
            label = float(self.data[key]["v"]), float(self.data[key]["a"]), float(self.data[key]["d"])
        else:
            label = self.data[key]["emo"]
            label = self.class_dict[label]
        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(self.data[key]["wav"])
        wav_len = len(wav)
        return wav, wav_len, label#, key

    def __len__(self):
        return len(self.keys)

SAMPLE_RATE = 16000
MIN_SECOND = 0.05
def wav_transform( wavs,
                    wavs_len):
    original_wavs_len = wavs_len
    if max(original_wavs_len) < MIN_SECOND * SAMPLE_RATE:
        padded_samples = int(MIN_SECOND * SAMPLE_RATE) - max(original_wavs_len)
        wavs = torch.cat(
            (wavs, wavs.new_zeros(wavs.size(0), padded_samples)),
            dim=1,
        )
        wavs_len = wavs_len + padded_samples

    wavs_list = []
    for wav, wav_len in zip(wavs, wavs_len):
        wavs_list.append(wav[:wav_len].numpy())

    max_wav_len = int(max(wavs_len))    
    return wavs_list, original_wavs_len, max_wav_len


device = "cuda" if torch.cuda.is_available() else "cpu"


class IEMOCAPCollator():
    def __init__(self,
                 path,
                 continuous=False,
                 multi=False,
                 return_attention_mask=True,
                 do_normalize=False,
                 sample_rate=16000):
        
        self.continuous = continuous
        self.sample_rate = sample_rate
        self.multi = multi
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(path,
                                                                          return_attention_mask=return_attention_mask,
                                                                          do_normalize=do_normalize)


    def __call__(self,
                samples):

        wavs, wavs_len, labels = list(zip(*samples))        
        if  self.continuous or self.multi:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        wavs_list, original_wavs_len, max_wav_len = wav_transform(wavs, wavs_len)
        input_values = self.feature_extractor(
                    wavs_list,
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True,
                    sampling_rate=self.sample_rate)


        
        return {"input_values": input_values["input_values"], #original_wavs_len, max_wav_len), 
                "attention_mask": input_values["attention_mask"],
                "labels": labels}
