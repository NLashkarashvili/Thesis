import re, os, torch
import pandas as pd
import numpy as np
import torch, torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample
from transformers import Wav2Vec2FeatureExtractor

ENTERFACE_DIR = "/home/nl438/rds/hpc-work/PROJECT/data/enterface_audios"

class ESDDataset(Dataset):
    def __init__(self,
                 esd_dir,
                 set = 'train',
                 english_only = True,
                 iemocap_only = True,
                 mandarin_only = False,
                 ):
        self.audio_path = esd_dir
        if iemocap_only:
            self.esd2iemocap = {"Sad": "sad",
                                "Angry": "ang",
                                "Happy": "hap",
                                "Neutral": "neu"
                                }
            self.class_dict = {"neu": 0, "hap": 1, "ang": 2, "sad": 3}
        else:
            self.esd2iemocap = {"Sad": "sad",
                                "Angry": "ang",
                                "Happy": "hap",
                                "Neutral": "neu",
                                "Surprise": "sur"
                                }
            self.class_dict = {"neu": 0, "hap": 1, "ang": 2, "sad": 3, "sur": 4}
        self.audios = []
        self.labels = []
        # self.resampler = Resample(22050, 16000)
        self.set = set
        self.english_only = english_only
        self.mandarin_only = mandarin_only
        self.load()    

    def __len__(self,):
        return len(self.audios)


    def load(self):
        wavs = []
        labels = []
        for subject in os.listdir(self.audio_path):
            if subject.endswith("txt"):
                continue
            if self.english_only:
                if int(subject) >= 11:
                    subject_dir = os.path.join(self.audio_path, subject)
                    for emotion in os.listdir(subject_dir):
                        if emotion in self.esd2iemocap.keys():
                            label = self.class_dict[self.esd2iemocap[emotion]]
                            emo_dir = os.path.join(subject_dir, emotion)
                            for set in os.listdir(emo_dir):
                                if set == self.set:
                                    set_path = os.path.join(emo_dir, set)
                                    for audio_file in os.listdir(set_path):
                                        self.full_audio_path = os.path.join(set_path, audio_file)
                                        wav, _ = torchaudio.load(self.full_audio_path)
                                        wavs.append(wav.squeeze())
                                        labels.append(label)
            elif self.mandarin_only:
                if int(subject) < 11:
                    subject_dir = os.path.join(self.audio_path, subject)
                    for emotion in os.listdir(subject_dir):
                        if emotion in self.esd2iemocap.keys():
                            label = self.class_dict[self.esd2iemocap[emotion]]
                            emo_dir = os.path.join(subject_dir, emotion)
                            for set in os.listdir(emo_dir):
                                if set == self.set:
                                    set_path = os.path.join(emo_dir, set)
                                    for audio_file in os.listdir(set_path):
                                        self.full_audio_path = os.path.join(set_path, audio_file)
                                        wav, _ = torchaudio.load(self.full_audio_path)
                                        wavs.append(wav.squeeze())
                                        labels.append(label)
            else:
                subject_dir = os.path.join(self.audio_path, subject)
                for emotion in os.listdir(subject_dir):
                    if emotion in self.esd2iemocap.keys():
                        label = self.class_dict[self.esd2iemocap[emotion]]
                        emo_dir = os.path.join(subject_dir, emotion)
                        for set in os.listdir(emo_dir):
                            if set == self.set:
                                set_path = os.path.join(emo_dir, set)
                                for audio_file in os.listdir(set_path):
                                    self.full_audio_path = os.path.join(set_path, audio_file)
                                    wav, _ = torchaudio.load(self.full_audio_path)
                                    wavs.append(wav.squeeze())
                                    labels.append(label)
            
            
            
        self.labels = labels
        self.audios = wavs

    def __getitem__(self, id):
        audio = self.audios[id]
        label = self.labels[id]
        audio_len = len(audio)
        return audio, audio_len,  label
    
    

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
        print(wavs_len)
        wavs_len = wavs_len + padded_samples

    wavs_list = []
    for wav, wav_len in zip(wavs, wavs_len):
        wavs_list.append(wav[:wav_len].numpy())

    max_wav_len = int(max(wavs_len))    
    return wavs_list, original_wavs_len, max_wav_len


device = "cuda" if torch.cuda.is_available() else "cpu"


class ESDCollator():
    def __init__(self,
                 path,
                 return_attention_mask=True,
                 do_normalize=True,
                 sample_rate=16000):
        
        self.sample_rate = sample_rate
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(path,
                                                                    return_attention_mask=return_attention_mask,
                                                                    do_normalize=do_normalize)


    def __call__(self,
                samples):

        wavs, wavs_len, labels = list(zip(*samples))        
        labels = torch.LongTensor(labels)
        wavs_list, _, _ = wav_transform(wavs, wavs_len)
        input_values = self.feature_extractor(
                    wavs_list,
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True,
                    sampling_rate=self.sample_rate)


        
        return {"input_values": input_values["input_values"], 
                "attention_mask": input_values["attention_mask"],
                "labels": labels}