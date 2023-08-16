import os, torch
import torchaudio
import pandas as pd
import numpy as np
from torchaudio.transforms import Resample
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor


s = \
"""
03 : male, 31 years old
08 : female, 34 years
09 : female, 21 years
10 : male, 32 years
11 : male, 26 years
12 : male, 30 years
13 : female, 32 years
14 : female, 35 years
15 : male, 25 years
16 : female, 31 years
"""

actor_dict = {}
for sent in s.split("\n"):
    if len(sent) > 0:
        sent = sent.split(":")
        key = sent[0].strip()
        value = sent[1].split(",")[0].strip()
        actor_dict[key] = value


class EMODBDataset(Dataset):
    def __init__(self, 
                 data_path = "//home/nl438/rds/hpc-work/PROJECT/data/emodb_data/wav/",
                 iemocap_only = True,
                 set = "train"):
        self.audio_path = data_path

        self.emotion_dict = {'A' : "fear",
                        'E': "disgust",
                        "L" : "boredom",
                        "T" : "sadness",
                        "W": "anger",
                        "F": "joy",
                        "N": "neutral"
                        }

        if iemocap_only:
            self.emodb2iemocap = {"sadness": "sad",
                                "anger": "ang",
                                "joy": "hap",
                                "neutral": "neu"
                                }
            self.class_dict = {"neu": 0, "hap": 1, "ang": 2, "sad": 3}
        else:
            self.emodb2iemocap = {"sadness": "sad",
                                "anger": "ang",
                                "joy": "hap",
                                "neutral": "neu",
                                "sor": "sur"
                                }
            self.class_dict = {"neu": 0, "hap": 1, "ang": 2, "sad": 3, "sur": 4}   
        self.id2emo = {v:k for k,v in self.class_dict.items()}
        self.wavs = []
        self.labels = []
        self.names = []
        self.glabels = []
        if set == "train":
            self.actors = [ "11", "12", "13", "14", "15", "16"]
        if set == "test":
            self.actors = ["03", "08", "09", "10" ]
        self.load()

    def __len__(self,):
        return len(self.wavs)

    def load(self):
        wavs = []
        labels = []
        audio_files = sorted(os.listdir(self.audio_path))
        pname = audio_files[0]
        for audio_file in audio_files[1:]:
            full_audio_file = os.path.join(self.audio_path, audio_file)
            actor = audio_file[:2]
            emo_label = self.emotion_dict[audio_file[5]]
            if actor in self.actors and emo_label in self.emodb2iemocap and pname[:6]!=audio_file[:6]:
                wav, sr = torchaudio.load(full_audio_file)
                wav = wav.squeeze(0)
                label = self.class_dict[self.emodb2iemocap[emo_label]]
                wavs.append(wav)
                labels.append(label)
                self.glabels.append(actor_dict[actor][0] + str(label))
                self.names.append(pname)
                pname = audio_file
            else:
                pname = audio_file
                continue

        

        self.wavs = wavs
        self.labels = labels

    def __getitem__(self, 
                    id):
        audio = self.wavs[id]
        label = self.labels[id]
        audio_len = len(audio)
        return audio, audio_len, label
    
    
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


class EMODBCollator():
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