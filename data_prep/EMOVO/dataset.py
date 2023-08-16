import os, torch
import torchaudio
import pandas as pd
import numpy as np
from torchaudio.transforms import Resample
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor

class EMOVODataset(Dataset):
    def __init__(self,
                 emovo_path = "/home/nl438/rds/hpc-work/PROJECT/data/emovo_data/EMOVO/",
                 set = "train",
                 iemocap_only = True
                 ):
        self.audio_path = emovo_path
        self.set = set

        if iemocap_only:
            self.emovo2iemocap = {"tri": "sad",
                                "rab": "ang",
                                "gio": "hap",
                                "neu": "neu"
                                }
            self.class_dict = {"neu": 0, "hap": 1, "ang": 2, "sad": 3}
        else:
            self.emovo2iemocap = {"tri": "sad",
                                "rab": "ang",
                                "gio": "hap",
                                "neu": "neu",
                                "sor": "sur"
                                }
            self.class_dict = {"neu": 0, "hap": 1, "ang": 2, "sad": 3, "sur": 4}   
        self.id2emo = {v:k for k,v in self.class_dict.items()}
        self.wavs = []
        self.labels = []
        self.glabels = []
        if set == "train":
            self.actors = ["f1", "m1", "f2", "m2"]
        if set == "test":
            self.actors = ["f3", "m3"]
        self.load()



    def __len__(self,):
        return len(self.wavs)

    def load(self):
        wavs = []
        labels = []
        for actor in self.actors:
            g = actor[0]
            actor_path = os.path.join(self.audio_path, actor)
            for audio_name in os.listdir(actor_path):
                audio_file = os.path.join(actor_path, audio_name)
                label = audio_name.split("-")[0]
                if label in self.emovo2iemocap.keys():
                    label = self.class_dict[self.emovo2iemocap[label]]
                    glabel = g + str(label)
                    self.glabels.append(glabel)

                else:
                    continue
                wav, sr = torchaudio.load(audio_file)
                wav = Resample(sr, 16000)(wav).mean(0).squeeze(0)
                wavs.append(wav)
                labels.append(label)
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


class EMOVOCollator():
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