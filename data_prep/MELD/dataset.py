import re, os, torch
import pandas as pd
import numpy as np
import torch, torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample
from transformers import Wav2Vec2FeatureExtractor

class MELDDataset(Dataset):
    def __init__(self,
                 audio_path,
                 df_path):
        self.audio_path = audio_path
        self.audio_list = os.listdir(audio_path)
        self.df = pd.read_csv(df_path)
        self.pattern = r"([0-9]+)(_[a-z]+)([0-9]+)"
        self.meld2iemocap = {"sadness": "sad",
                             "anger": "ang",
                             "joy": "hap",
                             "neutral": "neu"
                             }
        self.class_dict = {"neu": 0, "hap": 1, "ang": 2, "sad": 3}
        self.audios = []
        self.labels = []
        self.resampler = Resample(22050, 16000)
        self.load()    


    def __len__(self,):
        return len(self.audios)

    def get_label_audio(self, audio_file):
        groups = re.findall(self.pattern, audio_file)[0]
        dialogue_id = int(groups[0])
        utterance_id = int(groups[-1])
        try:
            label = self.df[(self.df["Dialogue_ID"]==dialogue_id)&(self.df["Utterance_ID"]==utterance_id)]["Emotion"].values[0]
        except:
            return        
        if label in ["surprise", "disgust", "fear"]:
            return
        label = self.meld2iemocap[label]
        label = self.class_dict[label]
        return label


    def load(self):
        wavs = []
        labels = []
        for audio_file in self.audio_list:
            if audio_file.endswith("wav"):
                self.full_path = os.path.join(self.audio_path, audio_file)
                label = self.get_label_audio(self.full_path)
                if label is None:
                    continue
                wav, _ = torchaudio.load(self.full_path)
                wav = self.resampler(wav)[0]#.mean(dim=0)
                wavs.append(wav)
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


class MELDCollator():
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