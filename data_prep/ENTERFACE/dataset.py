import re, os, torch
import pandas as pd
import numpy as np
import torch, torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample
from transformers import Wav2Vec2FeatureExtractor


ENTERFACE_DIR = "/home/nl438/rds/hpc-work/PROJECT/data/enterface_audios"

class ENTDataset(Dataset):
    def __init__(self,
                 ent_dir = ENTERFACE_DIR,
                 iemocap_only=True,
                 all_esd = False,
                 sets = False,
                 train = False,
                 val = False,
                 test = False
                 ):
        self.audio_path = ent_dir
        self.iemocap_only = iemocap_only
        self.all_esd = all_esd
        gen_subject = open("/home/nl438/rds/hpc-work/PROJECT/data/enterface_audios/subject_gender.txt", "r").readlines()
        gen_dict = {}
        self.train = train
        self.val = val
        self.test = test
        for line in gen_subject:
            sub, gen = line.split()
            gen_dict[sub] = gen.strip()
        self.gen_dict = gen_dict

        if iemocap_only:
            self.ent2iemocap = {"sadness": "sad",
                                "anger": "ang",
                                "happiness": "hap",
                                }
            self.class_dict = {"neu": 0, "hap": 1, "ang": 2, "sad": 3}
        else:
            self.ent2iemocap = {"sadness": "sad",
                                "anger": "ang",
                                "happiness": "hap",
                                "surprise": "sur",
                                # "disgust" : "dis",
                                # "fear" : "fear"
                                }
            # self.class_dict = {"fear": 0, "hap": 1, "ang": 2, "sad": 3, "sur": 4, "dis": 5}   
            self.class_dict = {"sur": 0, "hap": 1, "ang": 2, "sad": 3}#, "fear": 4, "dis": 5}   

        self.id2emo = {v:k for k,v in self.class_dict.items()}
        self.audios = []
        self.labels = []
        self.emotions = []
        # self.resampler = Resample(22050, 16000)
        self.genders = []
        self.load()
        if sets:
            import numpy as np
            from sklearn.model_selection import train_test_split
            np.random.seed(0)
            emotions = np.array(self.emotions)
            genders = np.array(self.genders)
            n = len(emotions)
            y = np.core.defchararray.add(emotions, genders) 
            X = np.arange(n)
            train_x, test_x, train_y, test_y = train_test_split(X, 
                                                                y, 
                                                                test_size=0.4,
                                                                shuffle=True)
            train_x, val_x, train_y, val_y = train_test_split(train_x, 
                                                            train_y, 
                                                            test_size=0.4,
                                                            shuffle=True)
            self.train_ids = train_x
            self.val_ids = val_x
            self.test_ids = test_x


    def __len__(self,):
        if self.train:
            return len(self.train_ids)
        elif self.val:
            return len(self.val_ids)
        elif self.test:
            return len(self.test_ids)
        return len(self.audios)

    def load(self):
        wavs = []
        labels = []
        genders = []
        emotions = []
        for subject in os.listdir(self.audio_path):
            subject_dir = os.path.join(self.audio_path, subject)
            if not os.path.isdir(subject_dir):
                continue
            for emotion in os.listdir(subject_dir):
                if emotion in self.ent2iemocap.keys():
                    label = self.class_dict[self.ent2iemocap[emotion]]
                    emo_dir = os.path.join(subject_dir, emotion)
                    for sentence in os.listdir(emo_dir):
                        sent_dir = os.path.join(emo_dir, sentence)
                        for audio_file in os.listdir(sent_dir):    
                            self.full_audio_path = os.path.join(sent_dir, audio_file)
                            wav, sr = torchaudio.load(self.full_audio_path)
                            wav = wav.mean(0)
                            wavs.append(wav.squeeze())
                            labels.append(label)
                            sub_joined = ''.join(subject.split())
                            genders.append(self.gen_dict[sub_joined])
                            emotions.append(self.id2emo[label])
        self.labels = labels
        self.audios = wavs
        self.genders = genders
        self.emotions = emotions

    def __getitem__(self, 
                    id):
        if self.train:
            id = self.train_ids[id]
        elif self.val:
            id = self.val_ids[id]
        elif self.test:
            id = self.test_ids[id]
        audio = self.audios[id]
        label = self.labels[id]
        audio_len = len(audio)
        gender = self.genders[id]
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


class ENTCollator():
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