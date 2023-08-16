"""
Downloads and creates data manifest files for IEMOCAP
(https://paperswithcode.com/dataset/iemocap).

Authors:
 * Mirco Ravanelli, 2021
 * Modified by Pierre-Yves Yanni, 2021
 * Abdel Heba, 2021
 * Yingzhi Wang, 2022
"""

import os
import sys
import re
import json
import random
import logging
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
SAMPLERATE = 16000
NUMBER_UTT = 5531



def create_json(wav_list, 
                json_file,
                continuous=False,
                multi=False,
                all_c=False,
                all_continuous = False):
    """
    Creates the json file given a list of wav information.

    Arguments
    ---------
    wav_list : list of list
        The list of wav information (path, label, gender).
    json_file : str
        The path of the output json file
    """

    json_dict = {}
    for obj in wav_list:
        wav_file = obj[0]
        emo = obj[1]
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        uttid = wav_file.split("/")[-1][:-4]
        
        if multi:
            v, a, d, = obj[2:-1]
            json_dict[uttid] = {
                "wav": wav_file,
                "length": duration,
                "v": v,
                "a": a,
                "d": d,
                "emo": emo
            }
            
        elif continuous:
            v, a, d = obj[2:-1]
            json_dict[uttid] = {
                "wav": wav_file,
                "length": duration,
                "v": v,
                "a": a,
                "d": d
            }
            if all_c:
                json_dict[uttid]["emo"] = emo
        else:
            
            # Create entry for this utterance
            json_dict[uttid] = {
                "wav": wav_file,
                "length": duration,
                "emo": emo,
            }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def split_different_speakers_cv(speaker_dict, 
                                test_spk_ids = [1, 2], #session1
                                ):
    """Constructs train, validation and test sets that do not share common
    speakers. There are two different speakers in each session. Train set is
    constituted of 4 sessions (8 speakers), while validation set and test set
    contain each 1 speaker. If test_spk_id is 1, then speaker 2 is selected
    automatically for validation set, and training set contains other 8 speakers.
    If test_spk_id is 2, then speaker 1 is selected for validation set.

    Arguments
    ---------
    speaker_dict: dict
        a dictionary of speaker id and its corresponding audio information
    test_spk_id: int
        Id of speaker used for test set, 10 speakers in total.
        Session1 contains speaker 1&2, Session2 contains speaker 3&4, ...

    Returns
    ------
    dictionary containing train, and test splits.
    """
    data_split = {k: [] for k in ["train", "test"]}
    data_split["test"].extend(speaker_dict[str(test_spk_ids[0])]) 
    data_split["test"].extend(speaker_dict[str(test_spk_ids[1])]) 

    for i in range(1, 11):
        if i not in test_spk_ids:
            data_split["train"].extend(speaker_dict[str(i)])
            
    return data_split


def split_sets(speaker_dict, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.
    Note that a better approach is to make sure that all the classes have the
    same proportion of samples (e.g, spk01 should have 80% of samples in
    training, 10% validation, 10% test, the same for speaker2 etc.). This
    is the approach followed in some recipes such as the Voxceleb one. For
    simplicity, we here simply split the full list without necessarly
    respecting the split ratio within each class.

    Arguments
    ---------
    speaker_dict : list
        a dictionary of speaker id and its corresponding audio information
    split_ratio: list
        List composed of three integers that sets split ratios for train,
        valid, and test sets, respectively.
        For instance split_ratio=[80, 10, 10] will assign 80% of the sentences
        to training, 10% for validation, and 10% for test.

    Returns
    ------
    dictionary containing train, valid, and test splits.
    """

    wav_list = []
    for key in speaker_dict.keys():
        wav_list.extend(speaker_dict[key])

    # Random shuffle of the list
    random.shuffle(wav_list)
    tot_split = sum(split_ratio)
    tot_snts = len(wav_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = wav_list[0:n_snts]
        del wav_list[0:n_snts]
    data_split["test"] = wav_list

    return data_split


def transform_data(path_loadSession,
                   five_way=False, 
                   continuous=False,
                   multi=False,
                   all_continuous = False):
    """
    Create a dictionary that maps speaker id and corresponding wavs

    Arguments
    ---------
    path_loadSession : str
        Path to the folder where the original IEMOCAP dataset is stored.

    Example
    -------
    >>> data_original = '/path/to/iemocap/IEMOCAP_full_release/Session'
    >>> data_transformed = '/path/to/iemocap/IEMOCAP_ahsn_leave-two-speaker-out'
    >>> transform_data(data_original, data_transformed)
    """

    speaker_dict = {str(i + 1): [] for i in range(10)}

    speaker_count = 0
    for k in range(5):
        session = load_session("%s%s" % (path_loadSession, k + 1),
                               five_way=five_way,
                               continuous=continuous,
                               multi=multi,
                               all_continuous=all_continuous)
        for idx in range(len(session)):
            if session[idx][-1] == "F":
                speaker_dict[str(speaker_count + 1)].append(session[idx])
            else:
                speaker_dict[str(speaker_count + 2)].append(session[idx])
        speaker_count += 2

    return speaker_dict


def load_utterInfo(inputFile):
    """
    Load utterInfo from original IEMOCAP database
    """

    # this regx allow to create a list with:
    # [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]
    # [V, A, D] means [Valence, Arousal, Dominance]
    pattern = re.compile(
        "[\[]*[0-9]*[.][0-9]*[ -]*[0-9]*[.][0-9]*[\]][\t][a-z0-9_]*[\t][a-z]{3}[\t][\[][0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[\]]",
        re.IGNORECASE,
    )  # noqa
    with open(inputFile, "r") as myfile:
        data = myfile.read().replace("\n", " ")
    result = pattern.findall(data)
    out = []
    for i in result:
        a = i.replace("[", "")
        b = a.replace(" - ", "\t")
        c = b.replace("]", "")
        x = c.replace(", ", "\t")
        out.append(x.split("\t"))
    return out



def load_session(pathSession,
                 five_way=False, 
                 continuous=False,
                 multi=False,
                 all_c=False,
                 all_continuous = False):
    """Load wav file from IEMOCAP session
    and keep only the following 4 emotions:
    [neural, happy, sad, anger].

    Arguments
    ---------
        pathSession: str
            Path folder of IEMOCAP session.
    Returns
    -------
        improvisedUtteranceList: list
            List of improvised utterancefor IEMOCAP session.
    """
    pathEmo = pathSession + "/dialog/EmoEvaluation/"
    pathWavFolder = pathSession + "/sentences/wav/"
    fourWayList = ["neu", "hap", "sad", "ang", "exc"]
    improvisedUtteranceList = []
    if continuous:
        for emoFile in [
            f
            for f in os.listdir(pathEmo)
            if os.path.isfile(os.path.join(pathEmo, f))
        ]:
            for utterance in load_utterInfo(pathEmo + emoFile):

                path = (
                    pathWavFolder
                    + utterance[2][:-5]
                    + "/"
                    + utterance[2]
                    + ".wav"
                )
                emo = utterance[3]
                v, a, d = utterance[-3:]
                if not all_continuous:
                    if utterance[3] not in fourWayList:
                        if not (five_way and utterance[3]!='xxx'):
                            # label = "oth"
                            continue
                # if emo not in fourWayList:
                #     emo = "hap"
                # if emo == "exc":
                    # emo = "hap"/
                if emoFile[7] != "i" and utterance[2][7] == "s":
                    
                    improvisedUtteranceList.append(
                        [path, emo, v, a, d, utterance[2][18]]
                    )
                else:
                    improvisedUtteranceList.append(
                        [path, emo, v, a, d, utterance[2][15]]
                    )
    else:
        from collections import defaultdict
        for emoFile in [
            f
            for f in os.listdir(pathEmo)
            if os.path.isfile(os.path.join(pathEmo, f))
        ]:
            for utterance in load_utterInfo(pathEmo + emoFile):

                path = (
                    pathWavFolder
                    + utterance[2][:-5]
                    + "/"
                    + utterance[2]
                    + ".wav"
                )

                label = utterance[3]
                if label == "exc":
                    label = "hap"

                # if utterance[3] not in fourWayList:
                    # label = "hap"
                    # if five_way and utterance[3]!='xxx':
                    #     label = "oth"
                    # else:
                    #     # continue
                    #     label = "hap"
                if multi:
                    v, a, d, = utterance[-3:]
                    if emoFile[7] != "i" and utterance[2][7] == "s":
                        
                        improvisedUtteranceList.append(
                            [path,  label, v, a, d, utterance[2][18]]
                        )
                    else:
                        improvisedUtteranceList.append(
                            [path,  label, v, a, d, utterance[2][15]]
                        )
                else:
                    if emoFile[7] != "i" and utterance[2][7] == "s":
                        
                        improvisedUtteranceList.append(
                            [path, label, utterance[2][18]]
                        )
                    else:
                        improvisedUtteranceList.append(
                            [path, label, utterance[2][15]]
                        )
    return improvisedUtteranceList