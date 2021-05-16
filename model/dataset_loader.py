import torch
from torch.utils.data import Dataset
import pandas as pd
import os

import numpy as np
import json
from datetime import datetime
import itertools
import random

# default parameters
filters = {
    "bpm": ["beat"],
}
mso_parameters = {
    "sr": 44100,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 441,
    "n_bins_per_octave": 16,
    "n_octaves": 9,
    "f_min": 40,
    "mean_filter_size": 22
}
voices_parameters = {"voice_idx":[0,1],
                 "min_n_voices_to_remove":1,
                 "max_n_voices_to_remove":2,
                 "prob":[1,1],
                  "k": 5}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GrooveMidiDataset(Dataset):
    def __init__(self,
                 subset,
                 subset_info, # in order to store them in parameters json
                 mso_parameters=mso_parameters,
                 voices_parameters=voices_parameters,
                 sf_path="../soundfonts/filtered_soundfonts/",
                 max_len=32,
                 max_aug_items=10,      # max number of combinations to obtain from one item
                 dataset_name=None
                 ):

        metadata = pd.read_csv(os.path.join(subset_info["pickle_source_path"], subset_info["subset"],
                                            subset_info["metadata_csv_filename"]))

        # init lists to store hvo sequences and processed io
        self.hvo_sequences = []
        self.processed_inputs = []
        self.processed_outputs = []

        # init list with configurations
        self.hvo_index = []
        self.voices_reduced = []
        self.soundfonts = []

        # list of soundfonts
        sfs = [os.path.join(sf_path) + sf for sf in os.listdir(sf_path)]

        for hvo_idx, hvo_seq in enumerate(subset):  # only one subset because only one set of filters
            if len(hvo_seq.time_signatures) == 1:  # ignore if time_signature change happens

                all_zeros = not np.any(hvo_seq.hvo.flatten())
                if not all_zeros:  # ignore silent patterns

                    # add metadata to hvo_seq scores
                    hvo_seq.drummer = metadata.loc[hvo_idx].at["drummer"]
                    hvo_seq.session = metadata.loc[hvo_idx].at["session"]
                    hvo_seq.master_id = metadata.loc[hvo_idx].at["master_id"]
                    hvo_seq.style_primary = metadata.loc[hvo_idx].at["style_primary"]
                    hvo_seq.style_secondary = metadata.loc[hvo_idx].at["style_secondary"]
                    hvo_seq.beat_type = metadata.loc[hvo_idx].at["beat_type"]
                    hvo_seq.loop_id = metadata.loc[hvo_idx].at["loop_id"]
                    hvo_seq.bpm = metadata.loc[hvo_idx].at["bpm"]

                    # pad with zeros to match max_len
                    pad_count = max(max_len - hvo_seq.hvo.shape[0], 0)
                    hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, pad_count), (0, 0)), 'constant')
                    hvo_seq.hvo = hvo_seq.hvo[:max_len, :]  # in case seq exceeds max len

                    # append hvo_seq
                    self.hvo_sequences.append(hvo_seq)

                    # voice_combinations
                    voice_idx_comb = get_voice_combinations(voices_parameters)
                    # combinations of sf and voices
                    sf_v_comb = list(itertools.product(sfs, voice_idx_comb))

                    # if there's more combinations than max_aug_items, choose randomly
                    if len(sf_v_comb) > max_aug_items:
                        sf_v_comb = random.choices(sf_v_comb, k=max_aug_items)

                    # for every sf and voice combination
                    for sf, v_idx in sf_v_comb:
                        v_idx = list(v_idx)

                        # reset voices in hvo
                        hvo_seq_in, hvo_seq_out = hvo_seq.reset_voices(voice_idx=v_idx)
                        # if resetted hvo is 0 after removing the voices, skip
                        if not np.any(hvo_seq.hvo.flatten()): continue

                        # store hvo, v_idx and sf
                        self.hvo_index.append(hvo_idx)
                        self.voices_reduced.append(v_idx)
                        self.soundfonts.append(sf)

                        # processed inputs: mso
                        mso = hvo_seq_in.mso(sf_path=sf)
                        self.processed_inputs.append(mso)

                        # processed outputs complementary hvo_seq with reset voices
                        self.processed_outputs.append(hvo_seq_out.hvo)

        # store hvo index and soundfonts in csv
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_at_%H_%M_hrs")

        # dataset name
        if dataset_name is None: dataset_name = "Dataset_" + dt_string

        # save parameters
        parameters_path = os.path.join('../result', dataset_name)
        if not os.path.exists(parameters_path): os.makedirs(parameters_path)

        parameters = {
            "dataset_name": dataset_name,
            "timestamp": dt_string,
            "subset_info" : {**subset_info,
                              "sf_path": sf_path,
                                "max_len": max_len,
                                "max_aug_items": max_aug_items},
            "mso_parameters": mso_parameters,
            "voices_parameters": voices_parameters,
            "dictionaries": {
                "hvo_index": self.hvo_index,
                "voices_reduced": self.voices_reduced,
                "soundfonts": self.soundfonts
            }

        }
        parameters_json = os.path.join(parameters_path, 'parameters.json')
        with open(parameters_json, 'w') as f:
            json.dump(parameters, f)

        # convert to torch tensors
        self.processed_inputs = torch.Tensor(self.processed_inputs,device=device)
        self.processed_outputs = torch.Tensor(self.processed_outputs,device=device)

    def get_hvo_sequence(self, idx):
        hvo_idx = self.hvo_index[idx]
        return self.hvo_sequences[hvo_idx]

    def get_soundfont(self, idx):
        return self.soundfonts[idx]

    def get_voices_idx(self, idx):
        return self.voices_reduced[idx]

    def __len__(self):
        return len(self.processed_inputs)

    def __getitem__(self, idx):
        return self.processed_inputs[idx], self.processed_outputs[idx], idx


def get_voice_combinations(**kwargs):
    voice_idx = kwargs.get("voice_idx",[0, 1, 2, 3, 4])
    min_n_voices_to_remove = kwargs.get("min_n_voices_to_remove", 1)
    max_n_voices_to_remove = kwargs.get("max_n_voices_to_remove",3)
    prob = kwargs.get("prob",[1, 1, 1, 1, 1])
    k = kwargs.get("k", 5)


    assert (len(voice_idx) == len(prob)), "The voice_idx list and the prob list must be the same length"
    assert (len(voice_idx) >= min_n_voices_to_remove and len(voice_idx) <= max_n_voices_to_remove), " " \
                      "min_n_voices_to_remove <= len(voice_idx) <= max_n_voices_to_remove"

    voice_idx_comb = []
    weights = []

    for i, n_voices_to_remove in enumerate(range(min_n_voices_to_remove, max_n_voices_to_remove + 1)):
        _voice_idx_comb = list(itertools.combinations(voice_idx, n_voices_to_remove))
        voice_idx_comb.extend(_voice_idx_comb)

        _weights = list(np.repeat(prob[i], len(_voice_idx_comb)))
        weights.extend(_weights)

    voice_idx_comb = random.choices(voice_idx_comb, weights=weights, k=k)

    return voice_idx_comb