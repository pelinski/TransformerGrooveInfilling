import torch
from torch.utils.data import Dataset
import pandas as pd
import os

import numpy as np
import json
from datetime import datetime

from utils import get_sf_v_combinations

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
voices_parameters = {"voice_idx": [0, 1],
                     "min_n_voices_to_remove": 1,
                     "max_n_voices_to_remove": 2,
                     "prob": [1, 1],
                     "k": 5}  # set k to None to get all possible combinations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GrooveMidiDataset(Dataset):
    def __init__(self,
                 subset,
                 subset_info,  # in order to store them in parameters json
                 max_len=32,
                 mso_parameters=mso_parameters,
                 voices_parameters=voices_parameters,
                 sf_path="../soundfonts/filtered_soundfonts/",
                 max_n_sf=None,
                 max_aug_items=10,
                 dataset_name=None
                 ):

        """
        Groove Midi Dataset Loader. Max number of items in dataset is N x M x K where N is the number of items in the
        subset, M the maximum number of soundfonts to sample from for each item (max_n_sf) and K is the maximum number
        of voice combinations.

        @param subset:              GrooveMidiDataset subset generated with the Subset_Creator
        @param subset_info:         Dictionary with the routes and filters passed to the Subset_Creator to generate the
                                    subset
        @param max_len:             Max_length of sequences
        @param mso_parameters:      Dictionary with the parameters for calculating the Multiband Synthesized Onsets.
                                    Refer to `hvo_sequence.hvo_seq.mso()` for the documentation
        @param voices_parameters:   Dictionary with parameters for generating the combinations of the voices to remove
                                    Refer to utils.get_voice_combinations for documentation
        @param sf_path:             Path with soundfonts
        @param max_n_sf:            Maximum number of soundfonts to sample from for each example
        @param max_aug_items:       Maximum number of synthesized examples per example in subset
        @param dataset_name:        Dataset name (for experiment tracking)
        """

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
        sfs_list = [os.path.join(sf_path) + sf for sf in os.listdir(sf_path)]
        if max_n_sf is not None:
            assert (max_n_sf <= len(sfs_list)), "max_n_sf can not be larger than number of available " \
                                                "soundfonts"

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

                    # append hvo_seq to hvo_sequences list
                    self.hvo_sequences.append(hvo_seq)

                    # get voices and sf combinations
                    sf_v_comb = get_sf_v_combinations(voices_parameters, max_aug_items, max_n_sf, sfs_list)

                    # for every sf and voice combination
                    for sf, v_idx in sf_v_comb:
                        v_idx = list(v_idx)

                        # reset voices in hvo
                        hvo_seq_in, hvo_seq_out = hvo_seq.reset_voices(voice_idx=v_idx)
                        # if the resulting hvos are 0, skip
                        if not np.any(hvo_seq_in.hvo.flatten()): continue
                        if not np.any(hvo_seq_out.hvo.flatten()): continue

                        # store hvo, v_idx and sf
                        self.hvo_index.append(hvo_idx)
                        self.voices_reduced.append(v_idx)
                        self.soundfonts.append(sf)

                        # processed inputs: mso
                        mso = hvo_seq_in.mso(sf_path=sf)
                        self.processed_inputs.append(mso)

                        # processed outputs complementary hvo_seq with reset voices
                        self.processed_outputs.append(hvo_seq_out.hvo)

        # current time
        dt_string = datetime.now().strftime("%d_%m_%Y_at_%H_%M_hrs")

        # dataset name
        if dataset_name is None: dataset_name = "Dataset_" + dt_string

        # dataset creation parameters
        parameters = {
            "dataset_name": dataset_name,
            "timestamp": dt_string,
            "subset_info": {**subset_info,
                            "sf_path": sf_path,
                            "max_len": max_len,
                            "max_aug_items": max_aug_items},
            "mso_parameters": mso_parameters,
            "voices_parameters": voices_parameters,
            "max_n_sf": max_n_sf,
            "dictionaries": {
                "hvo_index": self.hvo_index,
                "voices_reduced": self.voices_reduced,
                "soundfonts": self.soundfonts
            }

        }

        # save parameters
        parameters_path = os.path.join('../result', dataset_name)
        if not os.path.exists(parameters_path): os.makedirs(parameters_path)
        parameters_json = os.path.join(parameters_path, 'parameters.json')
        with open(parameters_json, 'w') as f:
            json.dump(parameters, f)

        # convert inputs and outputs to torch tensors
        self.processed_inputs = torch.Tensor(self.processed_inputs, device=device)
        self.processed_outputs = torch.Tensor(self.processed_outputs, device=device)

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
