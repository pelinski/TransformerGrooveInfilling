import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import wandb

from utils import get_sf_list, add_metadata_to_hvo_seq, pad_to_match_max_len, get_voice_idx_for_item, \
    get_sf_v_combinations, save_parameters_to_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GrooveMidiDataset(Dataset):
    def __init__(self,
                 subset,
                 subset_info,  # in order to store them in parameters json
                 **kwargs):

        """
        Groove Midi Dataset Loader. Max number of items in dataset is N x M x K where N is the number of items in the
        subset, M the maximum number of soundfonts to sample from for each item (max_n_sf) and K is the maximum number
        of voice combinations.

        @param subset:              GrooveMidiDataset subset generated with the Subset_Creator
        @param subset_info:         Dictionary with the routes and filters passed to the Subset_Creator to generate the
                                    subset. Example:
                                    subset_info = {
                                    "pickle_source_path": '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2'
                                               '/Processed_On_17_05_2021_at_22_32_hrs',
                                    "subset": 'GrooveMIDI_processed_train',
                                    "metadata_csv_filename": 'metadata.csv',
                                    "hvo_pickle_filename": 'hvo_sequence_data.obj',
                                    "filters": "bpm": ["beat"],
                                    }

        @param max_len:             Max_length of sequences
        @param mso_parameters:      Dictionary with the parameters for calculating the Multiband Synthesized Onsets.
                                    Refer to `hvo_sequence.hvo_seq.mso()` for the documentation
        @param voices_params:   Dictionary with parameters for generating the combinations of the voices to remove
                                    Refer to utils.get_voice_combinations for documentation
        @param sf_path:             Path with soundfonts
        @param max_n_sf:            Maximum number of soundfonts to sample from for each example
        @param max_aug_items:       Maximum number of synthesized examples per example in subset
        @param dataset_name:        Dataset name (for experiment tracking)
        """

        # default values for kwargs

        self.max_len = kwargs.get('max_len', 32)
        self.mso_parameters = kwargs.get('mso_parameters', {"sr": 44100, "n_fft": 1024, "win_length": 1024,
                                                            "hop_length": 441, "n_bins_per_octave": 16, "n_octaves":
                                                                9, "f_min": 40, "mean_filter_size": 22})
        self.voices_params = kwargs.get('voices_parameters', {"voice_idx": [0, 1], "min_n_voices_to_remove": 1,
                                                              "max_n_voices_to_remove": 2, "prob": [1, 1], "k": 5})
        self.sf_path = kwargs.get('sf_path', "../soundfonts/filtered_soundfonts/")
        self.max_n_sf = kwargs.get('max_n_sf', None)
        self.max_aug_items = kwargs.get('max_aug_items', 10)
        self.timestamp = datetime.now().strftime("%d_%m_%Y_at_%H_%M_hrs")
        self.dataset_name =  "Dataset_" + self.timestamp if kwargs.get('dataset_name') is None else kwargs.get(
            'dataset_name', "Dataset_" + self.timestamp )

        self.subset_info = subset_info
        self.metadata = pd.read_csv(os.path.join(subset_info["pickle_source_path"], subset_info["subset"],
                                                 subset_info["metadata_csv_filename"]))

        # list of soundfonts
        self.sfs_list = get_sf_list(self.sf_path)

        if self.max_n_sf is not None:
            assert (self.max_n_sf <= len(self.sfs_list)), "max_n_sf can not be larger than number of available " \
                                                          "soundfonts"

        # assigning here so that preprocess_dataset can be used as external method for processing the samples given
        # by the evaluator
        (self.hvo_sequences, self.processed_inputs, self.processed_outputs), \
        (self.hvo_index, self.voices_reduced, self.soundfonts) = self.preprocess_dataset(subset)

        # dataset creation parameters
        params = {
            "dataset_name": self.dataset_name,
            "length": len(self.processed_inputs),
            "timestamp": self.timestamp,
            "subset_info": {**self.subset_info,
                            "sf_path": self.sf_path,
                            "max_len": self.max_len,
                            "max_aug_items": self.max_aug_items},
            "mso_parameters": self.mso_parameters,
            "voices_parameters": self.voices_params,
            "max_n_sf": self.max_n_sf,
            "dictionaries": {
                "hvo_index": self.hvo_index,
                "voices_reduced": self.voices_reduced,
                "soundfonts": self.soundfonts
            }
        }

        # log parameters to wandb
        if wandb.ensure_configured():  # if running experiment file with wandb.init()
            wandb.config.update(params, allow_val_change=True)  # update defaults

        # save parameters to json
        save_parameters_to_json(params)

        # convert inputs and outputs to torch tensors
        self.processed_inputs = torch.Tensor(self.processed_inputs).to(device=device)
        self.processed_outputs = torch.Tensor(self.processed_outputs).to(device=device)

    def preprocess_dataset(self, subset):
        # init lists to store hvo sequences and processed io
        hvo_sequences = []
        processed_inputs = []
        processed_outputs = []

        # init list with configurations
        hvo_index = []
        voices_reduced = []
        soundfonts = []

        for hvo_idx, hvo_seq in enumerate(tqdm(subset)):  # only one subset because only one set of filters
            if len(hvo_seq.time_signatures) == 1:  # ignore if time_signature change happens

                all_zeros = not np.any(hvo_seq.hvo.flatten())

                if not all_zeros:  # ignore silent patterns

                    # add metadata to hvo_seq scores
                    add_metadata_to_hvo_seq(hvo_seq, hvo_idx, self.metadata)

                    # pad with zeros to match max_len
                    hvo_seq = pad_to_match_max_len(hvo_seq, self.max_len)

                    # append hvo_seq to hvo_sequences list
                    hvo_sequences.append(hvo_seq)

                    # remove voices in voice_idx not present in item
                    _voice_idx, _voices_params = get_voice_idx_for_item(hvo_seq, self.voices_params)
                    if len(_voice_idx) == 0: continue  # if there are no voices to remove, continue

                    # get voices and sf combinations
                    sf_v_comb = get_sf_v_combinations(_voices_params, self.max_aug_items, self.max_n_sf, self.sfs_list)

                    # for every sf and voice combination
                    for sf, v_idx in sf_v_comb:

                        # reset voices in hvo
                        hvo_seq_in, hvo_seq_out = hvo_seq.reset_voices(voice_idx=v_idx)
                        # if the resulting hvos are 0, skip
                        if not np.any(hvo_seq_in.hvo.flatten()): continue
                        if not np.any(hvo_seq_out.hvo.flatten()): continue

                        # store hvo, v_idx and sf
                        hvo_index.append(hvo_idx)
                        voices_reduced.append(v_idx)
                        soundfonts.append(sf)

                        # processed inputs: mso
                        mso = hvo_seq_in.mso(sf_path=sf, **self.mso_parameters)
                        processed_inputs.append(mso)

                        # processed outputs complementary hvo_seq with reset voices
                        processed_outputs.append(hvo_seq_out.hvo)

        return (hvo_sequences, processed_inputs, processed_outputs), (hvo_index, voices_reduced, soundfonts)

    # getters

    def get_hvo_sequence(self, idx):
        hvo_idx = self.hvo_index[idx]
        return self.hvo_sequences[hvo_idx]

    def get_soundfont(self, idx):
        return self.soundfonts[idx]

    def get_voices_idx(self, idx):
        return self.voices_reduced[idx]

    # dataset methods

    def __len__(self):
        return len(self.processed_inputs)

    def __getitem__(self, idx):
        return self.processed_inputs[idx], self.processed_outputs[idx], idx



