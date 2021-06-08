import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import wandb

from utils import add_metadata_to_hvo_seq, pad_to_match_max_len, get_voice_idx_for_item, get_sf_v_combinations, \
    save_parameters_to_json, NpEncoder

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
        @param voices_parameters:   Dictionary with parameters for generating the combinations of the voices to remove
                                    Refer to utils.get_voice_combinations for documentation
        @param sf_path:             Path with soundfonts
        @param max_n_sf:            Maximum number of soundfonts to sample from for each example
        @param max_aug_items:       Maximum number of synthesized examples per example in subset
        @param dataset_name:        Dataset name (for experiment tracking)
        """

        # default values for kwargs
        max_len = kwargs.get('max_len', 32)
        mso_parameters = kwargs.get('mso_parameters', {"sr": 44100, "n_fft": 1024, "win_length": 1024, "hop_length":
            441, "n_bins_per_octave": 16, "n_octaves": 9, "f_min": 40, "mean_filter_size": 22})
        voices_parameters = kwargs.get('voices_parameters', {"voice_idx": [0, 1], "min_n_voices_to_remove": 1,
                                                             "max_n_voices_to_remove": 2, "prob": [1, 1], "k": 5})
        sf_path = kwargs.get('sf_path', "../soundfonts/filtered_soundfonts/")
        max_n_sf = kwargs.get('max_n_sf', None)
        max_aug_items = kwargs.get('max_aug_items', 10)
        dataset_name = kwargs.get('dataset_name', None)

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
        if sf_path.endswith('.sf2'):  # if the sf_path is to one sf2 file
            sfs_list = [sf_path]
        else:  # if sf_path is a dir with sf2 files
            sfs_list = [os.path.join(sf_path) + sf for sf in os.listdir(sf_path) if sf.endswith('.sf2')]
        if max_n_sf is not None:
            assert (max_n_sf <= len(sfs_list)), "max_n_sf can not be larger than number of available " \
                                                "soundfonts"

        for hvo_idx, hvo_seq in enumerate(tqdm(subset)):  # only one subset because only one set of filters
            if len(hvo_seq.time_signatures) == 1:  # ignore if time_signature change happens

                all_zeros = not np.any(hvo_seq.hvo.flatten())

                if not all_zeros:  # ignore silent patterns

                    # add metadata to hvo_seq scores
                    add_metadata_to_hvo_seq(hvo_seq,hvo_idx,metadata)

                    # pad with zeros to match max_len
                    hvo_seq = pad_to_match_max_len(hvo_seq,max_len)

                    # append hvo_seq to hvo_sequences list
                    self.hvo_sequences.append(hvo_seq)

                    # remove voices in voice_idx not present in item
                    _voice_idx, _voices_params = get_voice_idx_for_item(hvo_seq, voices_parameters["voice_idx"])
                    if len(_voice_idx) == 0: continue  # if there are no voices to remove, continue

                    # get voices and sf combinations
                    sf_v_comb = get_sf_v_combinations(_voices_params, max_aug_items, max_n_sf, sfs_list)

                    # for every sf and voice combination
                    for sf, v_idx in sf_v_comb:

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

        # dataset creation parameters
        parameters = {
            "dataset_name": dataset_name if dataset_name is not None else "Dataset_" + dt_string,
            "length": len(self.processed_inputs),
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

        if wandb.ensure_configured(): # if running experiment file with wandb.init()
            wandb.config.update(parameters, allow_val_change=True) # update defaults

        # save parameters
        save_parameters_to_json(parameters)

        # convert inputs and outputs to torch tensors
        self.processed_inputs = torch.Tensor(self.processed_inputs).to(device=device)
        self.processed_outputs = torch.Tensor(self.processed_outputs).to(device=device)

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
