import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import wandb

from utils import get_voice_combinations, NpEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIXME
class GrooveMidiDatasetSymbolic(Dataset):
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
        voices_parameters = kwargs.get('voices_parameters', {"voice_idx": [0, 1], "min_n_voices_to_remove": 1,
                                                             "max_n_voices_to_remove": 2, "prob": [1, 1], "k": 5})
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

        for hvo_idx, hvo_seq in enumerate(tqdm(subset)):  # only one subset because only one set of filters
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

                    # remove voices in voice_idx not present in item
                    active_voices = hvo_seq.get_active_voices()
                    _voice_idx = voices_parameters["voice_idx"]
                    non_present_voices_idx = np.argwhere(~np.isin(_voice_idx, active_voices)).flatten()
                    _voice_idx = np.delete(_voice_idx, non_present_voices_idx).tolist()
                    if len(_voice_idx) == 0: continue  # if there are no voices to remove, continue

                    # create voices_parameters dict with adapted voices for item
                    v_params = voices_parameters
                    v_params["voice_idx"] = list(_voice_idx)
                    v_params["prob"] = voices_parameters["prob"][:len(_voice_idx)]

                    # get voices and sf combinations
                    # FIXME what if only one voice?
                    v_comb = get_voice_combinations(**voices_parameters)
                    # for every sf and voice combination
                    for v_idx in v_comb:

                        # reset voices in hvo
                        hvo_seq_in, hvo_seq_out = hvo_seq.reset_voices(voice_idx=v_idx)
                        # if the resulting hvos are 0, skip
                        if not np.any(hvo_seq_in.hvo.flatten()): continue
                        if not np.any(hvo_seq_out.hvo.flatten()): continue

                        # store hvo, v_idx
                        self.hvo_index.append(hvo_idx)
                        self.voices_reduced.append(v_idx)

                        # processed inputs
                        self.processed_inputs.append(hvo_seq_in.hvo)

                        # processed outputs complementary hvo_seq with reset voices
                        self.processed_outputs.append(hvo_seq_out.hvo)

        # current time
        dt_string = datetime.now().strftime("%d_%m_%Y_at_%H_%M_hrs")

        # dataset name
        if dataset_name is None: dataset_name = "Dataset_" + dt_string

        # dataset creation parameters
        parameters = {
            "dataset_name": dataset_name,
            "length": len(self.processed_inputs),
            "timestamp": dt_string,
            "subset_info": {**subset_info,
                            "max_len": max_len,
                            "max_aug_items": max_aug_items},
            "voices_parameters": voices_parameters,
            "dictionaries": {
                "hvo_index": self.hvo_index,
                "voices_reduced": self.voices_reduced,
            }
        }

        if wandb.ensure_configured(): # if running experiment file with wandb.init()
            wandb.config.update(parameters, allow_val_change=True) # update defaults

        # save parameters
        parameters_path = os.path.join('../result', dataset_name)
        if not os.path.exists(parameters_path): os.makedirs(parameters_path)
        parameters_json = os.path.join(parameters_path, 'parameters.json')
        with open(parameters_json, 'w') as f:
            json.dump(parameters, f, cls=NpEncoder)

        # convert inputs and outputs to torch tensors
        self.processed_inputs = torch.Tensor(self.processed_inputs).to(device=device)
        self.processed_outputs = torch.Tensor(self.processed_outputs).to(device=device)

    def get_hvo_sequence(self, idx):
        hvo_idx = self.hvo_index[idx]
        return self.hvo_sequences[hvo_idx]

    def get_voices_idx(self, idx):
        return self.voices_reduced[idx]

    def __len__(self):
        return len(self.processed_inputs)

    def __getitem__(self, idx):
        return self.processed_inputs[idx], self.processed_outputs[idx], idx
