import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import wandb
import pickle
import random
import copy

from hvo_sequence.hvo_sequence.hvo_seq import HVO_Sequence

from utils import (
    get_sf_list,
    pad_to_match_max_seq_len,
    get_voice_idx_for_item,
    get_sf_v_combinations,
    get_voice_combinations,
    save_to_pickle,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GrooveMidiDatasetInfilling(Dataset):
    def __init__(self, data=None, load_dataset_path=None, **kwargs):
        """
        Groove Midi Dataset Loader. Max number of items in dataset is N x M x K where N is the number of items in the
        subset, M the maximum number of soundfonts to sample from for each item (max_n_sf) and K is the maximum number
        of voice combinations.

        @param data:              GrooveMidiDataset subset generated with the Subset_Creator
        @param subset_info:         Dictionary with the routes and filters passed to the Subset_Creator to generate the
                                    subset. Example:
                                    subset_info = {
                                    "pickle_source_path": '../../processed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2'
                                               '/Processed_On_17_05_2021_at_22_32_hrs',
                                    "subset": 'GrooveMIDI_processed_train',
                                    "metadata_csv_filename": 'metadata.csv',
                                    "hvo_pickle_filename": 'hvo_sequence_data.obj',
                                    "filters": "bpm": ["beat"],
                                    }

        @param max_seq_len:             Max_length of sequences
        @param mso_params:      Dictionary with the parameters for calculating the Multiband Synthesized Onsets.
                                    Refer to `hvo_sequence.hvo_seq.mso()` for the documentation
        @param voices_params:   Dictionary with parameters for generating the combinations of the voices to remove
                                    Refer to utils.get_voice_combinations for documentation
        @param sf_path:             Path with soundfonts
        @param max_n_sf:            Maximum number of soundfonts to sample from for each example
        @param max_aug_items:       Maximum number of synthesized examples per example in subset
        @param dataset_name:        Dataset name (for experiment tracking)
        """
        self.__version__ = "0.1.2"
        # pickle file module not found fix
        self.__module__ = "dataset"

        # get params
        if load_dataset_path:
            self.dataset_name = (
                load_dataset_path.split("/")[-1]
                if load_dataset_path.split("/")[-1]
                else load_dataset_path.split("/")[-2]
            )
            self.load_params_from_pickle(load_dataset_path)
        else:
            # default values for kwargs
            self.max_seq_len = kwargs.get("max_seq_len", 32)
            self.mso_params = kwargs.get(
                "mso_params",
                {
                    "sr": 44100,
                    "n_fft": 1024,
                    "win_length": 1024,
                    "hop_length": 441,
                    "n_bins_per_octave": 16,
                    "n_octaves": 9,
                    "f_min": 40,
                    "mean_filter_size": 22,
                },
            )
            self.voices_params = kwargs.get(
                "voices_params",
                {
                    "voice_idx": [0, 1],
                    "min_n_voices_to_remove": 1,
                    "max_n_voices_to_remove": 2,
                    "prob": [1, 1],
                    "k": 5,
                },
            )
            self.sf_path = kwargs.get("sf_path", "../soundfonts/filtered_soundfonts/")
            self.max_n_sf = kwargs.get("max_n_sf", None)
            self.max_aug_items = kwargs.get("max_aug_items", 10)
            self.timestamp = datetime.now().strftime("%d_%m_%Y_at_%H_%M_hrs")
            self.dataset_name = (
                "Dataset_" + self.timestamp
                if kwargs.get("dataset_name") is None
                else kwargs.get("dataset_name", "Dataset_" + self.timestamp)
            )
            self.subset_info = kwargs.get(
                "subset_info",
                {
                    "pickle_source_path": "",
                    "subset": "",
                    "metadata_csv_filename": "",
                    "hvo_pickle_filename": "",
                    "filters": "",
                },
            )
            self.split = kwargs.get("split", "")

            self.sfs_list = get_sf_list(self.sf_path)

            if self.max_n_sf is not None:
                assert self.max_n_sf <= len(self.sfs_list), (
                    "max_n_sf can not be larger than number of available " "soundfonts"
                )

            self.save_dataset_path = kwargs.get(
                "save_dataset_path", os.path.join("../dataset", self.dataset_name)
            )
            """
            self.metadata = pd.read_csv(os.path.join(self.subset_info["pickle_source_path"], self.subset_info["subset"],
                                                     self.subset_info["metadata_csv_filename"]))
            """
        # process dataset
        print("GMD path: ", self.subset_info["pickle_source_path"])
        processed_dataset = (
            self.load_dataset_from_pickle(load_dataset_path)
            if load_dataset_path
            else self.process_dataset(data)
        )

        # store processed dataset in dataset attrs
        for key in processed_dataset.keys():
            self.__setattr__(key, processed_dataset[key])

        # dataset params dict
        params = self.get_params()

        # log params to wandb
        if wandb.ensure_configured():  # if running experiment file with wandb.init()
            wandb.config.update(params, allow_val_change=True)  # update defaults

        # save dataset to pickle file
        if load_dataset_path is None:
            if not os.path.exists(self.save_dataset_path):
                os.makedirs(self.save_dataset_path)
            print(self.save_dataset_path)
            # move tensor to cpu (tensors saved while on gpu cannot be loaded from pickle file in cpu)
            processed_dataset["processed_inputs"] = processed_dataset[
                "processed_inputs"
            ].to(device="cpu")
            processed_dataset["processed_outputs"] = processed_dataset[
                "processed_outputs"
            ].to(device="cpu")

            # save to pickle
            params_pickle_filename = os.path.join(
                self.save_dataset_path,
                self.dataset_name
                + "_"
                + self.split
                + "_"
                + self.__version__
                + "_params.pickle",
            )
            dataset_pickle_filename = os.path.join(
                self.save_dataset_path,
                self.dataset_name
                + "_"
                + self.split
                + "_"
                + self.__version__
                + "_dataset.pickle",
            )
            save_to_pickle(params, params_pickle_filename)
            save_to_pickle(processed_dataset, dataset_pickle_filename)

            print("Saved dataset to path: ", self.save_dataset_path)

    def process_dataset(self, data):
        self.save_dataset_path = os.path.join(
            os.path.join(self.save_dataset_path, self.__version__), self.split
        )

        print("GrooveMidiDatasetInfilling version " + self.__version__)

        # init lists to store hvo sequences and processed io
        hvo_sequences = []
        hvo_sequences_inputs, hvo_sequences_outputs = [], []
        processed_inputs, processed_outputs = [], []
        unused_items = []

        # init list with configurations
        hvo_index, voices_reduced, soundfonts = [], [], []

        for hvo_idx, hvo_seq in enumerate(
            tqdm(
                data, desc="processing dataset {}".format(self.subset_info["subset"])
            )
        ):

            all_zeros = not np.any(hvo_seq.hvo.flatten())  # silent patterns

            if (
                len(hvo_seq.time_signatures) == 1 and not all_zeros
            ):  # ignore if time_signature change happens

                # add metadata to hvo_seq scores
                # add_metadata_to_hvo_seq(hvo_seq, hvo_idx, self.metadata)

                # pad with zeros to match max_len
                hvo_seq = pad_to_match_max_seq_len(hvo_seq, self.max_seq_len)

                # append hvo_seq to hvo_sequences list
                hvo_sequences.append(hvo_seq)

                # remove voices in voice_idx not present in item
                _voice_idx, _voices_params = get_voice_idx_for_item(
                    hvo_seq, self.voices_params
                )
                if len(_voice_idx) == 0:
                    unused_items.append(hvo_idx)
                    continue  # if there are no voices to remove, continue

                # get voices and sf combinations
                sf_v_comb = get_sf_v_combinations(
                    _voices_params, self.max_aug_items, self.max_n_sf, self.sfs_list
                )

                # for every sf and voice combination
                for sf, v_idx in sf_v_comb:

                    # reset voices in hvo
                    hvo_seq_in, hvo_seq_out = hvo_seq.reset_voices(voice_idx=v_idx)
                    # if the resulting hvos are 0, skip
                    if not np.any(hvo_seq_in.hvo.flatten()) or not np.any(
                        hvo_seq_out.hvo.flatten()
                    ):
                        unused_items.append(hvo_idx)
                        continue

                    hvo_sequences_inputs.append(hvo_seq_in)
                    hvo_sequences_outputs.append(hvo_seq_out)

                    # store hvo, v_idx and sf
                    hvo_index.append(hvo_idx)
                    voices_reduced.append(v_idx)
                    soundfonts.append(sf)

                    # processed inputs: mso
                    mso = hvo_seq_in.mso(sf_path=sf, **self.mso_params)
                    processed_inputs.append(mso)

                    # processed outputs complementary hvo_seq with reset voices
                    processed_outputs.append(hvo_seq_out.hvo)

        # convert inputs and outputs to torch tensors
        processed_inputs = torch.Tensor(processed_inputs).to(device=device)
        processed_outputs = torch.Tensor(processed_outputs).to(device=device)

        processed_dict = {
            "processed_inputs": processed_inputs,
            "processed_outputs": processed_outputs,
            "hvo_sequences": hvo_sequences,
            "hvo_sequences_inputs": hvo_sequences_inputs,
            "hvo_sequences_outputs": hvo_sequences_outputs,
            "hvo_index": hvo_index,
            "voices_reduced": voices_reduced,
            "soundfonts": soundfonts,
            "unused_items": unused_items,
        }

        return processed_dict

    # load from pickle

    def load_params_from_pickle(self, dataset_path):
        params_file = os.path.join(
            dataset_path,
            list(
                filter(lambda x: x.endswith("_params.pickle"), os.listdir(dataset_path))
            )[0],
        )

        with open(params_file, "rb") as f:
            params = pickle.load(f)

        for key in params.keys():
            self.__setattr__(key, params[key])

        print("Loaded parameters from path: ", params_file)

    def load_dataset_from_pickle(self, dataset_path):
        pickle_file = os.path.join(
            dataset_path,
            list(
                filter(
                    lambda x: x.endswith("_dataset.pickle"), os.listdir(dataset_path)
                )
            )[0],
        )

        with open(pickle_file, "rb") as f:
            processed_dataset = pickle.load(f)

        for key in processed_dataset.keys():
            self.__setattr__(key, processed_dataset[key])

        print("Loaded dataset from path: ", pickle_file)

        print(str(self.__len__()) + " items")

        return processed_dataset

    # getters

    def get_hvo_sequence(self, idx):
        hvo_idx = self.hvo_index[idx]
        return self.hvo_sequences[hvo_idx]

    def get_soundfont(self, idx):
        return self.soundfonts[idx]

    def get_voices_idx(self, idx):
        return self.voices_reduced[idx]

    def get_params(self):
        params = copy.deepcopy(self.__dict__)

        # delete dataset attr
        params["hvo_sequences_inputs"] = {}
        params["hvo_sequences_outputs"] = {}
        params["processed_inputs"] = {}
        params["processed_outputs"] = {}
        params["hvo_sequences"] = {}

        del params["hvo_sequences_inputs"]
        del params["hvo_sequences_outputs"]
        del params["processed_inputs"]
        del params["processed_outputs"]
        del params["hvo_sequences"]

        return params

    # dataset methods

    def __len__(self):
        return len(self.processed_inputs)

    def __getitem__(self, idx):
        return self.processed_inputs[idx], self.processed_outputs[idx], idx


class GrooveMidiDatasetInfillingSymbolic(GrooveMidiDatasetInfilling):
    def __init__(self, data=None, load_dataset_path=None, **kwargs):

        super(GrooveMidiDatasetInfillingSymbolic, self).__init__(
            data=data, load_dataset_path=load_dataset_path, **kwargs
        )

        # audio attrs inherited from GMDInfilling
        del self.mso_params
        del self.sfs_list
        del self.sf_path
        del self.max_n_sf

    # override processing dataset method
    # keep unused audio attrs (sfs) for simplicity
    def process_dataset(self, data):
        self.__version__ = "0.1.1"
        self.save_dataset_path = os.path.join(
            os.path.join(self.save_dataset_path, self.__version__), self.split
        )
        print("GrooveMidiDatasetInfillingSymbolic version " + self.__version__)

        # init lists to store hvo sequences and processed io
        hvo_sequences = []
        hvo_sequences_inputs, hvo_sequences_outputs = [], []
        processed_inputs, processed_outputs = [], []

        # init list with configurations
        hvo_index, voices_reduced = [], []
        unused_items = []

        for hvo_idx, hvo_seq in enumerate(
            tqdm(
                data, desc="processing dataset {}".format(self.subset_info["subset"])
            )
        ):

            all_zeros = not np.any(hvo_seq.hvo.flatten())  # silent patterns

            if (
                len(hvo_seq.time_signatures) == 1 and not all_zeros
            ):  # ignore if time_signature change happens
                # add metadata to hvo_seq scores
                # add_metadata_to_hvo_seq(hvo_seq, hvo_idx, self.metadata)

                # pad with zeros to match max_len
                hvo_seq = pad_to_match_max_seq_len(hvo_seq, self.max_seq_len)

                # append hvo_seq to hvo_sequences list
                hvo_sequences.append(hvo_seq)

                # remove voices in voice_idx not present in item
                _voice_idx, _voices_params = get_voice_idx_for_item(
                    hvo_seq, self.voices_params
                )
                if len(_voice_idx) == 0:
                    unused_items.append(hvo_idx)
                    continue  # if there are no voices to remove, continue

                # get voices and sf combinations
                v_comb = get_voice_combinations(**_voices_params)

                # for every sf and voice combination
                for v_idx in v_comb:

                    # reset voices in hvo
                    hvo_seq_in, hvo_seq_out = hvo_seq.reset_voices(voice_idx=v_idx)
                    # if the resulting hvos are 0, skip
                    if not np.any(hvo_seq_in.hvo.flatten()) or not np.any(
                        hvo_seq_out.hvo.flatten()
                    ):
                        unused_items.append(hvo_idx)
                        continue

                    hvo_sequences_inputs.append(hvo_seq_in)
                    hvo_sequences_outputs.append(hvo_seq_out)

                    # store hvo, v_idx and sf
                    hvo_index.append(hvo_idx)
                    voices_reduced.append(v_idx)

                    # processed inputs
                    processed_inputs.append(hvo_seq_in.hvo)

                    # processed outputs complementary hvo_seq with reset voices
                    processed_outputs.append(hvo_seq_out.hvo)

        # convert inputs and outputs to torch tensors
        processed_inputs = torch.Tensor(processed_inputs).to(device=device)
        processed_outputs = torch.Tensor(processed_outputs).to(device=device)

        processed_dict = {
            "processed_inputs": processed_inputs,
            "processed_outputs": processed_outputs,
            "hvo_sequences": hvo_sequences,
            "hvo_sequences_inputs": hvo_sequences_inputs,
            "hvo_sequences_outputs": hvo_sequences_outputs,
            "hvo_index": hvo_index,
            "voices_reduced": voices_reduced,
            "unused_items": unused_items,
        }

        return processed_dict


class GrooveMidiDatasetInfillingRandom(GrooveMidiDatasetInfilling):
    def __init__(self, data=None, load_dataset_path=None, **kwargs):

        self.thres_range = kwargs.get("thres_range", (0.4, 0.6))

        super(GrooveMidiDatasetInfillingRandom, self).__init__(
            data=data, load_dataset_path=load_dataset_path, **kwargs
        )

        # del voices attrs inherited from GMD Infilling
        del self.voices_params
        del self.max_n_sf

    # override processing dataset method
    def process_dataset(self, data):
        self.__version__ = "0.0.0"
        self.save_dataset_path = os.path.join(
            os.path.join(self.save_dataset_path, self.__version__), self.split
        )
        print("GrooveMidiDatasetInfillingRandom version " + self.__version__)

        # init lists to store hvo sequences and processed io
        hvo_sequences = []
        hvo_sequences_inputs, hvo_sequences_outputs = [], []
        processed_inputs, processed_outputs = [], []
        unused_items = []

        # init list with configurations
        hvo_index, soundfonts = [], []

        for hvo_idx, hvo_seq in enumerate(
            tqdm(
                data, desc="processing dataset {}".format(self.subset_info["subset"])
            )
        ):
            all_zeros = not np.any(hvo_seq.hvo.flatten())  # silent patterns

            if (
                len(hvo_seq.time_signatures) == 1 and not all_zeros
            ):  # ignore if time_signature change happens

                # add metadata to hvo_seq scores
                # add_metadata_to_hvo_seq(hvo_seq, hvo_idx, self.metadata)

                # pad with zeros to match max_len
                hvo_seq = pad_to_match_max_seq_len(hvo_seq, self.max_seq_len)

                # append hvo_seq to hvo_sequences list
                hvo_sequences.append(hvo_seq)

                for i in range(self.max_aug_items):

                    hvo_seq_in, hvo_seq_out = hvo_seq.remove_random_events(
                        thres_range=self.thres_range
                    )

                    if not np.any(hvo_seq_in.hvo.flatten()) or not np.any(
                        hvo_seq_out.hvo.flatten()
                    ):
                        unused_items.append(hvo_idx)
                        continue

                    hvo_sequences_inputs.append(hvo_seq_in)
                    hvo_sequences_outputs.append(hvo_seq_out)

                    # processed inputs: mso
                    sf = random.choice(self.sfs_list)
                    mso = hvo_seq_in.mso(sf_path=sf, **self.mso_params)
                    processed_inputs.append(mso)

                    hvo_index.append(hvo_idx)
                    soundfonts.append(sf)

                    # processed outputs
                    processed_outputs.append(hvo_seq_out.hvo)

        # convert inputs and outputs to torch tensors
        processed_inputs = torch.Tensor(processed_inputs).to(device=device)
        processed_outputs = torch.Tensor(processed_outputs).to(device=device)

        processed_dict = {
            "processed_inputs": processed_inputs,
            "processed_outputs": processed_outputs,
            "hvo_sequences": hvo_sequences,
            "hvo_sequences_inputs": hvo_sequences_inputs,
            "hvo_sequences_outputs": hvo_sequences_outputs,
            "hvo_index": hvo_index,
            "soundfonts": soundfonts,
            "unused_items": unused_items,
        }

        return processed_dict
