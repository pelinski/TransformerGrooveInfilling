import copy
from dataset import (
    GrooveMidiDatasetInfilling,
    GrooveMidiDatasetInfillingSymbolic,
    GrooveMidiDatasetInfillingRandom,
)

from src.preprocessed_dataset import GrooveMidiSubsetter

subset_info = {
    "pickle_source_path": "src/preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.5.2_submod/Processed_On_10_04_2022_at_01_28_hrs",
    "subset": "GrooveMIDI_processed_",
    "metadata_csv_filename": "metadata.csv",
    "hvo_pickle_filename": "hvo_sequence_data.obj",
    "filters": {"beat_type": ["beat"], "time_signature": ["4-4"],},
}

params = {
    "InfillingClosedHH": {
        "dataset_name": "InfillingClosedHH",
        "subset_info": subset_info,
        "mso_params": {
            "sr": 44100,
            "n_fft": 1024,
            "win_length": 1024,
            "hop_length": 441,
            "n_bins_per_octave": 16,
            "n_octaves": 9,
            "f_min": 40,
            "mean_filter_size": 22,
        },
        "voices_params": {
            "voice_idx": [2],
            "min_n_voices_to_remove": 1,
            "max_n_voices_to_remove": 1,
            "prob": [1],
            "k": None,
        },
        "sf_path": ["soundfonts/filtered_soundfonts/Standard_Drum_Kit.sf2"],
        "max_n_sf": 1,
        "max_aug_items": 1,
        "save_dataset_path": "datasets/InfillingClosedHH/",
    },
    "InfillingKicksAndSnares": {
        "dataset_name": "InfillingKicksAndSnares",
        "subset_info": subset_info,
        "mso_params": {
            "sr": 44100,
            "n_fft": 1024,
            "win_length": 1024,
            "hop_length": 441,
            "n_bins_per_octave": 16,
            "n_octaves": 9,
            "f_min": 40,
            "mean_filter_size": 22,
        },
        "voices_params": {
            "voice_idx": [0, 1],
            "min_n_voices_to_remove": 1,
            "max_n_voices_to_remove": 2,
            "prob": [1, 1],
            "k": 3,
        },
        "sf_path": "soundfonts/filtered_soundfonts/",
        "max_n_sf": 3,
        "max_aug_items": 4,
        "save_dataset_path": "datasets/InfillingKicksAndSnares/",
    },
    "InfillingMultipleVoices": {
        "dataset_name": "InfillingMultipleVoices",
        "subset_info": subset_info,
        "max_len": 32,
        "mso_params": {
            "sr": 44100,
            "n_fft": 1024,
            "win_length": 1024,
            "hop_length": 441,
            "n_bins_per_octave": 16,
            "n_octaves": 9,
            "f_min": 40,
            "mean_filter_size": 22,
        },
        "voices_params": {
            "voice_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "min_n_voices_to_remove": 1,
            "max_n_voices_to_remove": 2,
            "prob": [1, 1],
            "k": 3,
        },
        "sf_path": "soundfonts/filtered_soundfonts/",
        "max_n_sf": 3,
        "max_aug_items": 6,
        "save_dataset_path": "datasets/InfillingMultipleVoices/",
    },
    "InfillingRandom": {
        "dataset_name": "InfillingRandom",
        "subset_info": subset_info,
        "sf_path": "soundfonts/filtered_soundfonts/",
        "max_aug_items": 4,
        "thres_range": (0.4, 0.7),
        "save_dataset_path": "datasets/InfillingRandom/",
    },
    "InfillingRandomLow": {
        "dataset_name": "InfillingRandomLow",
        "subset_info": subset_info,
        "sf_path": "soundfonts/filtered_soundfonts/",
        "max_aug_items": 4,
        "thres_range": (0.1, 0.3),
        "save_dataset_path": "datasets/InfillingRandomLow/",
    },
    "InfillingClosedHH_Symbolic": {
        "dataset_name": "InfillingClosedHH_Symbolic",
        "subset_info": subset_info,
        "voices_params": {
            "voice_idx": [2],
            "min_n_voices_to_remove": 1,
            "max_n_voices_to_remove": 1,
            "prob": [1],
            "k": None,
        },
        "max_aug_items": 1,
        "save_dataset_path": "datasets/InfillingClosedHH_Symbolic/",
    },
}


def preprocess_dataset(params, exp):
    _, subset_list = GrooveMidiSubsetter(
        pickle_source_path=params["subset_info"]["pickle_source_path"],
        subset=params["subset_info"]["subset"],
        hvo_pickle_filename=params["subset_info"]["hvo_pickle_filename"],
        list_of_filter_dicts_for_subsets=[params["subset_info"]["filters"]],
    ).create_subsets()

    if exp == "InfillingClosedHH_Symbolic":
        _dataset = GrooveMidiDatasetInfillingSymbolic(data=subset_list[0], **params)
    elif exp == "InfillingRandom" or exp == "InfillingRandomLow":
        _dataset = GrooveMidiDatasetInfillingRandom(data=subset_list[0], **params)
    else:
        _dataset = GrooveMidiDatasetInfilling(data=subset_list[0], **params)

    return _dataset


def load_preprocessed_dataset(load_dataset_path, exp):
    if exp == "InfillingClosedHH_Symbolic":
        print("Loading GrooveMidiDatasetInfillingSymbolic...")
        _dataset = GrooveMidiDatasetInfillingSymbolic(
            load_dataset_path=load_dataset_path
        )
    elif exp == "InfillingRandom" or exp == "InfillingRandomLow":
        print("Loading GrooveMidiDatasetInfillingRandom...")
        _dataset = GrooveMidiDatasetInfillingRandom(load_dataset_path=load_dataset_path)
    else:
        print("Loading GrooveMidiDatasetInfilling...")
        _dataset = GrooveMidiDatasetInfilling(load_dataset_path=load_dataset_path)

    return _dataset


if __name__ == "__main__":

    testing = True

    # change experiment and split here
    exps = ["InfillingClosedHH"]
    splits = ["train", "test", "validation"]

    for exp in exps:
        if testing:
            params[exp]["subset_info"]["filters"]["master_id"] = ["drummer2/session2/8"]
            params[exp]["dataset_name"] = params[exp]["dataset_name"] + "_testing"
            params[exp]["save_dataset_path"] = (
                "datasets/" + params[exp]["dataset_name"] + "/"
            )

        print(
            "------------------------\n"
            + params[exp]["dataset_name"]
            + "\n------------------------\n"
        )

        for split in splits:
            params_exp = copy.deepcopy(params[exp])
            params_exp["split"] = split
            params_exp["subset_info"]["subset"] = (
                params_exp["subset_info"]["subset"] + split
            )

            preprocess_dataset(params_exp, exp=exp)
