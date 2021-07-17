import copy
import sys
from dataset import GrooveMidiDatasetInfilling, GrooveMidiDatasetInfillingSymbolic, GrooveMidiDatasetInfillingRandom

sys.path.append('../../preprocessed_dataset/')
from Subset_Creators.subsetters import GrooveMidiSubsetter

subset_info = {
    "pickle_source_path": "../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.5.1/Processed_On_16_07_2021_at_19_17_hrs",
    "subset": "GrooveMIDI_processed_",
    "metadata_csv_filename": "metadata.csv",
    "hvo_pickle_filename": "hvo_sequence_data.obj",
    "filters": {
        "beat_type": ["beat"],
        "time_signature": ["4-4"],
#        "master_id": ["drummer2/session2/8"] # testing

    }
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
            "mean_filter_size": 22
        },
        "voices_params": {
            "voice_idx": [2],
            "min_n_voices_to_remove": 1,
            "max_n_voices_to_remove": 1,
            "prob": [1],
            "k": None
        },
        "sf_path": ["../soundfonts/filtered_soundfonts/Standard_Drum_Kit.sf2"],
        "max_n_sf": 1,
        "max_aug_items": 1,
        "save_dataset_path": '../preprocessed_infilling_datasets/InfillingClosedHH/'

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
            "mean_filter_size": 22
        },
        "voices_params": {
            "voice_idx": [0, 1],
            "min_n_voices_to_remove": 1,
            "max_n_voices_to_remove": 2,
            "prob": [1, 1],
            "k": 3
        },
        "sf_path": "../soundfonts/filtered_soundfonts/",
        "max_n_sf": 3,
        "max_aug_items": 4,
        "save_dataset_path": '../preprocessed_infilling_datasets/InfillingKicksAndSnares/'
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
            "mean_filter_size": 22
        },
        "voices_params": {
            "voice_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "min_n_voices_to_remove": 1,
            "max_n_voices_to_remove": 2,
            "prob": [1, 1],
            "k": 3
        },
        "sf_path": "../soundfonts/filtered_soundfonts/",
        "max_n_sf": 3,
        "max_aug_items": 6,
        "save_dataset_path": '../preprocessed_infilling_datasets/InfillingMultipleVoices/'
    },

    "InfillingRandom": {
        "dataset_name": "InfillingRandom",
        "subset_info": subset_info,
        "sf_path": "../soundfonts/filtered_soundfonts/",
        "max_aug_items": 4,
        "thres_range": (0.4, 0.7),
        "save_dataset_path": '../preprocessed_infilling_datasets/InfillingRandom/'
    }

}


def preprocess_dataset(params, exp):
    _, subset_list = GrooveMidiSubsetter(pickle_source_path=params["subset_info"]["pickle_source_path"],
                                         subset=params["subset_info"]["subset"],
                                         hvo_pickle_filename=params["subset_info"]["hvo_pickle_filename"],
                                         list_of_filter_dicts_for_subsets=[
                                             params["subset_info"]['filters']]).create_subsets()

    if exp == 'InfillingSymbolic':
        _dataset = GrooveMidiDatasetInfillingSymbolic(data=subset_list[0], **params)
    elif exp == 'InfillingRandom':
        _dataset = GrooveMidiDatasetInfillingRandom(data=subset_list[0], **params)
    else:
        _dataset = GrooveMidiDatasetInfilling(data=subset_list[0], **params)

    return _dataset


def load_preprocessed_dataset(load_dataset_path, exp):
    if exp == 'InfillingSymbolic':
        _dataset = GrooveMidiDatasetInfillingSymbolic(load_dataset_path=load_dataset_path)
    elif exp == 'InfillingRandom':
        _dataset = GrooveMidiDatasetInfillingRandom(load_dataset_path=load_dataset_path)
    else:
        _dataset = GrooveMidiDatasetInfilling(load_dataset_path=load_dataset_path)

    return _dataset


if __name__ == "__main__":
    # change experiment and split here
    # exps = ['InfillingRandom', 'InfillingMultipleVoices', 'InfillingClosedHH']
    exps = ['InfillingKicksAndSnares', 'InfillingClosedHH','InfillingMultipleVoices' ]
    splits = ['train', 'test', 'validation']

    for exp in exps:
        print('------------------------\n'+exp+'\n------------------------\n')
        for split in splits:
            params_exp = copy.deepcopy(params[exp])
            params_exp['split'] = split
            params_exp['subset_info']['subset'] = params_exp['subset_info']['subset'] + split

            preprocess_dataset(params_exp, exp=exp)
