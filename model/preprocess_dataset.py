import sys

sys.path.append('../../preprocessed_dataset/')
from dataset import GrooveMidiDatasetInfilling
from Subset_Creators.subsetters import GrooveMidiSubsetter

def preprocess_dataset(params):
    _, subset_list = GrooveMidiSubsetter(pickle_source_path=params["dataset"]["subset_info"]["pickle_source_path"],
                                         subset=params["dataset"]["subset_info"]["subset"],
                                         hvo_pickle_filename=params["dataset"]["subset_info"]["hvo_pickle_filename"],
                                         list_of_filter_dicts_for_subsets=[
                                             params['dataset']["subset_info"]['filters']]).create_subsets()

    dataset = GrooveMidiDatasetInfilling(data=subset_list[0], **params['dataset'])

    return dataset

def load_preprocessed_dataset(load_dataset_path):
    dataset = GrooveMidiDatasetInfilling(load_dataset_path=load_dataset_path)
    return dataset


if __name__ == "__main__":
    params = {
        "dataset": {
            "subset_info": {
                "pickle_source_path": '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.5/Processed_On_14_06_2021_at_14_26_hrs',
                "subset": 'GrooveMIDI_processed_train',
                "metadata_csv_filename": 'metadata.csv',
                "hvo_pickle_filename": 'hvo_sequence_data.obj',
                "filters": {
                    "beat_type": ["beat"],
                    "time_signature": ["4-4"],
                }
            },
            'max_len': 32,
            'mso_params': {'sr': 44100, 'n_fft': 1024, 'win_length': 1024, 'hop_length':
                441, 'n_bins_per_octave': 16, 'n_octaves': 9, 'f_min': 40, 'mean_filter_size': 22},
            'voices_params': {'voice_idx': [2], 'min_n_voices_to_remove': 1,  # closed hh
                              'max_n_voices_to_remove': 1, 'prob': [1], 'k': None},
            'sf_path': ['../soundfonts/filtered_soundfonts/Standard_Drum_Kit.sf2'],
            'max_n_sf': 1,
            'max_aug_items': 1,
            'dataset_name': None
        }
    }

    dataset = preprocess_dataset(params)

