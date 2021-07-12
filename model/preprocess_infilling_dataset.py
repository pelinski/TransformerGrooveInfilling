import sys

sys.path.append('../../preprocessed_dataset/')
from dataset import GrooveMidiDatasetInfilling, GrooveMidiDatasetInfillingSymbolic
from utils import save_parameters_to_json
from Subset_Creators.subsetters import GrooveMidiSubsetter


def preprocess_dataset(params, symbolic=False):
    _, subset_list = GrooveMidiSubsetter(pickle_source_path=params["dataset"]["subset_info"]["pickle_source_path"],
                                         subset=params["dataset"]["subset_info"]["subset"],
                                         hvo_pickle_filename=params["dataset"]["subset_info"]["hvo_pickle_filename"],
                                         list_of_filter_dicts_for_subsets=[
                                             params['dataset']["subset_info"]['filters']]).create_subsets()

    _dataset = GrooveMidiDatasetInfilling(data=subset_list[0], **params['dataset']) if not symbolic else \
        GrooveMidiDatasetInfillingSymbolic(data=subset_list[0], **params['dataset'])

    return _dataset


def load_preprocessed_dataset(load_dataset_path, symbolic=False):
    _dataset = GrooveMidiDatasetInfilling(load_dataset_path=load_dataset_path) if not symbolic else \
        GrooveMidiDatasetInfillingSymbolic(load_dataset_path=load_dataset_path)

    return _dataset

"""
ROLAND_REDUCED_MAPPING = {
    0_ "KICK": [36],
    1_ "SNARE": [38, 37, 40],
    2_ "HH_CLOSED": [42, 22, 44],
    3_ "HH_OPEN": [46, 26],
    4_ "TOM_3_LO": [43, 58],
    5_ "TOM_2_MID": [47, 45],
    6_ "TOM_1_HI": [50, 48],
    7_ "CRASH": [49, 52, 55, 57],
    8_ "RIDE":  [51, 53, 59]
}
"""

if __name__ == "__main__":
    split = 'test'
    params = {
        "dataset": {
            "subset_info": {
                "pickle_source_path": '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.5/Processed_On_14_06_2021_at_14_26_hrs',
                "subset": 'GrooveMIDI_processed_' + split,
                "metadata_csv_filename": 'metadata.csv',
                "hvo_pickle_filename": 'hvo_sequence_data.obj',
                "filters": {
                    "beat_type": ["beat"],
                    "time_signature": ["4-4"],
                    #  "master_id": ["drummer1/session1/201"] rapid testing
                }
            },
            'max_len': 32,
            'mso_params': {'sr': 44100, 'n_fft': 1024, 'win_length': 1024, 'hop_length':
                441, 'n_bins_per_octave': 16, 'n_octaves': 9, 'f_min': 40, 'mean_filter_size': 22},
            'voices_params': {'voice_idx': [0,1,2,3,4,5,6,7,8], 'min_n_voices_to_remove': 1,  # closed hh
                              'max_n_voices_to_remove': 2, 'prob': [1,1], 'k': 3},
            'sf_path': '../soundfonts/filtered_soundfonts/',
            'max_n_sf': 3,
            'max_aug_items': 6,
            'dataset_name': 'exp2'
        }
    }
    print(params["dataset"]["subset_info"])
    save_parameters_to_json(params["dataset"], params_path='./') #experiment parameters
    #preprocess_dataset(params, symbolic=False)
