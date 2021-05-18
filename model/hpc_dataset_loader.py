import sys
import numpy as np

sys.path.append('../../hvo_sequence/')
sys.path.append('../../preprocessed_dataset/')

from dataset_loader import GrooveMidiDataset
from Subset_Creators.subsetters import GrooveMidiSubsetter

if __name__ == "__main__":
    # GMD

    # load subset
    filters = {"beat_type": ["beat"],
               "master_id": ["drummer1/eval_session/10"]}

    # subset creator
    pickle_source_path = '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2/Processed_On_17_05_2021_at_22_32_hrs'
    subset_name = 'GrooveMIDI_processed_train'
    metadata_csv_filename = 'metadata.csv'
    hvo_pickle_filename = 'hvo_sequence_data.obj'

    gmd_subsetter = GrooveMidiSubsetter(
        pickle_source_path=pickle_source_path,
        subset=subset_name,
        hvo_pickle_filename=hvo_pickle_filename,
        list_of_filter_dicts_for_subsets=[filters],
    )
    _, subset_list = gmd_subsetter.create_subsets()

    subset_info = {"pickle_source_path": pickle_source_path,
                   "subset": subset_name,
                   "metadata_csv_filename": metadata_csv_filename,
                   "hvo_pickle_filename": hvo_pickle_filename,
                   "filters": filters}

    mso_parameters = {"sr": 44100,
                      "n_fft": 1024,
                      "win_length": 1024,
                      "hop_length": 441,
                      "n_bins_per_octave": 16,
                      "n_octaves": 9,
                      "f_min": 40,
                      "mean_filter_size": 22
                      }

    voices_parameters = {"voice_idx": [0, 1, 3, 5],
                         "min_n_voices_to_remove": 1,
                         "max_n_voices_to_remove": 3,
                         "prob": [1, 1, 1],
                         "k": 5}

    gmd = GrooveMidiDataset(subset=subset_list[0], subset_info=subset_info, mso_parameters=mso_parameters,
                            max_aug_items=100, voices_parameters=voices_parameters)
