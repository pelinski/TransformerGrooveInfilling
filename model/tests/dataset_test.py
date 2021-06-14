import sys
sys.path.insert(1, "../")
sys.path.append('../../../hvo_sequence/')
sys.path.append('../../../preprocessed_dataset/')

from dataset import GrooveMidiDataset
from Subset_Creators.subsetters import GrooveMidiSubsetter
import numpy as np

if __name__ == "__main__":
    #GMD
    # load subset
    filters = {"beat_type": ["beat"],
               "master_id": ["drummer1/eval_session/10"]}
    # subset creator
    pickle_source_path = '../../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.3.0/Processed_On_13_05_2021_at_12_56_hrs'
    subset_name = 'GrooveMIDI_processed_test'
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

    # check that inputs are mso
    check_inputs = False
    if check_inputs:
        # load gmd
        gmd = GrooveMidiDataset(subset=subset_list[0], subset_info=subset_info, mso_parameters=mso_parameters)
        # check that input corresponds to mso
        _in, _, _ = gmd.__getitem__(1)
        hvo_in = gmd.get_hvo_sequence(1)
        hvo_in.get_active_voices()
        hvo_in_reset, _ = hvo_in.reset_voices(voice_idx=gmd.get_voices_idx(1))
        mso = hvo_in_reset.mso(sf_path=gmd.get_soundfont(1))
        print(np.sum(_in.numpy() - mso > 1e-5))

    ## test dataset with kwargs
    dataset_parameters = {
        'max_len': 32,
        'mso_parameters': {'sr': 44100, 'n_fft': 1024, 'win_length': 1024, 'hop_length':
            441, 'n_bins_per_octave': 16, 'n_octaves': 9, 'f_min': 40, 'mean_filter_size': 22},
        'voices_parameters': {'voice_idx': [0, 1], 'min_n_voices_to_remove': 1,
                              'max_n_voices_to_remove': 2, 'prob': [1, 1], 'k': 5},
        'sf_path': '../../soundfonts/filtered_soundfonts/',
        'max_n_sf': None,
        'max_aug_items': 10,
        'dataset_name': "test_kwargs"
    }
    gmd = GrooveMidiDataset(subset=subset_list[0], subset_info=subset_info, **dataset_parameters)
    print(gmd.__len__(), len(gmd.hvo_sequences))
    hvo = gmd.get_hvo_sequence(1)
    print(hvo.get_active_voices())
