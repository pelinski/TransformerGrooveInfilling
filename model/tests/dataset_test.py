import sys
import torch
sys.path.insert(1, "../")
sys.path.append('../../../hvo_sequence/')
sys.path.append('../../../preprocessed_dataset/')

from dataset import GrooveMidiDatasetInfilling
from Subset_Creators.subsetters import GrooveMidiSubsetter
import numpy as np

if __name__ == "__main__":

    params = {
        "model": {
            'optimizer': 'sgd',
            'd_model': 128,
            'n_heads': 8,
            'dim_feedforward': 1280,
            'dropout': 0.1,
            'num_encoder_layers': 5,
            'num_decoder_layers': 5,
            'max_len': 32,
            'embedding_size_src': 16,  # mso
            'embedding_size_tgt': 27,  # hvo
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        "training": {
            'learning_rate': 1e-3,
            'batch_size': 64,
            'lr_scheduler_step_size': 30,
            'lr_scheduler_gamma': 0.1
        },
        "dataset": {
            "pickle_source_path": '../../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.5/Processed_On_14_06_2021_at_14_26_hrs',
            "subset": 'GrooveMIDI_processed_train',
            "metadata_csv_filename": 'metadata.csv',
            "hvo_pickle_filename": 'hvo_sequence_data.obj',
            "filters": {
                "beat_type": ["beat"],
                "time_signature": ["4-4"],
                # "master_id": ["drummer9/session1/8"]
                "master_id": ["drummer1/session1/201"]
            },
            'max_len': 32,
            'mso_params': {'sr': 44100, 'n_fft': 1024, 'win_length': 1024, 'hop_length':
                441, 'n_bins_per_octave': 16, 'n_octaves': 9, 'f_min': 40, 'mean_filter_size': 22},
            'voices_params': {'voice_idx': [2], 'min_n_voices_to_remove': 1,  # closed hh
                              'max_n_voices_to_remove': 1, 'prob': [1], 'k': None},
            'sf_path': ['../../soundfonts/filtered_soundfonts/Standard_Drum_Kit.sf2'],
            'max_n_sf': 1,
            'max_aug_items': 1,
            'dataset_name': None
        },
        "evaluator": {"n_samples_to_use": 12,
                      "n_samples_to_synthesize_visualize_per_subset": 10},
        "cp_paths": {
            'checkpoint_path': '../train_results/',
            'checkpoint_save_str': '../train_results/transformer_groove_infilling-epoch-{}'
        },
        "load_model": None,
    }

    #GMD

    _, subset_list = GrooveMidiSubsetter(pickle_source_path=params["dataset"]["pickle_source_path"],
                                         subset=params["dataset"]["subset"],
                                         hvo_pickle_filename=params["dataset"]["hvo_pickle_filename"],
                                         list_of_filter_dicts_for_subsets=[
                                             params['dataset']['filters']]).create_subsets()

    # check that inputs are mso
    check_inputs = False
    if check_inputs:
        # load gmd
        gmd = GrooveMidiDatasetInfilling(data=subset_list[0], **params['dataset'])
        # check that input corresponds to mso
        _in, _, _ = gmd.__getitem__(1)
        hvo_in = gmd.get_hvo_sequence(1)
        hvo_in.get_active_voices()
        hvo_in_reset, _ = hvo_in.reset_voices(voice_idx=gmd.get_voices_idx(1))
        mso = hvo_in_reset.mso(sf_path=gmd.get_soundfont(1))
        print(np.sum(_in.numpy() - mso > 1e-5))

    gmd = GrooveMidiDatasetInfilling(data=subset_list[0], **params['dataset'])
    print(gmd.__len__(), len(gmd.hvo_sequences))
    hvo = gmd.get_hvo_sequence(1)
    print(hvo.get_active_voices())
