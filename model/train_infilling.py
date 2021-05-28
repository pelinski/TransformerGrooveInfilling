import sys
import torch

from dataset import GrooveMidiDataset
sys.path.append("../../BaseGrooveTransformers/models/")
from train import initialize_model, load_dataset, calculate_loss, train_loop

if __name__ == "__main__":

    save_info = {
        'checkpoint_path': '../train_results/',
        'checkpoint_save_str': '../train_results/transformer_groove_infilling-epoch-{}',
        'df_path': '../train_results/losses_df/'
    }

    filters = {
        "beat_type": ["beat"],
        "time_signature": ["4-4"],
        "master_id": ["drummer9/session1/8"]
    }

    subset_info = {
        "pickle_source_path": '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2'
                              '/Processed_On_17_05_2021_at_22_32_hrs',
        "subset": 'GrooveMIDI_processed_train',
        "metadata_csv_filename": 'metadata.csv',
        "hvo_pickle_filename": 'hvo_sequence_data.obj',
        "filters": filters
    }

    # TRANSFORMER MODEL PARAMETERS
    model_parameters = {
        'd_model': 128,
        'n_heads': 8,
        'dim_feedforward': 1280,
        'dropout': 0.1,
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,
        'max_len': 32,
        'embedding_size_src': 16,  # mso
        'embedding_size_tgt': 27,  # hvo
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # TRAINING PARAMETERS
    training_parameters = {
        'learning_rate': 1e-3,
        'batch_size': 64
    }

    # PYTORCH LOSS FUNCTIONS
    BCE_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    MSE_fn = torch.nn.MSELoss(reduction='none')

    model, optimizer, ep = initialize_model(model_parameters, training_parameters, save_info,
                                            load_from_checkpoint=False)
    dataset_parameters = {
        'max_len': 32,
        'mso_parameters': {'sr': 44100, 'n_fft': 1024, 'win_length': 1024, 'hop_length':
            441, 'n_bins_per_octave': 16, 'n_octaves': 9, 'f_min': 40, 'mean_filter_size': 22},
        'voices_parameters': {'voice_idx': [2], 'min_n_voices_to_remove': 1,    # closed hh
                              'max_n_voices_to_remove': 1, 'prob': [1], 'k': None},
        'sf_path': '../soundfonts/filtered_soundfonts/Standard_Drum_Kit.sf2',
        'max_n_sf': 1,
        'max_aug_items': 1,
        'dataset_name': None
    }

    dataloader = load_dataset(GrooveMidiDataset, subset_info, filters, training_parameters['batch_size'], dataset_parameters)

    epoch_save_div = 100

    while True:
        ep += 1
        print(f"Epoch {ep}\n-------------------------------")
        train_loop(dataloader=dataloader, groove_transformer=model, opt=optimizer, epoch=ep,
                   loss_fn=calculate_loss, bce_fn=BCE_fn, mse_fn=MSE_fn, save_epoch=epoch_save_div, cp_info=save_info,
                   device=model_parameters['device'])
        print("-------------------------------\n")
