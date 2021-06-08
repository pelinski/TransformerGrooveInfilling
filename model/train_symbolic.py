import sys
import torch
import wandb
import numpy as np

from dataset_symbolic import GrooveMidiDatasetSymbolic

sys.path.insert(1, "../../BaseGrooveTransformers/")
sys.path.insert(1, "../BaseGrooveTransformers/")
sys.path.insert(1, "../../GrooveEvaluator")
sys.path.insert(1, "../GrooveEvaluator")
from models.train import initialize_model, load_dataset, calculate_loss, train_loop
from GrooveEvaluator.evaluator import Evaluator

# disable wandb for testing
# import os
# os.environ['WANDB_MODE'] = 'offline'

if __name__ == "__main__":

    hyperparameter_defaults = dict(
        optimizer_algorithm='sgd',
        d_model=128,
        n_heads=8,
        dropout=0.1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        learning_rate=1e-3,
        batch_size=64,
        dim_feedforward=1280,
        epochs=100,
        lr_scheduler_step_size=30,
        lr_scheduler_gamma=0.1
    )

    wandb.init(config=hyperparameter_defaults,project='infilling-symbolic')


    save_info = {
        'checkpoint_path': '../symbolic_train_results/',
        'checkpoint_save_str': '../symbolic_train_results/transformer_groove_infilling_sym-epoch-{}',
        'df_path': '../symbolic_train_results/losses_df/'
    }

    filters = {
        "beat_type": ["beat"],
        "time_signature": ["4-4"],
        #"master_id": ["drummer9/session1/8"]
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
        'optimizer': wandb.config.optimizer_algorithm,
        'd_model': wandb.config.d_model,
        'n_heads': wandb.config.n_heads,
        'dim_feedforward': wandb.config.dim_feedforward,
        'dropout': wandb.config.dropout,
        'num_encoder_layers': wandb.config.num_encoder_layers,
        'num_decoder_layers': wandb.config.num_decoder_layers,
        'max_len': 32,
        'embedding_size_src': 27,  # hvo
        'embedding_size_tgt': 27,  # hvo
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # TRAINING PARAMETERS
    training_parameters = {
        'learning_rate': wandb.config.learning_rate,
        'batch_size':  wandb.config.batch_size,
        'batch_size': wandb.config.batch_size,
        'lr_scheduler_step_size': wandb.config.lr_scheduler_step_size,
        'lr_scheduler_gamma': wandb.config.lr_scheduler_gamma
    }

    # PYTORCH LOSS FUNCTIONS
    BCE_fn = torch.nn.BCEWithLogitsLoss()
    MSE_fn = torch.nn.MSELoss()

    model, optimizer, scheduler, ep = initialize_model(model_parameters, training_parameters, save_info,
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

    wandb.config.update(dataset_parameters)
    wandb.watch(model)
    dataloader = load_dataset(GrooveMidiDatasetSymbolic, subset_info, filters, training_parameters['batch_size'],
                              dataset_parameters)

    evaluator = Evaluator(
        pickle_source_path=subset_info["pickle_source_path"],
        set_subfolder=subset_info["subset"],
        hvo_pickle_filename=subset_info["hvo_pickle_filename"],
        list_of_filter_dicts_for_subsets=[filters],
        max_hvo_shape=(32,27),
        n_samples_to_use=training_parameters['batch_size'],
        n_samples_to_synthesize_visualize_per_subset=10,
        disable_tqdm=False,
        analyze_heatmap=True,
        analyze_global_features=True
    )

    epoch_save_div = 100
    eps = wandb.config.epochs

    try:
        for i in np.arange(eps):
            ep += 1
            print(f"Epoch {ep}\n-------------------------------")
            train_loop(dataloader=dataloader, groove_transformer=model, opt=optimizer, scheduler=scheduler, epoch=ep,
                   loss_fn=calculate_loss, bce_fn=BCE_fn, mse_fn=MSE_fn, save_epoch=epoch_save_div, cp_info=save_info,
                   device=model_parameters['device'])
            print("-------------------------------\n")
    finally:
        wandb.finish()
