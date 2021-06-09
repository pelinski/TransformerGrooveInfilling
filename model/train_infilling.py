import sys
import torch
import wandb
import numpy as np

from dataset import GrooveMidiDataset
from torch.utils.data import DataLoader

sys.path.insert(1, "../../BaseGrooveTransformers/")
sys.path.insert(1, "../../GrooveEvaluator")
sys.path.append('../../preprocessed_dataset/')
sys.path.insert(1, "../../hvo_sequence")

from models.train import initialize_model, calculate_loss, train_loop
from GrooveEvaluator.evaluator import Evaluator
from Subset_Creators.subsetters import GrooveMidiSubsetter
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

# disable wandb for testing
import os
os.environ['WANDB_MODE'] = 'offline'

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

    wandb_run = wandb.init(config=hyperparameter_defaults, project='infilling')

    # PARAMETERS
    save_info = {
        'checkpoint_path': '../train_results/',
        'checkpoint_save_str': '../train_results/transformer_groove_infilling-epoch-{}',
        'df_path': '../train_results/losses_df/'
    }

    # DATASET PARAMETERS
    dataset_parameters = {
        'max_len': 32,
        'mso_parameters': {'sr': 44100, 'n_fft': 1024, 'win_length': 1024, 'hop_length':
            441, 'n_bins_per_octave': 16, 'n_octaves': 9, 'f_min': 40, 'mean_filter_size': 22},
        'voices_parameters': {'voice_idx': [2], 'min_n_voices_to_remove': 1,  # closed hh
                              'max_n_voices_to_remove': 1, 'prob': [1], 'k': None},
        'sf_path': ['../soundfonts/filtered_soundfonts/Standard_Drum_Kit.sf2',
                    '../soundfonts/filtered_soundfonts/HardRockDrums.sf2'],
        'max_n_sf': 2,
        'max_aug_items': 1,
        'dataset_name': None
    }

    subset_info = {
        "pickle_source_path": '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.4'
                              '/Processed_On_09_06_2021_at_12_41_hrs',
        "subset": 'GrooveMIDI_processed_train',
        "metadata_csv_filename": 'metadata.csv',
        "hvo_pickle_filename": 'hvo_sequence_data.obj',
        "filters": {
            "beat_type": ["beat"],
            "time_signature": ["4-4"],
            "master_id": ["drummer9/session1/8"]
        }
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
        'embedding_size_src': 16,  # mso
        'embedding_size_tgt': 27,  # hvo
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # TRAINING PARAMETERS
    training_parameters = {
        'learning_rate': wandb.config.learning_rate,
        'batch_size': wandb.config.batch_size,
        'lr_scheduler_step_size': wandb.config.lr_scheduler_step_size,
        'lr_scheduler_gamma': wandb.config.lr_scheduler_gamma
    }

    # PYTORCH LOSS FUNCTIONS
    BCE_fn = torch.nn.BCEWithLogitsLoss()
    MSE_fn = torch.nn.MSELoss()

    model, optimizer, scheduler, ep = initialize_model(model_parameters, training_parameters, save_info,
                                                       load_from_checkpoint=False)

    wandb.config.update(dataset_parameters)
    wandb.watch(model)

    _, subset_list = GrooveMidiSubsetter(pickle_source_path=subset_info["pickle_source_path"],
                                         subset=subset_info["subset"],
                                         hvo_pickle_filename=subset_info["hvo_pickle_filename"],
                                         list_of_filter_dicts_for_subsets=[subset_info["filters"]]).create_subsets()

    gmd = GrooveMidiDataset(subset=subset_list[0], subset_info=subset_info, **dataset_parameters)
    dataloader = DataLoader(gmd, batch_size=training_parameters['batch_size'], shuffle=True)

    # styles filters for eval subsetter
    styles = ["hiphop", "funk", "reggae", "soul", "latin", "jazz", "pop", "afrobeat", "highlife", "punk", "rock"]

    list_of_filter_dicts_for_subsets = []
    for style in styles:
        list_of_filter_dicts_for_subsets.append(
            {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
        )

    evaluator = Evaluator(
        pickle_source_path=subset_info["pickle_source_path"],
        set_subfolder=subset_info["subset"],
        hvo_pickle_filename=subset_info["hvo_pickle_filename"],
        list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
        max_hvo_shape=(32, 27),
        n_samples_to_use=3,
        n_samples_to_synthesize_visualize_per_subset=3,
        disable_tqdm=False,
        analyze_heatmap=True,
        analyze_global_features=True
    )

    # get gt evaluator
    evaluator_subset = evaluator.get_ground_truth_hvo_sequences()
    (eval_hvo_sequences, eval_processed_inputs, eval_processed_outputs), \
    (eval_hvo_index, eval_voices_reduced, eval_soundfonts) = gmd.preprocess_dataset(evaluator_subset)
    wandb.config.update({"eval_hvo_index": eval_hvo_index,
                         "eval_voices_reduced": eval_voices_reduced,
                         "eval_soundfons": eval_soundfonts})

    epoch_save_div = 100
    eps = wandb.config.epochs

    # GENERATE FREQUENCY LOG ARRAYS
    first_epochs_step = 1
    first_epochs_lim = 10 if eps >= 10 else eps
    epoch_save_partial = np.arange(first_epochs_lim, step=first_epochs_step)
    epoch_save_all = np.arange(first_epochs_lim, step=first_epochs_step)
    if first_epochs_lim != eps:
        remaining_epochs_step_partial, remaining_epochs_step_all = 5, 10
        epoch_save_partial = np.append(epoch_save_partial,
                                       np.arange(start=first_epochs_lim, step=remaining_epochs_step_partial, stop=eps))
        epoch_save_all = np.append(epoch_save_all,
                                   np.arange(start=first_epochs_lim, step=remaining_epochs_step_all, stop=eps))

    try:
        for i in np.arange(eps):
            ep += 1
            save_model = (i in epoch_save_partial or i in epoch_save_all)
            print(f"Epoch {ep}\n-------------------------------")
            train_loop(dataloader=dataloader, groove_transformer=model, opt=optimizer, scheduler=scheduler, epoch=ep,
                       loss_fn=calculate_loss, bce_fn=BCE_fn, mse_fn=MSE_fn, save=save_model, device=model_parameters[
                    'device'])
            print("-------------------------------\n")

            # generate evaluator predictions after each epoch
            eval_pred = model.predict(eval_processed_inputs, use_thres=True, thres=0.5)
            eval_pred_hvo_array = np.concatenate(eval_pred, axis=2)
            # FIXME what gt and pred to compare
            evaluator.add_predictions(eval_pred_hvo_array)

            if i in epoch_save_partial or i in epoch_save_all:
                # Evaluate
                acc_h = evaluator.get_hits_accuracies(drum_mapping=ROLAND_REDUCED_MAPPING)
                mse_v = evaluator.get_velocity_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                mse_o = evaluator.get_micro_timing_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                rhythmic_distances = evaluator.get_rhythmic_distances()

                # Log
                wandb.log(acc_h, commit=False)
                wandb.log(mse_v, commit=False)
                wandb.log(mse_o, commit=False)
                wandb.log(rhythmic_distances, commit=False)

            if i in epoch_save_all:

                # Heatmaps
                heatmaps_global_features = evaluator.get_wandb_logging_media(sf_paths=eval_soundfonts,
                                                                             use_custom_sf=True)
                if len(heatmaps_global_features.keys()) > 0:
                    wandb.log(heatmaps_global_features, commit=False)

            evaluator.dump(path="misc/evaluator_run_{}_Epoch_{}.Eval".format(wandb_run.name, ep))
            wandb.log({"epoch": ep})

    finally:
        wandb.finish()
