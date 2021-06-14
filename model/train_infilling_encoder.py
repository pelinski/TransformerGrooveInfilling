import sys
import torch
import wandb
import numpy as np

from dataset import GrooveMidiDatasetInfilling
from torch.utils.data import DataLoader

sys.path.insert(1, "../../BaseGrooveTransformers/")
sys.path.append('../../preprocessed_dataset/')
sys.path.insert(1, "../../hvo_sequence")

from models.train_encoder import initialize_model, calculate_loss, train_loop
from Subset_Creators.subsetters import GrooveMidiSubsetter
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

from evaluator import InfillingEvaluator
# disable wandb for testing
#import os
#os.environ['WANDB_MODE'] = 'offline'

if __name__ == "__main__":

    hyperparameter_defaults = dict(
        optimizer_algorithm='sgd',
        d_model=128,
        n_heads=8,
        dropout=0.1,
        num_encoder_decoder_layers=1,
        learning_rate=1e-3,
        batch_size=64,
        dim_feedforward=1280,
        epochs=100,
        lr_scheduler_step_size=30,
        lr_scheduler_gamma=0.1
    )

    wandb_run = wandb.init(config=hyperparameter_defaults, project='infilling-encoder')

    params = {
        "model": {
            'optimizer': wandb.config.optimizer_algorithm,
            'd_model': wandb.config.d_model,
            'n_heads': wandb.config.n_heads,
            'dim_feedforward': wandb.config.dim_feedforward,
            'dropout': wandb.config.dropout,
            'num_encoder_layers': wandb.config.num_encoder_decoder_layers,
            'num_decoder_layers': wandb.config.num_encoder_decoder_layers,
            'max_len': 32,
            'embedding_size_src': 16,  # mso
            'embedding_size_tgt': 27,  # hvo
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        "training": {
            'learning_rate': wandb.config.learning_rate,
            'batch_size': wandb.config.batch_size,
            'lr_scheduler_step_size': wandb.config.lr_scheduler_step_size,
            'lr_scheduler_gamma': wandb.config.lr_scheduler_gamma
        },
        "dataset": {
            "pickle_source_path": '../../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.5/Processed_On_14_06_2021_at_14_26_hrs',
            "subset": 'GrooveMIDI_processed_train',
            "metadata_csv_filename": 'metadata.csv',
            "hvo_pickle_filename": 'hvo_sequence_data.obj',
            "filters": {
                "beat_type": ["beat"],
                "time_signature": ["4-4"],
                #     "master_id": ["drummer9/session1/8"]
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
        },
        "evaluator": {"n_samples_to_use": 3,
                      "n_samples_to_synthesize_visualize_per_subset": 3},
        "cp_paths": {
            'checkpoint_path': '../train_results/',
            'checkpoint_save_str': '../train_results/transformer_groove_infilling-epoch-{}'
        },
        "load_model": None,

        # load_model options
        # "load_model": {
        #    "location": "local",
        #    "dir": "./wandb/run-20210609_162149-1tsi1g1n/files/saved_models/",
        #    "file_pattern": "transformer_run_{}_Epoch_{}.Model"
        # }
        # "load_model": {
        #    "location": "wandb",
        #    "dir": "marinaniet0/tap2drum/1tsi1g1n/",
        #    "file_pattern": "saved_models/transformer_run_{}_Epoch_{}.Model",
        #    "epoch": 51,
        #    "run": "1tsi1g1n"
        # }
    }

    BCE_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    MSE_fn = torch.nn.MSELoss(reduction='none')

    model, optimizer, scheduler, ep = initialize_model(params)

    # log all params to wandb
    wandb.config.update(params)
    wandb.watch(model)

    # load gmd class and dataset
    _, subset_list = GrooveMidiSubsetter(pickle_source_path=params["dataset"]["pickle_source_path"],
                                         subset=params["dataset"]["subset"],
                                         hvo_pickle_filename=params["dataset"]["hvo_pickle_filename"],
                                         list_of_filter_dicts_for_subsets=[
                                             params['dataset']['filters']]).create_subsets()

    dataset = GrooveMidiDatasetInfilling(data=subset_list[0], **params['dataset'])
    dataloader = DataLoader(dataset, batch_size=params['training']['batch_size'], shuffle=True)

    # instance evaluator and set gt
    evaluator = InfillingEvaluator(
        pickle_source_path=params["dataset"]["pickle_source_path"],
        set_subfolder=params["dataset"]["subset"],
        hvo_pickle_filename=params["dataset"]["hvo_pickle_filename"],
        max_hvo_shape=(32, 27),
        n_samples_to_use=params["evaluator"]["n_samples_to_use"],
        n_samples_to_synthesize_visualize_per_subset=params["evaluator"][
            "n_samples_to_synthesize_visualize_per_subset"],
        disable_tqdm=False,
        analyze_heatmap=True,
        analyze_global_features=True,
        dataset=dataset,
        model=model,
        n_epochs=wandb.config.epochs)
    evaluator.set_gt()

    # log eval_subset parameters to wandb
    wandb.config.update({"eval_hvo_index": evaluator.eval_hvo_index,
                         "eval_voices_reduced": evaluator.eval_voices_reduced,
                         "eval_soundfons": evaluator.eval_soundfonts})

    eps = wandb.config.epochs

    try:
        for i in np.arange(eps):
            ep += 1
            save_model = (i in evaluator.epoch_save_partial or i in evaluator.epoch_save_all)
            print(f"Epoch {ep}\n-------------------------------")
            train_loop(dataloader=dataloader, groove_transformer=model, opt=optimizer, scheduler=scheduler, epoch=ep,
                       loss_fn=calculate_loss, bce_fn=BCE_fn, mse_fn=MSE_fn, save=save_model, device=params["model"][
                    'device'])
            print("-------------------------------\n")

            # generate evaluator predictions after each epoch
            evaluator.set_pred()
            evaluator.identifier = 'Test_Epoch_{}'.format(ep)

            if i in evaluator.epoch_save_partial or i in evaluator.epoch_save_all:
                # get metrics
                acc_h = evaluator.get_hits_accuracies(drum_mapping=ROLAND_REDUCED_MAPPING)
                mse_v = evaluator.get_velocity_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                mse_o = evaluator.get_micro_timing_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                rhythmic_distances = evaluator.get_rhythmic_distances()

                # log metrics to wandb
                wandb.log(acc_h, commit=False)
                wandb.log(mse_v, commit=False)
                wandb.log(mse_o, commit=False)
                wandb.log(rhythmic_distances, commit=False)

            if i in evaluator.epoch_save_all:
                heatmaps_global_features = evaluator.get_wandb_logging_media(use_sf_dict=True)
                if len(heatmaps_global_features.keys()) > 0:
                    wandb.log(heatmaps_global_features, commit=False)

            evaluator.dump(path="misc/evaluator_run_{}_Epoch_{}.Eval".format(wandb_run.name, ep))
            wandb.log({"epoch": ep})

    finally:
        wandb.finish()
