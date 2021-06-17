import os
import torch
import wandb
from torch.utils.data import DataLoader

import sys

sys.path.insert(1, "../../BaseGrooveTransformers/")
sys.path.insert(1, "../../hvo_sequence")

from evaluator import InfillingEvaluator
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING
from models.train import initialize_model, calculate_loss, train_loop
from utils import get_epoch_log_freq
from preprocess_infilling_dataset import preprocess_dataset, load_preprocessed_dataset

# ================================= SETTINGS ==================================================== #
preprocessed_dataset_path = '../preprocessed_infilling_datasets/train/0.0.1/Dataset_17_06_2021_at_17_20_hrs'  # train ds
#preprocessed_dataset_path = '../dataset/Dataset_17_06_2021_at_18_13_hrs' # test symbolic
symbolic = False
use_wandb = True
use_evaluator = True
encoder_only = True
load_dataset = True if preprocessed_dataset_path else False

# wandb
os.environ['WANDB_MODE'] = 'online' if use_wandb else 'offline'
project_name = 'infilling-encoder' if encoder_only else 'infilling'

# ============================================================================================== #


hyperparameter_defaults = dict(
    optimizer_algorithm='sgd',
    d_model=32,
    n_heads=1,
    dropout=0,
    num_encoder_decoder_layers=1,
    learning_rate=1e-3,
    batch_size=64,
    dim_feedforward=32,
    epochs=4,
    #    lr_scheduler_step_size=30,
    #    lr_scheduler_gamma=0.1
)

wandb_run = wandb.init(config=hyperparameter_defaults, project=project_name)

params = {
    "model": {
        "encoder_only": encoder_only,
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
        #        'lr_scheduler_step_size': wandb.config.lr_scheduler_step_size,
        #        'lr_scheduler_gamma': wandb.config.lr_scheduler_gamma
    },
    "evaluator": {"n_samples_to_use": 2048,
                  "n_samples_to_synthesize_visualize_per_subset": 10},
    "cp_paths": {
        'checkpoint_path': '../train_results/',
        'checkpoint_save_str': '../train_results/transformer_groove_infilling-epoch-{}'
    },
    "load_model": None,
}

BCE_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
MSE_fn = torch.nn.MSELoss(reduction='none')

model, optimizer, ep = initialize_model(params)

wandb.watch(model)

# load dataset
if load_dataset:
    dataset = load_preprocessed_dataset(preprocessed_dataset_path, symbolic=symbolic)

else:  # small subset
    params["dataset"] = {
        "subset_info": {
            "pickle_source_path": '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.5/Processed_On_14_06_2021_at_14_26_hrs',
            "subset": 'GrooveMIDI_processed_train',
            "metadata_csv_filename": 'metadata.csv',
            "hvo_pickle_filename": 'hvo_sequence_data.obj',
            "filters": {
                "beat_type": ["beat"],
                "time_signature": ["4-4"],
                "master_id": ["drummer9/session1/8"]
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
    dataset = preprocess_dataset(params)

dataloader = DataLoader(dataset, batch_size=params['training']['batch_size'], shuffle=True)

# log all params to wandb
wandb.config.update(params)

# instance evaluator and set gt
if use_evaluator:
    evaluator = InfillingEvaluator(
        pickle_source_path=dataset.subset_info["pickle_source_path"],
        set_subfolder=dataset.subset_info["subset"],
        hvo_pickle_filename=dataset.subset_info["hvo_pickle_filename"],
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

    # log eval_subset parameters to wandb
    wandb.config.update({"eval_hvo_index": evaluator.hvo_index,
                         "eval_voices_reduced": evaluator.voices_reduced,
                         "eval_soundfons": evaluator.soundfonts})

eps = wandb.config.epochs

try:
    # epoch_save_all, epoch_save_partial = get_epoch_log_freq(eps)
    epoch_save_all, epoch_save_partial = [wandb.config.epochs - 1], []  # last epoch idx

    for i in range(eps):
        ep += 1
        save_model = (i in epoch_save_partial or i in epoch_save_all)
        print(f"Epoch {ep}\n-------------------------------")
        train_loop(dataloader=dataloader, groove_transformer=model, encoder_only=params["model"][
            "encoder_only"], opt=optimizer, epoch=ep, loss_fn=calculate_loss, bce_fn=BCE_fn,
                   mse_fn=MSE_fn, save=save_model, device=params["model"]['device'])
        print("-------------------------------\n")
        if use_evaluator:
            print("use_eval_i", i)
            if i in epoch_save_partial or i in epoch_save_all:
                evaluator.set_pred()
                evaluator.identifier = 'Test_Epoch_{}'.format(ep)

                # get metrics
                acc_h = evaluator.get_hits_accuracies(drum_mapping=ROLAND_REDUCED_MAPPING)
                mse_v = evaluator.get_velocity_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                mse_o = evaluator.get_micro_timing_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                # rhythmic_distances = evaluator.get_rhythmic_distances()

                # log metrics to wandb
                wandb.log(acc_h, commit=False)
                wandb.log(mse_v, commit=False)
                wandb.log(mse_o, commit=False)
                # wandb.log(rhythmic_distances, commit=False)

                evaluator.dump(path="misc/evaluator_run_{}_Epoch_{}.Eval".format(wandb_run.name, ep))

            if i in epoch_save_all:
                heatmaps_global_features = evaluator.get_wandb_logging_media()
                if len(heatmaps_global_features.keys()) > 0:
                    wandb.log(heatmaps_global_features, commit=False)

        wandb.log({"epoch": ep})

finally:
    wandb.finish()
