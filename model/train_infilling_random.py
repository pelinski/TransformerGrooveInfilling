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
from preprocess_infilling_dataset import load_preprocessed_dataset

# ================================= SETTINGS ==================================================== #
preprocessed_dataset_path_train = '../preprocessed_infilling_datasets/InfillingRandom/0.0.0/train'
preprocessed_dataset_path_test = '../preprocessed_infilling_datasets/InfillingRandom/0.0.0/test'
#preprocessed_dataset_path_train = '../preprocessed_infilling_datasets/InfillingKicksAndSnares/0.1.2/train'
#preprocessed_dataset_path_test = '../preprocessed_infilling_datasets/InfillingKicksAndSnares/0.1.2/test'

settings = {'log_to_wandb': True,
            'evaluator_test': True,
            'job_type': 'train'}
os.environ['WANDB_MODE'] = 'online' if settings['log_to_wandb'] else 'offline'

print(f"-------------------------------\nSettings: {settings}\n-------------------------------")

# ============================================================================================== #

hyperparameter_defaults = dict(
    experiment='InfillingRandom',
    encoder_only=1,
    optimizer_algorithm='sgd',
    d_model=64,
    n_heads=16,
    dropout=0.2,
    num_encoder_decoder_layers=7,
    batch_size=16,
    dim_feedforward=256,
    learning_rate=1e-3,
    epochs=1,
    use_evaluator=1,
    #    lr_scheduler_step_size=30,
    #    lr_scheduler_gamma=0.1
)
#project_name = 'infilling-encoder' + hyperparameter_defaults if hyperparameter_defaults['encoder_only'] else
# 'infilling'
wandb_run = wandb.init(config=hyperparameter_defaults, project=hyperparameter_defaults['experiment'], job_type=settings[
    'job_type'])

params = {
    "model": {
        'experiment': wandb.config.experiment,
        "encoder_only": wandb.config.encoder_only,
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
    "evaluator": {"n_samples_to_use": 2048,  # 2048
                  "n_samples_to_synthesize_visualize_per_subset": 10},  # 10
    "cp_paths": {
        'checkpoint_path': '../train_results/'+hyperparameter_defaults['experiment'],
        'checkpoint_save_str': '../train_results/'+hyperparameter_defaults[
            'experiment']+'/transformer_groove_infilling-epoch-{}'
    },
    "load_model": None,
}
# log params to wandb
wandb.config.update({**params["model"], 'evaluator': params['evaluator']})

# initialize model
model, optimizer, ep = initialize_model(params)
wandb.watch(model)

# load dataset
dataset_train = load_preprocessed_dataset(preprocessed_dataset_path_train, exp=wandb.config.experiment)
dataloader_train = DataLoader(dataset_train, batch_size=params['training']['batch_size'], shuffle=True)

# instance evaluator and set gt
if wandb.config.use_evaluator:

    pred_horizontal = False if wandb.config.experiment == 'InfillingRandom' else True

    evaluator_train = InfillingEvaluator(
        pickle_source_path=dataset_train.subset_info["pickle_source_path"],
        set_subfolder=dataset_train.subset_info["subset"],
        hvo_pickle_filename=dataset_train.subset_info["hvo_pickle_filename"],
        max_hvo_shape=(32, 27),
        n_samples_to_use=params["evaluator"]["n_samples_to_use"],
        n_samples_to_synthesize_visualize_per_subset=params["evaluator"][
            "n_samples_to_synthesize_visualize_per_subset"],
        _identifier='Train_Set',
        disable_tqdm=False,
        analyze_heatmap=True,
        analyze_global_features=True,
        dataset=dataset_train,
        model=model,
        n_epochs=wandb.config.epochs)

    # log eval_subset parameters to wandb
    wandb.config.update({"train_hvo_index": evaluator_train.hvo_index,
                         "train_soundfons": evaluator_train.soundfonts})
    if pred_horizontal:
        wandb.config.update({"train_voices_reduced": evaluator_train.voices_reduced})

    if settings['evaluator_test']:
        dataset_test = load_preprocessed_dataset(preprocessed_dataset_path_test, exp=wandb.config.experiment)

        evaluator_test = InfillingEvaluator(
            pickle_source_path=dataset_test.subset_info["pickle_source_path"],
            set_subfolder=dataset_test.subset_info["subset"],
            hvo_pickle_filename=dataset_test.subset_info["hvo_pickle_filename"],
            max_hvo_shape=(32, 27),
            n_samples_to_use=params["evaluator"]["n_samples_to_use"],
            n_samples_to_synthesize_visualize_per_subset=params["evaluator"][
                "n_samples_to_synthesize_visualize_per_subset"],
            _identifier='Test_Set',
            disable_tqdm=False,
            analyze_heatmap=True,
            analyze_global_features=True,
            dataset=dataset_test,
            model=model,
            n_epochs=wandb.config.epochs)

        # log eval_subset parameters to wandb
        wandb.config.update({"test_hvo_index": evaluator_test.hvo_index,
                             "test_soundfons": evaluator_test.soundfonts})
        if pred_horizontal:
            wandb.config.update({"train_voices_reduced": evaluator_test.voices_reduced})

eps = wandb.config.epochs
BCE_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
MSE_fn = torch.nn.MSELoss(reduction='none')

# epoch_save_all, epoch_save_partial = get_epoch_log_freq(eps)
epoch_save_all, epoch_save_partial = [eps - 1], []
print('Training...')
for i in range(eps):
    ep += 1
    save_model = (i in epoch_save_partial or i in epoch_save_all)
    print(f"Epoch {ep}\n-------------------------------")
    train_loop(dataloader=dataloader_train, groove_transformer=model, encoder_only=params["model"][
        "encoder_only"], opt=optimizer, epoch=ep, loss_fn=calculate_loss, bce_fn=BCE_fn,
               mse_fn=MSE_fn, save=save_model, device=params["model"]['device'])
    print("-------------------------------\n")
    if wandb.config.use_evaluator:
        if i in epoch_save_partial or i in epoch_save_all:
            # Train set evaluator
            evaluator_train._identifier = 'Train_Epoch_{}'.format(ep)
            evaluator_train.set_pred(horizontal=pred_horizontal)
            train_acc_h = evaluator_train.get_hits_accuracies(drum_mapping=ROLAND_REDUCED_MAPPING)
            train_mse_v = evaluator_train.get_velocity_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
            train_mse_o = evaluator_train.get_micro_timing_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
            wandb.log({**train_acc_h, **train_mse_v, **train_mse_v}, commit=False)

            if i in epoch_save_all:
                heatmaps_global_features_train = evaluator_train.get_wandb_logging_media(global_features_html=False)
                if len(heatmaps_global_features_train.keys()) > 0:
                    wandb.log(heatmaps_global_features_train, commit=False)

            # move torch tensors to cpu before saving so that they can be loaded in cpu machines
            evaluator_train.processed_inputs.to(device='cpu')
            evaluator_train.processed_gt.to(device='cpu')

            evaluator_train.dump(path="evaluator/evaluator_train_run_{}_Epoch_{}.Eval".format(wandb_run.name, ep))

            if settings['evaluator_test']:
                # Test set evaluator
                evaluator_test._identifier = 'Test_Epoch_{}'.format(ep)
                evaluator_test.set_pred(horizontal=pred_horizontal) # TODO if horizontal
                test_acc_h = evaluator_test.get_hits_accuracies(drum_mapping=ROLAND_REDUCED_MAPPING)
                test_mse_v = evaluator_test.get_velocity_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                test_mse_o = evaluator_test.get_micro_timing_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                wandb.log({**test_acc_h, **test_mse_v, **test_mse_v}, commit=False)

                if i in epoch_save_all:
                    heatmaps_global_features_test = evaluator_test.get_wandb_logging_media(global_features_html=False)
                    if len(heatmaps_global_features_test.keys()) > 0:
                        wandb.log(heatmaps_global_features_test, commit=False)

                # move torch tensors to cpu before saving so that they can be loaded in cpu machines
                evaluator_test.processed_inputs.to(device='cpu')
                evaluator_test.processed_gt.to(device='cpu')
                evaluator_test.dump(path="evaluator/evaluator_test_run_{}_Epoch_{}.Eval".format(wandb_run.name, ep))

            # rhythmic_distances = evaluator_train.get_rhythmic_distances()
            # wandb.log(rhythmic_distances, commit=False)

    wandb.log({"epoch": ep})

wandb.finish()
