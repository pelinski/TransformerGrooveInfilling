import os
import torch
import wandb
from torch.utils.data import DataLoader

import sys

sys.path.insert(1, "../../BaseGrooveTransformers/")

from models.train import initialize_model, calculate_loss, train_loop
from utils import get_epoch_log_freq
from preprocess_dataset import load_preprocessed_dataset
from evaluator import init_evaluator, log_eval

# ================================= SETTINGS ==================================================== #
preprocessed_dataset_path_train = '../datasets/InfillingKicksAndSnares_testing/0.1.2/train'
preprocessed_dataset_path_test = '../datasets/InfillingKicksAndSnares_testing/0.1.2/test'
evaluator_train_file = '../evaluators/InfillingKicksAndSnares_testing/InfillingKicksAndSnares_train_0.1.2_evaluator.pickle'
evaluator_test_file = '../evaluators/InfillingKicksAndSnares_testing/InfillingKicksAndSnares_test_0.1.2_evaluator' \
                       '.pickle'

# preprocessed_dataset_path_train = '../preprocessed_infilling_datasets/InfillingKicksAndSnares_testing/0.1.2/train'
# preprocessed_dataset_path_test = '../preprocessed_infilling_datasets/InfillingKicksAndSnares_testing/0.1.2/test'

# TODO select experiment here

settings = {'log_to_wandb': True,
            'evaluator_train':True,
            'evaluator_test': True,
            'job_type': 'train'}
os.environ['WANDB_MODE'] = 'online' if settings['log_to_wandb'] else 'offline'

print(f"-------------------------------\nSettings: {settings}\n-------------------------------")

# ============================================================================================== #

hyperparameter_defaults = dict(
    experiment='InfillingKicksAndSnares',
    encoder_only=1,
    optimizer_algorithm='sgd',
    d_model=64,
    n_heads=16,
    dropout=0.2,
    num_encoder_decoder_layers=7,
    batch_size=16,
    dim_feedforward=256,
    learning_rate=0.05,
    epochs=1,
    h_loss_multiplier=1,
    v_loss_multiplier=1,
    o_loss_multiplier=1
    #    lr_scheduler_step_size=30,
    #    lr_scheduler_gamma=0.1
)
wandb_run = wandb.init(config=hyperparameter_defaults, project=hyperparameter_defaults['experiment'] + '_testing',
                       job_type=settings['job_type'], settings=wandb.Settings(start_method="fork"))

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
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        "h_loss_multiplier": wandb.config.h_loss_multiplier,
        "v_loss_multiplier": wandb.config.v_loss_multiplier,
        "o_loss_multiplier": wandb.config.o_loss_multiplier
    },
    "training": {
        'learning_rate': wandb.config.learning_rate,
        'batch_size': wandb.config.batch_size,
        #        'lr_scheduler_step_size': wandb.config.lr_scheduler_step_size,
        #        'lr_scheduler_gamma': wandb.config.lr_scheduler_gamma
    },
    "cp_paths": {
        'checkpoint_path': '../train_results/' + hyperparameter_defaults['experiment'],
        'checkpoint_save_str': '../train_results/' + hyperparameter_defaults[
            'experiment'] + '/transformer_groove_infilling-epoch-{}'
    },
    "load_model": None,
}
# log params to wandb
wandb.config.update(params["model"])

# initialize model
model, optimizer, ep = initialize_model(params)
wandb.watch(model)

# load dataset
dataset_train = load_preprocessed_dataset(preprocessed_dataset_path_train, exp=wandb.config.experiment)
dataloader_train = DataLoader(dataset_train, batch_size=params['training']['batch_size'], shuffle=True)


if settings['evaluator_train']:
    evaluator_train = init_evaluator(evaluator_train_file)
if settings['evaluator_test']:
    evaluator_test = init_evaluator(evaluator_test_file)

eps = wandb.config.epochs
BCE_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
MSE_fn = torch.nn.MSELoss(reduction='none')

# epoch_save_all, epoch_save_partial = get_epoch_log_freq(eps)
epoch_save_all, epoch_save_partial = [eps - 1], [99, 199]

print('Training...')
for i in range(eps):

    ep += 1

    print(f"Epoch {ep}\n-------------------------------")
    train_loop(dataloader=dataloader_train, groove_transformer=model, encoder_only=params["model"][
        "encoder_only"], opt=optimizer, epoch=ep, loss_fn=calculate_loss, bce_fn=BCE_fn,
               mse_fn=MSE_fn, device=params["model"]['device'],
               test_inputs=evaluator_test.processed_inputs,
               test_gt=evaluator_test.processed_gt,
               h_loss_mult=params["model"]["h_loss_multiplier"],
               v_loss_mult=params["model"]["v_loss_multiplier"],
               o_loss_mult=params["model"]["o_loss_multiplier"],
               save=(i in epoch_save_partial or i in epoch_save_all))
    print("-------------------------------\n")

    if i in epoch_save_partial or i in epoch_save_all:
        if settings['evaluator_train']:
            log_eval(evaluator_train, model, log_media=i in epoch_save_all, epoch=ep)
        if settings['evaluator_test']:
            log_eval(evaluator_test, model, log_media=i in epoch_save_all, epoch=ep)

        # rhythmic_distances = evaluator_train.get_rhythmic_distances()
        # wandb.log(rhythmic_distances, commit=False)

    wandb.log({"epoch": ep})

wandb.finish()
