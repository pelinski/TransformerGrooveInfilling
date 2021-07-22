import os
import torch
import wandb
from torch.utils.data import DataLoader
import sys

from preprocess_dataset import load_preprocessed_dataset
from evaluator import init_evaluator, log_eval
from utils import get_epoch_log_freq

sys.path.insert(1, "../../BaseGrooveTransformers/")
from models.train import initialize_model, calculate_loss, train_loop

experiment = 'InfillingKicksAndSnares'

settings = {'testing': False,
            'log_to_wandb': True,
            'evaluator_train': True,
            'evaluator_test': True,
            'job_type': 'train'}

# –––––––––––––––––––––––––––––––––––––––––––––––––––

hyperparameter_defaults = dict(
    experiment=experiment,
    encoder_only=1,
    optimizer_algorithm='sgd',
    d_model=64,
    n_heads=16,
    dropout=0.2,
    num_encoder_decoder_layers=7,
    batch_size=16,
    dim_feedforward=256,
    learning_rate=0.05,
    epochs=1 if settings['testing'] else 250,
    h_loss_multiplier=1,
    v_loss_multiplier=1,
    o_loss_multiplier=1
    #    lr_scheduler_step_size=30,
    #    lr_scheduler_gamma=0.1
)

paths = {
    "InfillingKicksAndSnares": {
        'datasets': {
            "train": '../datasets/InfillingKicksAndSnares/0.1.2/train',
            "test": '../datasets/InfillingKicksAndSnares/0.1.2/test'},
        'evaluators': {
            'train': '../evaluators/InfillingKicksAndSnares/0.1.2/InfillingKicksAndSnares_train_0.1.2_evaluator.pickle',
            'test': '../evaluators/InfillingKicksAndSnares/0.1.2/InfillingKicksAndSnares_test_0.1.2_evaluator.pickle'
        }
    },
    "InfillingKicksAndSnares_testing": {
        'datasets': {
            "train": '../datasets/InfillingKicksAndSnares_testing/0.1.2/train',
            "test": '../datasets/InfillingKicksAndSnares_testing/0.1.2/test'},
        'evaluators': {
            'train': '../evaluators/InfillingKicksAndSnares_testing/0.1.2/InfillingKicksAndSnares_testing_train_0.1'
                     '.2_evaluator.pickle',
            'test': '../evaluators/InfillingKicksAndSnares_testing/0.1.2/InfillingKicksAndSnares_testing_test_0.1.2_evaluator.pickle'
        }
    },
    "InfillingRandom_testing": {
        'datasets': {
            "train": '../datasets/InfillingRandom_testing/0.0.0/train',
            "test": '../datasets/InfillingRandom_testing/0.0.0/test'},
        'evaluators': {
            'train': '../evaluators/InfillingRandom_testing/0.0.0/InfillingRandom_testing_train_0.0.0_evaluator.pickle',
            'test': '../evaluators/InfillingRandom_testing/0.0.0/InfillingRandom_testing_test_0.0.0_evaluator.pickle'
        }
    }
}

# –––––––––––––––––––––––––––––––––––––––––––––––––––

if __name__ == '__main__':
    os.environ['WANDB_MODE'] = 'online' if settings['log_to_wandb'] else 'offline'

    wandb.init(config=hyperparameter_defaults,
               project=experiment,
               job_type=settings['job_type'],
               settings=wandb.Settings(start_method="fork"))

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
            #        'lr_scheduler_step_size': 30,
            #        'lr_scheduler_gamma': 0.1
        },
        "cp_paths": {
            'checkpoint_path': '../train_results/' + wandb.config.experiment,
            'checkpoint_save_str': '../train_results/' + wandb.config.experiment +
                                   '/transformer_groove_infilling-epoch-{}'
        },
        "load_model": None,
    }

    # log params to wandb
    wandb.config.update(params["model"])

    # initialize model
    model, optimizer, ep = initialize_model(params)
    wandb.watch(model)

    # load dataset
    dataset_train = load_preprocessed_dataset(paths[wandb.config.experiment]['datasets']['train'],
                                              exp=wandb.config.experiment)
    dataloader_train = DataLoader(dataset_train, batch_size=wandb.config.batch_size, shuffle=True)

    if settings['evaluator_train']:
        evaluator_train = init_evaluator(paths[wandb.config.experiment]['evaluators']['train'])
    if settings['evaluator_test']:
        evaluator_test = init_evaluator(paths[wandb.config.experiment]['evaluators']['test'])

    eps = wandb.config.epochs
    BCE_fn, MSE_fn = torch.nn.BCEWithLogitsLoss(reduction='none'), torch.nn.MSELoss(reduction='none')

    # epoch_save_all, epoch_save_partial = get_epoch_log_freq(eps)
    epoch_save_all, epoch_save_partial = [eps - 1], [99, 199]  # FIXME

    for i in range(eps):
        ep += 1

        print(f"Epoch {ep}\n-------------------------------")
        train_loop(dataloader=dataloader_train,
                   groove_transformer=model,
                   encoder_only=wandb.config.encoder_only,
                   opt=optimizer,
                   epoch=ep,
                   loss_fn=calculate_loss,
                   bce_fn=BCE_fn,
                   mse_fn=MSE_fn,
                   device=params["model"]['device'],
                   test_inputs=evaluator_test.processed_inputs,
                   test_gt=evaluator_test.processed_gt,
                   h_loss_mult=wandb.config.h_loss_multiplier,
                   v_loss_mult=wandb.config.v_loss_multiplier,
                   o_loss_mult=wandb.config.o_loss_multiplier,
                   save=(i in epoch_save_partial or i in epoch_save_all))
        print("-------------------------------\n")

        if i in epoch_save_partial or i in epoch_save_all:
            if settings['evaluator_train']:
                log_eval(evaluator_train, model, log_media=i in epoch_save_all, epoch=ep)
            if settings['evaluator_test']:
                log_eval(evaluator_test, model, log_media=i in epoch_save_all, epoch=ep)

        wandb.log({"epoch": ep})

    wandb.finish()