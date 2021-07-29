import os
import torch
import wandb
from torch.utils.data import DataLoader
import sys
import yaml
import argparse
import pprint

from preprocess_dataset import load_preprocessed_dataset
from evaluator import init_evaluator, log_eval
from utils import get_epoch_log_freq

sys.path.insert(1, "../../BaseGrooveTransformers/")
from models.train import initialize_model, calculate_loss, train_loop

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="yaml config file", default='configs/hyperparameter_defaults.yaml')
parser.add_argument("--paths", help="paths file", default='configs/paths.yaml')
parser.add_argument("--testing", help="testing mode", default=False)
parser.add_argument("--wandb", help="log to wandb", default=True)
parser.add_argument("--eval_train", help="evaluator train set", default=True)
parser.add_argument("--eval_test", help="evaluator test set", default=True)
args = parser.parse_args()

with open(args.config, 'r') as f:
    hyperparameters = yaml.safe_load(f)
if args.testing:
    hyperparameters['epochs'] = 1
pprint.pprint(hyperparameters)

with open(args.paths, 'r') as f:
    paths = yaml.safe_load(f)

if __name__ == '__main__':
    os.environ['WANDB_MODE'] = 'online' if args.wandb else 'offline'

    wandb.init(config=hyperparameters,
               project=hyperparameters['experiment'],
               job_type='train',
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
            'num_decoder_layers': 0 if wandb.config.encoder_only else wandb.config.num_encoder_decoder_layers,
            'max_len': 32,
            'embedding_size_src': 16,  # mso
            'embedding_size_tgt': 27,  # hvo
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        "training": {
            'learning_rate': wandb.config.learning_rate,
            'batch_size': wandb.config.batch_size,
            'hit_loss_penalty': wandb.config.hit_loss_penalty
            #        'lr_scheduler_step_size': 30,
            #        'lr_scheduler_gamma': 0.1
        },
        "load_model": None,
    }

    # log params to wandb
    wandb.config.update(params["model"])

    # initialize model
    model, optimizer, ep = initialize_model(params)
    wandb.watch(model, log_freq=1000)

    # load dataset
    dataset_train = load_preprocessed_dataset(paths[wandb.config.experiment]['datasets']['train'],
                                              exp=wandb.config.experiment)
    dataloader_train = DataLoader(dataset_train, batch_size=wandb.config.batch_size, shuffle=True, pin_memory=True)

    if args.eval_train:
        evaluator_train = init_evaluator(paths[wandb.config.experiment]['evaluators']['train'], device=params[
            'model']['device'])
    if args.eval_test:
        evaluator_test = init_evaluator(paths[wandb.config.experiment]['evaluators']['test'], device=params[
            'model']['device'])

    eps = wandb.config.epochs
    BCE_fn, MSE_fn = torch.nn.BCEWithLogitsLoss(reduction='none'), torch.nn.MSELoss(reduction='none')

    # epoch_save_all, epoch_save_partial = get_epoch_log_freq(eps)
    epoch_save_all, epoch_save_partial = [eps - 1], []  # FIXME

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
                   test_inputs=evaluator_test.processed_inputs if args.eval_test else None,
                   test_gt=evaluator_test.processed_gt if args.eval_test else None,
                   hit_loss_penalty=wandb.config.hit_loss_penalty,
                   save=(i in epoch_save_partial or i in epoch_save_all))
        print("-------------------------------\n")

        if i in epoch_save_partial or i in epoch_save_all:
            if args.eval_train:
                # evaluator_train._identifier = 'Train_Set_Epoch_{}'.format(ep) # FIXME
                evaluator_train._identifier = 'Train_Set'
                log_eval(evaluator_train, model, log_media=i in epoch_save_all, epoch=ep, dump=not args.testing)

            if args.eval_test:
                evaluator_test._identifier = 'Test_Set'
                # evaluator_test._identifier = 'Test_Set_Epoch_{}'.format(ep) # FIXME
                log_eval(evaluator_test, model, log_media=i in epoch_save_all, epoch=ep, dump=not args.testing)

        wandb.log({"epoch": ep}, commit=True)

    wandb.finish()
