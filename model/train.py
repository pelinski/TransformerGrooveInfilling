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
from utils import eval_log_freq

sys.path.insert(1, "../../BaseGrooveTransformers/")
from models.train import initialize_model, calculate_loss, train_loop

parser = argparse.ArgumentParser()
parser.add_argument("--paths", help="paths file", default='configs/paths.yaml')
parser.add_argument("--testing", help="testing mode", default=False)
parser.add_argument("--wandb", help="log to wandb", default=True)
parser.add_argument("--eval_train", help="evaluator train set", default=True)
parser.add_argument("--eval_test", help="evaluator test set", default=False)
parser.add_argument("--eval_validation", help="evaluator validation set", default=True)
parser.add_argument("--only_final_eval", help="only final total evaluation", default=False)  # sweeps
parser.add_argument("--dump_eval", help="dump evaluator file", default=True)
parser.add_argument("--load_model", help="load model parameters", default=None)
parser.add_argument("--notes", help="wandb run notes", default=None)
parser.add_argument("--tags", help="wandb run tags", default=None)



# hyperparameters
parser.add_argument("--config", help="yaml config file. if given, the rest of the arguments are not taken into "
                                     "account", default=None)
parser.add_argument("--experiment", help="experiment id", default=None)
parser.add_argument("--encoder_only", help="transformer encoder only", default=1, type=int)
parser.add_argument("--optimizer_algorithm", help="optimizer_algorithm", default='sgd', type=str)
parser.add_argument("--d_model", help="model dimension", default=64, type=int)
parser.add_argument("--n_heads", help="number of heads for multihead attention", default=16, type=int)
parser.add_argument("--dropout", help="dropout factor", default=0.2, type=float)
parser.add_argument("--num_encoder_decoder_layers", help="number of encoder/decoder layers", default=7, type=int)
parser.add_argument("--hit_loss_penalty", help="non_hit loss multiplier (between 0 and 1)", default=1, type=float)
parser.add_argument("--batch_size", help="batch size", default=16, type=int)
parser.add_argument("--dim_feedforward", help="feed forward layer dimension", default=256, type=int)
parser.add_argument("--learning_rate", help="learning rate", default=0.05, type=float)
parser.add_argument("--epochs", help="number of training epochs", default=100, type=int)

args = parser.parse_args()

# args are loaded all from config file or all from cli
if args.config is not None:
    with open(args.config, 'r') as f:
        hyperparameters = yaml.safe_load(f)
else:
    hyperparameters = dict(
        encoder_only=args.encoder_only,
        optimizer_algorithm=args.optimizer_algorithm,
        d_model=args.d_model,
        n_heads=args.n_heads,
        dropout=args.dropout,
        num_encoder_decoder_layers=args.num_encoder_decoder_layers,
        hit_loss_penalty=args.hit_loss_penalty,
        batch_size=args.batch_size,
        dim_feedforward=args.dim_feedforward,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        load_model=args.load_model)

if args.testing:
    hyperparameters['epochs'] = 1

# config files without experiment specified
if args.experiment is not None:
    hyperparameters['experiment'] = args.experiment

assert 'experiment' in hyperparameters.keys(), 'experiment not specified'

pprint.pprint(hyperparameters)

with open(args.paths, 'r') as f:
    paths = yaml.safe_load(f)

os.environ['WANDB_MODE'] = 'online' if args.wandb else 'offline'

if __name__ == '__main__':
    wandb.init(config=hyperparameters,
               project=hyperparameters['experiment'],
               job_type='train',
               notes=args.notes,
               tags=args.tags,
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
            'embedding_size_src': 16 if wandb.config.experiment != 'InfillingClosedHH_Symbolic' else 27,  # mso
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
        "load_model": wandb.config.load_model
    }

    # log params to wandb
    wandb.config.update(params["model"])

    # initialize model
    model, optimizer, initial_epoch = initialize_model(params)
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
    if args.eval_validation:
        evaluator_validation = init_evaluator(paths[wandb.config.experiment]['evaluators']['validation'], device=params[
            'model']['device'])

    BCE_fn, MSE_fn = torch.nn.BCEWithLogitsLoss(reduction='none'), torch.nn.MSELoss(reduction='none')

    total_epochs = wandb.config.epochs
    epoch_save_all, epoch_save_partial = eval_log_freq(total_epochs=total_epochs, initial_epochs_lim=10,
                                                       initial_step_partial=1,
                                                       initial_step_all=1, secondary_step_partial=10,
                                                       secondary_step_all=20,
                                                       only_final=args.only_final_eval)
    ep = initial_epoch
    for i in range(initial_epoch, total_epochs):

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
                   validation_inputs=evaluator_validation.processed_inputs if args.eval_validation else None,
                   validation_gt=evaluator_validation.processed_gt if args.eval_validation else None,
                   hit_loss_penalty=wandb.config.hit_loss_penalty,
                   save=(ep in epoch_save_partial or ep in epoch_save_all))
        print("-------------------------------\n")

        #if ep in epoch_save_partial or ep in epoch_save_all:
        if args.eval_train:
            # evaluator_train._identifier = 'Train_Set_Epoch_{}'.format(ep)
            evaluator_train._identifier = 'Train_Set'
            log_eval(evaluator_train, model, log_media=ep in epoch_save_all, epoch=ep, dump=args.dump_eval)

        if args.eval_test:
            # evaluator_test._identifier = 'Test_Set_Epoch_{}'.format(ep)
            evaluator_test._identifier = 'Test_Set'
            log_eval(evaluator_test, model, log_media=ep in epoch_save_all, epoch=ep, dump=args.dump_eval)

        if args.eval_validation:
            # evaluator_test._identifier = 'Validation_Set_Epoch_{}'.format(ep)
            evaluator_validation._identifier = 'Validation_Set'
            log_eval(evaluator_validation, model, log_media=ep in epoch_save_all, epoch=ep, dump=args.dump_eval)

        wandb.log({"epoch": ep}, commit=True)

        ep += 1

    wandb.finish()
