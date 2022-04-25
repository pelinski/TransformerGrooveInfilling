import yaml
import pickle

from process_dataset import load_processed_dataset
from evaluator import init_evaluator
from BaseGrooveTransformers import initialize_model

experiment = "InfillingClosedHH_testing"

# first you need to download the dataset and preprocessed evaluators and store them in the main directory of the project. the dataset folder can be downloaded here: https://www.dropbox.com/sh/04hae4qnrw2yzjd/AACdf-6kyIGMxDBQ61RHdpQfa?dl=0


# the dataset paths are stored in the file configs/paths.yaml
# import the file config/paths.yaml into the variable paths
with open("configs/paths.yaml", "r") as f:
    paths = yaml.safe_load(f)

# load the train split of the dataset
dataset_train = load_processed_dataset(
    paths[experiment]["datasets"]["train"], exp=experiment
)

# there are evaluation subsets stored in the folder datasets/preprocessed_evaluators. processing the original GrooveMidiDataset is computationally expensive since the MSOs need to be generated. we avoid processing evaluation subset every time we want to run an evaluation by preprocessing the evaluation subset in advance. this also allows comparing  the same examples across different runs of the same experiment.

evaluator_train = init_evaluator(
    paths[experiment]["evaluators"]["train"],
    device='cpu')

# we can also load an evaluator saved after a model has been trained. evaluators are saved as zip files after training in the folder evaluator/. you can unzip the file and obtain a .Eval file. you can load it using pickle

with open("demo/evaluator_Train_Set_run_stilted-gorge-16_Epoch_0.Eval", "rb") as f:
    presaved_evaluator = pickle.load(f)
    
# you can also load models from wandb or stored locally. you can pass the directory where the model epochs are stored to the function initialize_model. this function will find the last stored epoch from the module and load it
# when you load the models you need to pass them the same parameters that the model was created with
# for example, we can load a model created with the params in configs/InfillingClosedHH_training.yaml
# models are locally stored under wandb/run_id/files/

params = {
    "model": {
        "experiment": experiment,
        "encoder_only": 1,
        "optimizer":"sgd",
        "d_model": 32,
        "n_heads": 16,
        "dim_feedforward": 16,
        "dropout": 0.18,
        "num_encoder_layers": 6,
        "num_decoder_layers": 0,
        "max_len": 32,
        "embedding_size_src": 16, # mso
        "embedding_size_tgt": 27,  # hvo
        "device": "cpu"
    },
    "training": {
        "learning_rate": 0.094,
        "batch_size": 32,
        "hit_loss_penalty": 0.47
    },
    "load_model": {
           "location": "local",
           "dir": "demo/",
           "file_pattern": "transformer_run_{}_Epoch_{}.Model"
        }
}
model, optimizer, initial_epoch = initialize_model(params)

# you can also run a model stored in wandb. we can load a model from the experiment "InfillingClosedHH" by loading first its parameters

with open("configs/InfillingClosedHH_training.yaml", "r") as f:
   hyperparameters= yaml.safe_load(f)

params = {
    "model": {
        "experiment":hyperparameters["experiment"],
        "encoder_only":hyperparameters["encoder_only"],
        "optimizer":hyperparameters["optimizer_algorithm"],
        "d_model":hyperparameters["d_model"],
        "n_heads":hyperparameters["n_heads"],
        "dim_feedforward":hyperparameters["dim_feedforward"],
        "dropout":hyperparameters["dropout"],
        "num_encoder_layers":hyperparameters["num_encoder_decoder_layers"],
        "num_decoder_layers": 0,
        "max_len": 32,
        "embedding_size_src": 16,
        "embedding_size_tgt": 27,  # hvo
        "device": "cuda",
    },
    "training": {
        "learning_rate":hyperparameters["learning_rate"],
        "batch_size":hyperparameters["batch_size"],
        "hit_loss_penalty":hyperparameters["hit_loss_penalty"],
    }
}

params["load_model"] = {       
   "location": "wandb",
           "dir": "mmil_infilling/InfillingClosedHH/y16izsyy",
           "file_pattern": "transformer_run_{}_Epoch_{}.Model",
           "epoch": 50,
           "run": "y16izsyy"
}
model, optimizer, initial_epoch = initialize_model(params)
# you won't be able to load models that have been stored from a cuda gpu if you don't have a cuda gpu.


# there are many options to pass to the script train.py, you can check them with python train.py --help