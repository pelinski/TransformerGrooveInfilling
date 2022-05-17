from evaluations_mgeval.utils import load_models_from_wandb as model_loader
import os
import torch
import json

from process_dataset import load_processed_dataset
from evaluator import init_evaluator, InfillingEvaluator
from BaseGrooveTransformers.models.transformer import GrooveTransformerEncoder

import yaml
import pickle


def initialize_model(params):
    model_params = params

    groove_transformer = GrooveTransformerEncoder(model_params['d_model'], model_params['embedding_size_src'],
                                                  model_params['embedding_size_tgt'], model_params['n_heads'],
                                                  model_params['dim_feedforward'], model_params['dropout'],
                                                  model_params['num_encoder_layers'],
                                                  model_params['max_len'], model_params['device'])
    return groove_transformer

def load_model(model_path, model_name):

    # load checkpoint
    params = torch.load(os.path.join(model_path, model_name+".params"), map_location='cpu')['model']
    print(params)
    model = initialize_model(params)
    model.load_state_dict(torch.load(os.path.join(model_path,model_name + ".pt")))
    model.eval()

    return model

####################################################
# ##########        Note        ####################
#   First make sure the evaluator templates are
#   ready by running process_evaluator.py
#
##########################

if __name__ == "__main__":

    download_models_from_wandb = False
    if download_models_from_wandb is True:
        # usage
        models_wandb_paths_epochs = {
            "InfillingClosedHH_Symbolic": ("mmil_infilling/InfillingClosedHH/runs/3vfpovvv", 70),
            "InfillingClosedHH": ("mmil_infilling/InfillingClosedHH/runs/y16izsyy", 110),
            "InfillingKicksAndSnares": ("mmil_infilling/InfillingKicksAndSnares/runs/330gw6v0", 60),
            "InfillingRandomLow": ("mmil_infilling/InfillingRandom/runs/3fj68xpy", 160),
            "InfillingRandom": ("mmil_infilling/InfillingRandom/runs/2y6luo8o", 150)
        }

        all_models, all_params = model_loader.cpu_load_all_models(models_wandb_paths_epochs)

        PATH = "evaluations_mgeval/trained_models"
        if os.path.exists(PATH) is False:
            os.makedirs(PATH, exist_ok=True)

        for model_name in all_params.keys():
            model = all_models[model_name]
            model_path = os.path.join(PATH, model_name + ".pt")
            print(model_path)
            torch.save(model.state_dict(), model_path)
            params_path = os.path.join(PATH, model_name + ".params")
            torch.save(all_params[model_name], params_path)

    with open("configs/paths.yaml", "r") as f:
        paths = yaml.safe_load(f)

    with open("datasets/preprocessed_evaluators/preprocessed_evaluators_parameters.json") as f:
        params = json.load(f)

    evaluator_sourcepath_filename = {
        "InfillingClosedHH_Symbolic": ("datasets/InfillingClosedHH_Symbolic/0.1.1", "InfillingClosedHH_Symbolic_validation_0.1.1_dataset.pickle"),
        "InfillingClosedHH": ("datasets/InfillingClosedHH/0.1.2/", "InfillingClosedHH_validation_0.1.2_dataset.pickle"),
        "InfillingKicksAndSnares": ("datasets/InfillingKicksAndSnares/0.1.2/", "InfillingKicksAndSnares_validation_0.1.2_dataset.pickle"),
        "InfillingRandomLow": ("datasets/InfillingRandomLow/0.0.0/", "InfillingRandomLow_validation_0.0.0_dataset.pickle"),
        "InfillingRandom": ("datasets/InfillingRandom/0.0.0/", "InfillingRandom_validation_0.0.0_dataset.pickle")
    }

    for experiment in evaluator_sourcepath_filename.keys():
        model_path = "trained_models"
        model_name = experiment
        # params = torch.load(os.path.join(model_path, model_name + ".params"), map_location='cpu')['model']
        model = load_model(model_path, model_name)
        evaluator = init_evaluator(
            paths[experiment]["evaluators"]["validation"],
            device='cpu')

        h, v, o = model.predict(evaluator.processed_inputs)
        predictions = torch.cat((h, v, o), axis=2).detach().numpy()
        evaluator.add_predictions(predictions)
        evaluator.save_as_pickle('datasets/final_evaluators/')
