import wandb
from BaseGrooveTransformers import initialize_model
import json
import os
import torch



def cpu_load_from_wandb(wandb_run_path, model_at_epoch):
    api = wandb.Api()
    run = api.run(wandb_run_path)
    configs = json.loads(run.json_config)
    print(configs)
    params = {"model": {
        "experiment": configs["experiment"]['value'],
        "encoder_only": configs["encoder_only"]['value'],
        "optimizer": configs["optimizer_algorithm"]['value'],
        "d_model": configs["d_model"]['value'],
        "n_heads": configs["n_heads"]['value'],
        "dim_feedforward": configs["dim_feedforward"]['value'],
        "dropout": configs["dropout"]['value'],
        "num_encoder_layers": configs["num_encoder_decoder_layers"]['value'],
        "num_decoder_layers": 0,
        "max_len": 32,
        "embedding_size_src": configs["embedding_size_src"]['value'],
        "embedding_size_tgt": configs["embedding_size_tgt"]['value'],  # hvo
        "device": "cpu",
    }, "training": {
        "learning_rate": configs["learning_rate"]['value'],
        "batch_size": configs["batch_size"]['value'],
        "hit_loss_penalty": configs["hit_loss_penalty"]['value']
    }, "load_model": {
        "location": "wandb",
        "dir": wandb_run_path,
        "file_pattern": "transformer_run_{}_Epoch_{}.Model",
        "epoch": model_at_epoch,
        "run": wandb_run_path.split("/")[-1]
    }}

    model, optimizer, initial_epoch = initialize_model(params)

    return model, optimizer, initial_epoch, params


def cpu_load_all_models(models_wandb_paths_epochs):
    all_models = dict()
    all_params = dict()
    for key, (wandb_run_path, model_at_epoch) in models_wandb_paths_epochs.items():
        model, optimizer, initial_epoch, params = cpu_load_from_wandb(wandb_run_path, model_at_epoch)
        all_models[key] = model
        all_params[key] = params
    return all_models, all_params

models_wandb_paths_epochs = {
    "ClosedHH_HVO": ("mmil_infilling/InfillingClosedHH/runs/3vfpovvv", 70),
    "ClosedHH_MSO": ("mmil_infilling/InfillingClosedHH/runs/y16izsyy", 110),
    "KicksSnares": ("mmil_infilling/InfillingKicksAndSnares/runs/330gw6v0", 60),
    "RandomLow": ("mmil_infilling/InfillingRandom/runs/3fj68xpy", 160),
    "RandomHigh": ("mmil_infilling/InfillingRandom/runs/2y6luo8o", 150)
}


if __name__ == '__main__':
    # usage
    models_wandb_paths_epochs = models_wandb_paths_epochs
    all_models, all_params = cpu_load_all_models(models_wandb_paths_epochs)

    PATH = "trained_models"
    if os.path.exists(PATH) is False:
        os.makedirs(PATH, exist_ok=True)

    for model_name in all_params.keys():
        model = all_models[model_name]
        model_path = os.path.join(PATH, model_name+".pt")
        print(model_path)
        torch.save(model.state_dict(), model_path)
        params_path = os.path.join(PATH, model_name + ".params")
        torch.save(all_params[model_name], params_path)
