import sys
import torch
import wandb
import numpy as np

sys.path.insert(1, "../../model")
from dataset import GrooveMidiDatasetInfilling
from torch.utils.data import DataLoader

sys.path.insert(1, "../../../GrooveEvaluator")
sys.path.insert(1, "../../../BaseGrooveTransformers/")
sys.path.append('../../../preprocessed_dataset/')
sys.path.insert(1, "../../../hvo_sequence")

from models.train import initialize_model, calculate_loss, train_loop
from Subset_Creators.subsetters import GrooveMidiSubsetter
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING
from utils import get_hvo_idx_for_voice

from evaluator import InfillingEvaluator
import os

use_wand = True
os.environ['WANDB_MODE'] = 'online' if use_wand else 'offline'

wandb.init()
params = {
    "model": {
        'encoder_only': True,
        'optimizer': 'sgd',
        'd_model': 128,
        'n_heads': 8,
        'dim_feedforward': 1280,
        'dropout': 0.1,
        'num_encoder_layers': 5,
        'num_decoder_layers': 5,
        'max_len': 32,
        'embedding_size_src': 16,  # mso
        'embedding_size_tgt': 27,  # hvo
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    },
    "training": {
        'learning_rate': 1e-3,
        'batch_size': 64,
        'lr_scheduler_step_size': 30,
        'lr_scheduler_gamma': 0.1
    },
    "dataset": {
        "subset_info": {
            "pickle_source_path": '../../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.5/Processed_On_14_06_2021_at_14_26_hrs',
            "subset": 'GrooveMIDI_processed_train',
            "metadata_csv_filename": 'metadata.csv',
            "hvo_pickle_filename": 'hvo_sequence_data.obj',
            "filters": {
                "beat_type": ["beat"],
                "time_signature": ["4-4"],
                # "master_id": ["drummer9/session1/8"]
                "master_id": ["drummer1/session1/201"]
            }
        },
        'max_len': 32,
        'mso_params': {'sr': 44100, 'n_fft': 1024, 'win_length': 1024, 'hop_length':
            441, 'n_bins_per_octave': 16, 'n_octaves': 9, 'f_min': 40, 'mean_filter_size': 22},
        'voices_params': {'voice_idx': [2], 'min_n_voices_to_remove': 1,  # closed hh
                          'max_n_voices_to_remove': 1, 'prob': [1], 'k': None},
        'sf_path': ['../../soundfonts/filtered_soundfonts/Standard_Drum_Kit.sf2'],
        'max_n_sf': 1,
        'max_aug_items': 1,
        'dataset_name': None
    },
    "evaluator": {"n_samples_to_use": 12,
                  "n_samples_to_synthesize_visualize_per_subset": 10},
    "cp_paths": {
        'checkpoint_path': '../train_results/',
        'checkpoint_save_str': '../train_results/transformer_groove_infilling-epoch-{}'
    },
    "load_model": None,
}

# load model
model, optimizer, ep = initialize_model(params)

_preprocess_dataset = False
if _preprocess_dataset:
    _, subset_list = GrooveMidiSubsetter(pickle_source_path=params["dataset"]["subset_info"]["pickle_source_path"],
                                         subset=params["dataset"]["subset_info"]["subset"],
                                         hvo_pickle_filename=params["dataset"]["subset_info"]["hvo_pickle_filename"],
                                         list_of_filter_dicts_for_subsets=[
                                             params['dataset']["subset_info"]['filters']]).create_subsets()

    dataset = GrooveMidiDatasetInfilling(data=subset_list[0], **params['dataset'])

else:
    load_dataset_path = '../dataset/Dataset_16_06_2021_at_11_52_hrs'
    dataset = GrooveMidiDatasetInfilling(load_dataset_path=load_dataset_path)
    params["dataset"] = dataset.get_params()

print(dataset.__len__())
dataloader = DataLoader(dataset, batch_size=params['training']['batch_size'], shuffle=True)

# instance evaluator and set gt
evaluator = InfillingEvaluator(pickle_source_path=params["dataset"]["subset_info"]["pickle_source_path"],
                               set_subfolder=params["dataset"]["subset_info"]["subset"],
                               hvo_pickle_filename=params["dataset"]["subset_info"]["hvo_pickle_filename"],
                               max_hvo_shape=(32, 27),
                               n_samples_to_use=params["evaluator"]["n_samples_to_use"],
                               n_samples_to_synthesize_visualize_per_subset=params["evaluator"][
                                   "n_samples_to_synthesize_visualize_per_subset"],
                               disable_tqdm=False,
                               analyze_heatmap=True,
                               analyze_global_features=True,
                               dataset=dataset,
                               model=model,
                               n_epochs=100)

# TEST set_gt() method
pre_gt = evaluator.get_gmd_ground_truth_hvo_sequences()  # gt without infilling processing

preprocessed_dataset = evaluator.dataset.preprocess_dataset(pre_gt)
gt_eval_processed_inputs = preprocessed_dataset["processed_inputs"]
gt_eval_processed_gt = preprocessed_dataset["hvo_sequences"]
eval_hvo_sequences_inputs = preprocessed_dataset["hvo_sequences_inputs"]
eval_hvo_sequences_gt = preprocessed_dataset["hvo_sequences_outputs"]
gt_eval_hvo_index = preprocessed_dataset["hvo_index"]
gt_eval_voices_reduced = preprocessed_dataset["voices_reduced"]
gt_eval_soundfonts = preprocessed_dataset["soundfonts"]

eval_hvo_array = np.stack([hvo_seq.hvo for hvo_seq in eval_hvo_sequences_gt])

print("set_gt()", np.all(evaluator._gt_hvos_array == eval_hvo_array))

# train for 1 epoch, updates model
train_loop(dataloader=dataloader, groove_transformer=model, encoder_only=params["model"]["encoder_only"],
opt = optimizer, epoch = ep, loss_fn = calculate_loss, bce_fn = torch.nn.BCEWithLogitsLoss(
    reduction='none'), mse_fn = torch.nn.MSELoss(reduction='none'), save = False, device = params["model"]['device'])

# TEST set_pred() method
evaluator.set_pred()

eval_pred = model.predict(gt_eval_processed_inputs, use_thres=True, thres=0.5)
eval_pred_hvo_array = np.concatenate(eval_pred, axis=2)
eval_pred = np.zeros_like(eval_pred_hvo_array)

for idx in range(eval_pred_hvo_array.shape[0]):  # N
    h_idx, v_idx, o_idx = get_hvo_idx_for_voice(voice_idx=gt_eval_voices_reduced[idx],
                                                n_voices=eval_pred_hvo_array.shape[2] // 3)
    eval_pred[idx, :, h_idx] = eval_pred_hvo_array[idx][:, h_idx]
    eval_pred[idx, :, v_idx] = eval_pred_hvo_array[idx][:, h_idx] * eval_pred_hvo_array[idx][:, v_idx]
    eval_pred[idx, :, o_idx] = eval_pred_hvo_array[idx][:, h_idx] * eval_pred_hvo_array[idx][:, o_idx]

print("set_pred()", np.all(evaluator._prediction_hvos_array == eval_pred))

media = evaluator.get_wandb_logging_media()
wandb.log(media)