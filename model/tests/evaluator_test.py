import sys
import torch
import wandb
import numpy as np

sys.path.insert(1, "../../model")
from dataset import GrooveMidiDataset
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
os.environ['WANDB_MODE'] = 'offline'

wandb.init()
params = {
    "model": {
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
        "pickle_source_path": '../../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.4'
                              '/Processed_On_09_06_2021_at_12_41_hrs',
        "subset": 'GrooveMIDI_processed_train',
        "metadata_csv_filename": 'metadata.csv',
        "hvo_pickle_filename": 'hvo_sequence_data.obj',
        "filters": {
            "beat_type": ["beat"],
            "time_signature": ["4-4"],
            "master_id": ["drummer9/session1/8"]
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
    "evaluator": {"n_samples_to_use": 3,
                  "n_samples_to_synthesize_visualize_per_subset": 3},
    "cp_paths": {
        'checkpoint_path': '../train_results/',
        'checkpoint_save_str': '../train_results/transformer_groove_infilling-epoch-{}'
    },
    "load_model": None,
}

# load model
model, optimizer, scheduler, ep = initialize_model(params)

# load gmd class and dataset
_, subset_list = GrooveMidiSubsetter(pickle_source_path=params["dataset"]["pickle_source_path"],
                                     subset=params["dataset"]["subset"],
                                     hvo_pickle_filename=params["dataset"]["hvo_pickle_filename"],
                                     list_of_filter_dicts_for_subsets=[
                                         params['dataset']['filters']]).create_subsets()

gmd = GrooveMidiDataset(data=subset_list[0], **params['dataset'])
dataloader = DataLoader(gmd, batch_size=params['training']['batch_size'], shuffle=True)

# instance evaluator and set gt
evaluator = InfillingEvaluator(pickle_source_path=params["dataset"]["pickle_source_path"],
                               set_subfolder=params["dataset"]["subset"],
                               hvo_pickle_filename=params["dataset"]["hvo_pickle_filename"],
                               max_hvo_shape=(32, 27),
                               n_samples_to_use=params["evaluator"]["n_samples_to_use"],
                               n_samples_to_synthesize_visualize_per_subset=params["evaluator"][
                                   "n_samples_to_synthesize_visualize_per_subset"],
                               disable_tqdm=False,
                               analyze_heatmap=True,
                               analyze_global_features=True,
                               dataset=gmd,
                               model=model,
                               n_epochs=100)

# TEST set_gt() method
pre_gt = evaluator.get_ground_truth_hvo_sequences()  # gt without processing
evaluator.set_gt()
post_gt = evaluator.get_ground_truth_hvo_sequences()  # gt after processing

(gt_eval_processed_inputs, gt_eval_processed_gt), (_, _, eval_hvo_sequences_gt), (
gt_eval_hvo_index, gt_eval_voices_reduced, gt_eval_soundfonts) = evaluator.dataset.preprocess_dataset(
    pre_gt)

eval_hvo_array = np.stack([hvo_seq.hvo for hvo_seq in eval_hvo_sequences_gt])

assert np.all(evaluator._gt_hvos_array == eval_hvo_array)
post_gt_eval_hvo_array = np.stack([hvo_seq.hvo for hvo_seq in post_gt])
assert np.all(evaluator._gt_hvos_array == post_gt_eval_hvo_array)

# train for 1 epoch, updates model
train_loop(dataloader=dataloader, groove_transformer=model, opt=optimizer, scheduler=scheduler, epoch=ep,
           loss_fn=calculate_loss, bce_fn=torch.nn.BCEWithLogitsLoss(), mse_fn=torch.nn.MSELoss(), save=False,
           device=params["model"][
               'device'])

# TEST set_pred() method
evaluator.set_pred()

eval_pred = model.predict(gt_eval_processed_inputs, use_thres=True, thres=0.5)
eval_pred_hvo_array = np.concatenate(eval_pred, axis=2)
eval_pred = np.zeros_like(eval_pred_hvo_array)

for idx in range(eval_pred_hvo_array.shape[0]):  # N
    h_idx, v_idx, o_idx = get_hvo_idx_for_voice(voice_idx=gt_eval_voices_reduced[idx],
                                                n_voices=eval_pred_hvo_array.shape[2] // 3)
    eval_pred[idx, :, h_idx] = eval_pred_hvo_array[idx][:, h_idx]
    eval_pred[idx, :, v_idx] = eval_pred_hvo_array[idx][:, v_idx]
    eval_pred[idx, :, o_idx] = eval_pred_hvo_array[idx][:, o_idx]

assert np.all(evaluator._prediction_hvos_array==eval_pred)

"""
    if i in evaluator.epoch_save_partial or i in evaluator.epoch_save_all:
        # get metrics
        acc_h = evaluator.get_hits_accuracies(drum_mapping=ROLAND_REDUCED_MAPPING)
        mse_v = evaluator.get_velocity_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
        mse_o = evaluator.get_micro_timing_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
        rhythmic_distances = evaluator.get_rhythmic_distances()

    if i in evaluator.epoch_save_all:
        heatmaps_global_features = evaluator.get_wandb_logging_media(sf_paths=evaluator.eval_soundfonts,
                                                                     use_custom_sf=True)
"""
