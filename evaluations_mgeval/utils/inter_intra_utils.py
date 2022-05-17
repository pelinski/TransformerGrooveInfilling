from evaluations_mgeval.utils.mgeval_rytm_utils import *
import os
import torch
import json
from BaseGrooveTransformers.models.transformer import GrooveTransformerEncoder
from GrooveEvaluator.GrooveEvaluator.evaluator import *
import yaml
import pickle
from copy import deepcopy
'''
evaluators_paths = {
        "InfillingClosedHH_Symbolic":
            "datasets/final_evaluators/InfillingEvaluator_0.3.2/InfillingClosedHH_Symbolic_validation_0.1.1_evaluator.pickle",
        "InfillingClosedHH": "datasets/final_evaluators/InfillingEvaluator_0.3.2/InfillingClosedHH_validation_0.1.2_evaluator.pickle",
        "InfillingKicksAndSnares": "datasets/final_evaluators/InfillingEvaluator_0.3.2/InfillingKicksAndSnares_validation_0.1.2_evaluator.pickle",
        "InfillingRandomLow": "datasets/final_evaluators/InfillingEvaluator_0.3.2/InfillingRandomLow_validation_0.0.0_evaluator.pickle",
        "InfillingRandom": "datasets/final_evaluators/InfillingEvaluator_0.3.2/InfillingRandom_validation_0.0.0_evaluator.pickle",
    }
'''

feature_labels = ['Statistical::NoI', 'Statistical::Total Step Density', 'Statistical::Avg Voice Density', 'Statistical::Lowness',
     'Statistical::Midness', 'Statistical::Hiness', 'Statistical::Vel Similarity Score',
     'Statistical::Weak to Strong Ratio', 'Statistical::Poly Velocity Mean', 'Statistical::Poly Velocity std',
     'Statistical::Poly Offset Mean', 'Statistical::Poly Offset std', 'Syncopation::Combined',
     'Syncopation::Polyphonic', 'Syncopation::Lowsync', 'Syncopation::Midsync', 'Syncopation::Hisync',
     'Syncopation::Lowsyness', 'Syncopation::Midsyness', 'Syncopation::Hisyness', 'Syncopation::Complexity',
     'Auto-Correlation::Skewness', 'Auto-Correlation::Max', 'Auto-Correlation::Centroid',
     'Auto-Correlation::Harmonicity', 'Micro-Timing::Swingness', 'Micro-Timing::Laidbackness',
     'Micro-Timing::Accuracy']

feature_dict_template = {feature_label: np.array([]) for feature_label in feature_labels}
feature_dict_template.update({'metadata': []})


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


def get_loop_ids(hvo_set, indices = None):
    loop_ids = []
    hvo_set_to_check = hvo_set._gt_hvo_sequences if indices is None else [hvo_set._gt_hvo_sequences[i] for i in indices]
    for hvo in hvo_set_to_check:
        loop_ids.append(hvo.metadata.loop_id)
    return loop_ids

def get_master_id_loops_dict(loop_ids):
    master_id_loops_dict = dict()
    for loop_id in loop_ids:
        master_id = loop_id.split(":")[0]
        if master_id not in master_id_loops_dict.keys():
            master_id_loops_dict.update({master_id: [loop_id]})
        else:
            master_id_loops_dict[master_id].append(loop_id)

    loops_in_master_counts_dict = dict()
    for key, set in master_id_loops_dict.items():
        loops_in_master_counts_dict.update({key: len(set)})
    return master_id_loops_dict, loops_in_master_counts_dict

def GetSpacedElements(array, numElems):
    indices = np.round(np.linspace(0, len(array) - 1, numElems)).astype(int).tolist()
    out = [array[i] for i in indices]
    return out

def n_samples_per_master_id(master_id_loops_dict, n_samples):
    sampled_loops = []
    sampled_dict = {master_id: [] for master_id in master_id_loops_dict.keys()}

    for master_id, loop_ids in master_id_loops_dict.items():
        loop_ids = sorted(loop_ids, reverse=True)

        elements = GetSpacedElements(loop_ids, n_samples)

        sampled_dict[master_id].extend(elements)
        sampled_loops.extend(elements)

    return sampled_dict, sampled_loops

def downsample_set(evaluator_gt, num_samples_per_master_id=None, loop_ids = None):

    if num_samples_per_master_id is not None:
        master_id_loops_dict, loops_in_master_counts_dict = get_master_id_loops_dict(get_loop_ids(evaluator_gt))
        sampled_dict, sampled_loop_ids = n_samples_per_master_id(master_id_loops_dict,
                                                                 n_samples=num_samples_per_master_id)
    else:
        assert loop_ids is not None, "Cant have both as None"
        sampled_loop_ids = loop_ids

    gt_hvos_array = []
    gt_hvo_seqs = []

    for hvo_seq in evaluator_gt.hvo_sequences:
        if hvo_seq.metadata.loop_id in sampled_loop_ids:
            gt_hvo_seqs.append(hvo_seq)
            gt_hvos_array.append(hvo_seq.hvo)

    gt_hvos_array = np.array(gt_hvos_array)

    return gt_hvo_seqs, gt_hvos_array, sampled_loop_ids


def convert_hvo_seqs_to_msos(hvo_seqs, sf_path):
    msos_arrays_gt = []
    for ix in tqdm(range(len(hvo_seqs)), f"converting to mso using soundfont {sf_path}"):
        try:
            msos_arrays_gt.append(hvo_seqs[ix].mso(sf_path=sf_path))
        except:
            msos_arrays_gt.append(np.zeros((32, 16)))
            continue
    return np.array(msos_arrays_gt)


def update_statistical_features(__extracted_features_dict, sample_hvo):

    statistical_keys = __extracted_features_dict.keys()

    if "Statistical::NoI" in statistical_keys:
        __extracted_features_dict["Statistical::NoI"] = np.append(
            __extracted_features_dict["Statistical::NoI"],
            sample_hvo.get_number_of_active_voices()
        )

    if "Statistical::Total Step Density" in statistical_keys:
        __extracted_features_dict["Statistical::Total Step Density"] = np.append(
            __extracted_features_dict["Statistical::Total Step Density"],
            sample_hvo.get_total_step_density()
        )

    if "Statistical::Avg Voice Density" in statistical_keys:
        __extracted_features_dict["Statistical::Avg Voice Density"] = np.append(
            __extracted_features_dict["Statistical::Avg Voice Density"],
            sample_hvo.get_average_voice_density()
        )

    if any(x in statistical_keys for x in ["Statistical::Lowness", "Statistical::Midness", "Statistical::Hiness"]):
        lowness, midness, hiness = sample_hvo.get_lowness_midness_hiness()
        if "Statistical::Lowness" in statistical_keys:
            __extracted_features_dict["Statistical::Lowness"] = np.append(
                __extracted_features_dict["Statistical::Lowness"],
                lowness
            )
        if "Statistical::Midness" in statistical_keys:
            __extracted_features_dict["Statistical::Midness"] = np.append(
                __extracted_features_dict["Statistical::Midness"],
                midness
            )
        if "Statistical::Hiness" in statistical_keys:
            __extracted_features_dict["Statistical::Hiness"] = np.append(
                __extracted_features_dict["Statistical::Hiness"],
                hiness
            )

    if "Statistical::Vel Similarity Score" in statistical_keys:
        __extracted_features_dict["Statistical::Vel Similarity Score"] = np.append(
            __extracted_features_dict["Statistical::Vel Similarity Score"],
            sample_hvo.get_velocity_score_symmetry()
        )

    if "Statistical::Weak to Strong Ratio" in statistical_keys:
        __extracted_features_dict["Statistical::Weak to Strong Ratio"] = np.append(
            __extracted_features_dict["Statistical::Weak to Strong Ratio"],
            sample_hvo.get_total_weak_to_strong_ratio()
        )

    if any(x in statistical_keys for x in ["Statistical::Poly Velocity Mean", "Statistical::Poly Velocity std"]):
        mean, std = sample_hvo.get_polyphonic_velocity_mean_stdev()
        if "Statistical::Poly Velocity Mean" in statistical_keys:
            __extracted_features_dict["Statistical::Poly Velocity Mean"] = np.append(
                __extracted_features_dict["Statistical::Poly Velocity Mean"],
                mean
            )
        if "Statistical::Poly Velocity std" in statistical_keys:
            __extracted_features_dict["Statistical::Poly Velocity std"] = np.append(
                __extracted_features_dict["Statistical::Poly Velocity std"],
                std
            )

    if any(x in statistical_keys for x in ["Statistical::Poly Offset Mean", "Statistical::Poly Offset std"]):
        mean, std = sample_hvo.get_polyphonic_offset_mean_stdev()
        if "Statistical::Poly Offset Mean" in statistical_keys:
            __extracted_features_dict["Statistical::Poly Offset Mean"] = np.append(
                __extracted_features_dict["Statistical::Poly Offset Mean"],
                mean
            )
        if "Statistical::Poly Offset std" in statistical_keys:
            __extracted_features_dict["Statistical::Poly Offset std"] = np.append(
                __extracted_features_dict["Statistical::Poly Offset std"],
                std
            )

def update_syncopation_features(__extracted_features_dict, sample_hvo):
    sync_keys = __extracted_features_dict.keys()

    if "Syncopation::Combined" in sync_keys:
        __extracted_features_dict["Syncopation::Combined"] = np.append(
            __extracted_features_dict["Syncopation::Combined"],
            sample_hvo.get_combined_syncopation()
        )

    if "Syncopation::Polyphonic" in sync_keys:
        __extracted_features_dict["Syncopation::Polyphonic"] = np.append(
            __extracted_features_dict["Syncopation::Polyphonic"],
            sample_hvo.get_witek_polyphonic_syncopation()
        )

    if any(shared_feats in sync_keys for shared_feats in ["Syncopation::Lowsync", "Syncopation::Midsync",
                                                          "Syncopation::Hisync","Syncopation::Lowsyness",
                                                          "Syncopation::Midsyness", "Syncopation::Hisyness"]):

        lmh_sync_info = sample_hvo.get_low_mid_hi_syncopation_info()

        for feat in ["Syncopation::Lowsync", "Syncopation::Midsync", "Syncopation::Hisync", "Syncopation::Lowsyness",
                     "Syncopation::Midsyness", "Syncopation::Hisyness"]:
            if feat.split("::")[-1].lower() in lmh_sync_info.keys():
                __extracted_features_dict[feat] = np.append(
                    __extracted_features_dict[feat],
                    lmh_sync_info[feat.split("::")[-1].lower()]
                )

    if "Syncopation::Complexity" in sync_keys:
        __extracted_features_dict["Syncopation::Complexity"] = np.append(
            __extracted_features_dict["Syncopation::Complexity"],
            sample_hvo.get_total_complexity()
        )

def update_autocorrelation_features(__extracted_features_dict, sample_hvo):
    autocorrelation_keys = __extracted_features_dict.keys()

    if any(shared_feats in autocorrelation_keys for shared_feats in [
        "Auto-Correlation::Skewness", "Auto-Correlation::Max",
        "Auto-Correlation::Centroid", "Auto-Correlation::Harmonicity"]
           ):
        autocorrelation_features = sample_hvo.get_velocity_autocorrelation_features()

        for feat in ["Auto-Correlation::Skewness", "Auto-Correlation::Max",
                     "Auto-Correlation::Centroid", "Auto-Correlation::Harmonicity"]:
            __extracted_features_dict[feat] = np.append(
                    __extracted_features_dict[feat],
                    autocorrelation_features[feat.split("::")[-1].lower()]
            )

def update_microtiming_features(__extracted_features_dict, sample_hvo):

    if "Micro-Timing::Swingness" in __extracted_features_dict.keys():
        __extracted_features_dict["Micro-Timing::Swingness"] = np.append(
            __extracted_features_dict["Micro-Timing::Swingness"],
            sample_hvo.swingness()
        )

    if "Micro-Timing::Laidbackness" in __extracted_features_dict.keys():
        __extracted_features_dict["Micro-Timing::Laidbackness"] = np.append(
            __extracted_features_dict["Micro-Timing::Laidbackness"],
            sample_hvo.laidbackness()
        )

    if "Micro-Timing::Accuracy" in __extracted_features_dict.keys():
        __extracted_features_dict["Micro-Timing::Accuracy"] = np.append(
            __extracted_features_dict["Micro-Timing::Accuracy"],
            sample_hvo.get_timing_accuracy()
        )

def update_metadata(__extracted_features_dict, sample_hvo):
    __extracted_features_dict['metadata'].append(sample_hvo.metadata)

def get_feature_dict_from (hvo_seqs_list, set_tag="Unknown", use_voices=None):

    __extracted_features_dict = deepcopy(feature_dict_template)
    for ix in tqdm(range(len(hvo_seqs_list)), f"Extracting Features for {set_tag} set"):
        if use_voices is None:
            hvo_seq_ = hvo_seqs_list[ix]
        else:
            hvo_seq_ = hvo_seqs_list[ix].copy_zero()
            for voice in use_voices:
                hvo_seq_.hvo[:, voice] = hvo_seqs_list[ix].hvo[:, voice]
                hvo_seq_.hvo[:, voice+9] = hvo_seqs_list[ix].hvo[:, voice+9]
                hvo_seq_.hvo[:, voice+18] = hvo_seqs_list[ix].hvo[:, voice+18]

        update_statistical_features(__extracted_features_dict, hvo_seq_)
        update_syncopation_features(__extracted_features_dict, hvo_seq_)
        update_autocorrelation_features(__extracted_features_dict, hvo_seq_)
        update_microtiming_features(__extracted_features_dict, hvo_seq_)
        update_metadata(__extracted_features_dict, hvo_seq_)
    return __extracted_features_dict


def get_version_with_removed_voices(hvo_seqs_list, voice_list, set_tag):
    hvo_seqs_list_removed = []
    for ix in tqdm(range(len(hvo_seqs_list)), f"removing voices {voice_list} from {set_tag} list"):
        _hvo_seq_removed_voices = hvo_seqs_list[ix].copy()
        for voice_idx in voice_list:
            _hvo_seq_removed_voices.hvo[:, voice_idx] = 0
            _hvo_seq_removed_voices.hvo[:, (voice_idx + 9)] = 0
            _hvo_seq_removed_voices.hvo[:, (voice_idx + 18)] = 0

        hvo_seqs_list_removed.append(_hvo_seq_removed_voices)

    for ix in range(len(hvo_seqs_list_removed)):
        _hvo_seq_removed_voices = hvo_seqs_list_removed[ix]
        for voice_idx in voice_list:
            if (_hvo_seq_removed_voices.hvo[:, voice_idx].sum() + _hvo_seq_removed_voices.hvo[:, (
                    voice_idx + 9)].sum() + _hvo_seq_removed_voices.hvo[:, (voice_idx + 18)].sum()) != 0:
                print(_hvo_seq_removed_voices.hvo[:, voice_idx].sum() +
                      _hvo_seq_removed_voices.hvo[:, (voice_idx + 9)].sum() +
                      _hvo_seq_removed_voices.hvo[:, (voice_idx + 18)].sum())

    return hvo_seqs_list_removed


def get_version_with_randomly_removed_voices(hvo_seqs_list, set_tag, thres_range=(0.4,0.6)):
    hvo_seqs_list_removed = []
    for ix in tqdm(range(len(hvo_seqs_list)), f"removing random voices with a prob in range {thres_range} from {set_tag} list"):
        hvo_reset, hvo_reset_comp = hvo_seqs_list[ix].copy().remove_random_events(thres_range=thres_range)
        hvo_seqs_list_removed.append(hvo_reset)

    return hvo_seqs_list_removed

def mix_input_with_prediction(input_hvos_array, predicted_hvos_array):
    input_with_predicted_output = input_hvos_array.copy()

    event_sample_ix, event_time_ix, event_voice_ix = predicted_hvos_array[:, :, :9].nonzero()
    input_with_predicted_output[event_sample_ix, event_time_ix, event_voice_ix] = predicted_hvos_array[
        event_sample_ix, event_time_ix, event_voice_ix]
    input_with_predicted_output[event_sample_ix, event_time_ix, event_voice_ix + 9] = predicted_hvos_array[
        event_sample_ix, event_time_ix, event_voice_ix + 9]
    input_with_predicted_output[event_sample_ix, event_time_ix, event_voice_ix + 18] = predicted_hvos_array[
        event_sample_ix, event_time_ix, event_voice_ix + 18]

    return input_with_predicted_output