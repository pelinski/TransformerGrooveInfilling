import numpy as np
import itertools
import random
import json
import pickle
import os
from copy import deepcopy

# hvo preprocess methods


def pad_to_match_max_seq_len(hvo_seq, max_len):
    pad_count = max(max_len - hvo_seq.hvo.shape[0], 0)
    hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, pad_count), (0, 0)), "constant")
    hvo_seq.hvo = hvo_seq.hvo[:max_len, :]  # in case seq exceeds max len

    return hvo_seq


def get_sf_list(sf_path):
    if not isinstance(sf_path, list) and sf_path.endswith(
        ".sf2"
    ):  # if only one sf is given
        sfs_list = [sf_path]
    elif not isinstance(sf_path, list) and os.path.isdir(
        sf_path
    ):  # if dir with sfs is given
        sfs_list = [
            os.path.join(sf_path) + sf
            for sf in os.listdir(sf_path)
            if sf.endswith(".sf2")
        ]
    else:
        sfs_list = sf_path  # list of paths
    return sfs_list


def get_hvo_idxs_for_voice(voice_idx, n_voices):
    """
    Gets index for hits, velocity and offsets for a voice. Used for copying hvo values from a voice from an
    hvo_sequence to another one.
    """
    h_idx = voice_idx
    v_idx = [_ + n_voices for _ in voice_idx]
    o_idx = [_ + 2 * n_voices for _ in voice_idx]

    return h_idx, v_idx, o_idx


# voice combinations methods


def get_voice_idx_for_item(hvo_seq, voices_params):
    """
    Removes the voices in voice_idx that are not present in the hvo_seq. Returns updated dict of voice props for item
    """
    active_voices = hvo_seq.get_active_voices()
    _voice_idx = deepcopy(voices_params["voice_idx"])
    non_present_voices_idx = np.argwhere(~np.isin(_voice_idx, active_voices)).flatten()
    _voice_idx = np.delete(_voice_idx, non_present_voices_idx).tolist()

    _voices_params = deepcopy(voices_params)
    _voices_params["voice_idx"] = list(_voice_idx)
    _voices_params["prob"] = _voices_params["prob"][: len(_voice_idx)]

    return _voice_idx, _voices_params


def get_voice_combinations(**kwargs):
    """
    Gets k possible combinations of voices from a list of voice indexes. If k is None, it will return all possible
    combinations. The combinations are of a minimum size min_n_voices_to_remove and a max size
    max_n_voices_to_remove. When choosing a k number a combinations from all possible combinations, the probability
    of choosing a combination of a number of voices above another can be passed with the prob list, where for a range
    of voices to remove from 1 to 3, [1, 1, 1] indicates equal probability, [1,1,2] indicates that combinations with
    3 voices have double probability of getting chosen, etc.
    @param kwargs:  see below
    @return voice_idx_comb: combinations of voice indexes
    """

    # list of voices to remove
    voice_idx = kwargs.get("voice_idx", [0, 1, 2, 3, 4])
    min_n_voices_to_remove = kwargs.get(
        "min_n_voices_to_remove", 1
    )  # min size of the combination
    max_n_voices_to_remove = kwargs.get(
        "max_n_voices_to_remove", 3
    )  # max size of the combination
    # prob of each n_voices_to_remove set in ascending order
    prob = kwargs.get("prob", [1, 1, 1])
    k = kwargs.get("k", 5)  # max number of combinations to return

    if len(voice_idx) < max_n_voices_to_remove:
        max_n_voices_to_remove = len(voice_idx)

    range_items = range(min_n_voices_to_remove, max_n_voices_to_remove + 1)

    assert len(prob) == len(
        range_items
    ), "The prob list must be the same length as the range(min_n_voices_to_remove, max_n_voices_to_remove)"

    voice_idx_comb = []
    weights = []

    for i, n_voices_to_remove in enumerate(range_items):
        _voice_idx_comb = list(itertools.combinations(voice_idx, n_voices_to_remove))
        voice_idx_comb.extend(_voice_idx_comb)

        _weights = list(np.repeat(prob[i], len(_voice_idx_comb)))
        weights.extend(_weights)

    if k is not None:  # if there is no k, return all possible combinations
        voice_idx_comb = random.choices(voice_idx_comb, weights=weights, k=k)

    return list(voice_idx_comb)


def get_sf_v_combinations(voices_parameters, max_aug_items, max_n_sf, sfs_list):
    """
    Gets soundfont and voices-to-remove combinations according to the parameters specified:
    @param voices_parameters:       Refer to get_voices_combinations docs
    @param max_aug_items:           Maximum number of inputs to obtain from each example in dataset. Each input is
                                    the mso of a synthesized hvo_seq with a particular soundfont and a combination
                                    of removed voices
    @param max_n_sf:                Maximum number of soundfonts from where to choose from for each voice combination
    @param sfs_list:                List of available soundfonts paths
    @return sf_v_comb:              Combinations of soundfonts and voices indexes.
    """

    # k, weighted, voice_combinations
    if len(voices_parameters["voice_idx"]) == 1:
        v_comb = voices_parameters["voice_idx"]
    else:
        v_comb = get_voice_combinations(**voices_parameters)

    # max_n_sf soundfonts to sample from
    if max_n_sf is not None:
        sfs = random.choices(sfs_list, k=max_n_sf)
    else:
        sfs = sfs_list

    # all possible v_comb and sfs combination
    sf_v_comb = list(itertools.product(sfs, v_comb))

    # if there's more combinations than max_aug_items, choose randomly
    if len(sf_v_comb) > max_aug_items:
        sf_v_comb = random.choices(sf_v_comb, k=max_aug_items)

    return sf_v_comb


# general


def add_metadata_to_hvo_seq(hvo_seq, hvo_idx, metadata):
    hvo_seq.drummer = metadata.loc[hvo_idx].at["drummer"]
    hvo_seq.session = metadata.loc[hvo_idx].at["session"]
    hvo_seq.master_id = metadata.loc[hvo_idx].at["master_id"]
    hvo_seq.style_primary = metadata.loc[hvo_idx].at["style_primary"]
    hvo_seq.style_secondary = metadata.loc[hvo_idx].at["style_secondary"]
    hvo_seq.beat_type = metadata.loc[hvo_idx].at["beat_type"]
    hvo_seq.loop_id = metadata.loc[hvo_idx].at["loop_id"]
    hvo_seq.bpm = metadata.loc[hvo_idx].at["bpm"]


# subsetter


def _convert_hvos_array_to_subsets(
    hvos_array_tags, hvos_array_predicted, hvo_seqs_templates_
):
    hvo_seqs_templates = deepcopy(hvo_seqs_templates_)

    tags = list(set(hvos_array_tags))
    temp_dict = {tag: [] for tag in tags}
    hvo_index_dict = {tag: [] for tag in tags}

    for i in range(hvos_array_predicted.shape[0]):
        hvo_seqs_templates[i].hvo = hvos_array_predicted[i, :, :]
        temp_dict[hvos_array_tags[i]].append(hvo_seqs_templates[i])
        hvo_index_dict[hvos_array_tags[i]].append(i)

    tags = list(temp_dict.keys())
    subsets = list(temp_dict.values())

    return tags, subsets, hvo_index_dict


def save_parameters_to_json(params, params_path=None):
    if params_path is None:
        params_path = os.path.join("../dataset")
    if not os.path.exists(params_path):
        os.makedirs(params_path)
    params_json = os.path.join(params_path, params["dataset_name"] + "_params.json")
    with open(params_json, "w") as f:
        json.dump(params, f, cls=NpEncoder)


class NpEncoder(json.JSONEncoder):
    """
    Encoder to store parameters in numpy data types in json file
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_parameters_to_pickle(params, params_path=None):
    if params_path is None:
        params_path = os.path.join("../dataset", params["dataset_name"])
    if not os.path.exists(params_path):
        os.makedirs(params_path)
    params_pickle = os.path.join(params_path, params["dataset_name"] + "_params.pickle")
    with open(params_pickle, "wb") as f:
        pickle.dump(params, f)


def save_to_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def eval_log_freq(
    total_epochs,
    initial_epochs_lim,
    initial_step_partial,
    initial_step_all,
    secondary_step_partial,
    secondary_step_all,
    only_final=False,
):

    if only_final:
        return [total_epochs - 1], []

    if initial_epochs_lim >= total_epochs:
        epoch_save_partial = np.arange(total_epochs, step=initial_step_partial)
        epoch_save_all = np.arange(total_epochs, step=initial_step_all)
        return epoch_save_partial, epoch_save_all

    epoch_save_partial = np.arange(initial_epochs_lim, step=initial_step_partial)
    epoch_save_all = np.arange(initial_epochs_lim, step=initial_step_all)
    epoch_save_partial = np.append(
        epoch_save_partial,
        np.arange(
            start=initial_epochs_lim, step=secondary_step_partial, stop=total_epochs
        ),
    )
    epoch_save_all = np.append(
        epoch_save_all,
        np.arange(start=initial_epochs_lim, step=secondary_step_all, stop=total_epochs),
    )
    if total_epochs - 1 not in epoch_save_partial:
        epoch_save_partial = np.append(epoch_save_partial, total_epochs - 1)
    if total_epochs - 1 not in epoch_save_all:
        epoch_save_all = np.append(epoch_save_all, total_epochs - 1)
    return epoch_save_partial, epoch_save_all  # return epoch index
