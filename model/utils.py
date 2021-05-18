import numpy as np
import itertools
import random
import json

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

    voice_idx = kwargs.get("voice_idx", [0, 1, 2, 3, 4])    # list of voices to remove
    min_n_voices_to_remove = kwargs.get("min_n_voices_to_remove", 1)    # min size of the combination
    max_n_voices_to_remove = kwargs.get("max_n_voices_to_remove", 3)    # max size of the combination
    prob = kwargs.get("prob", [1, 1, 1]) # prob of each n_voices_to_remove set in ascending order
    k = kwargs.get("k", 5)  # max number of combinations to return

    if len(voice_idx) < max_n_voices_to_remove: max_n_voices_to_remove = len(voice_idx)

    range_items = range(min_n_voices_to_remove, max_n_voices_to_remove + 1)

    assert (len(prob) == len(
        range_items)), "The prob list must be the same length as the range(min_n_voices_to_remove, max_n_voices_to_remove)"

    voice_idx_comb = []
    weights = []

    for i, n_voices_to_remove in enumerate(range_items):
        _voice_idx_comb = list(itertools.combinations(voice_idx, n_voices_to_remove))
        voice_idx_comb.extend(_voice_idx_comb)

        _weights = list(np.repeat(prob[i], len(_voice_idx_comb)))
        weights.extend(_weights)

    if k is not None: # if there is no k, return all possible combinations
        voice_idx_comb = random.choices(voice_idx_comb, weights=weights, k=k)

    return list(voice_idx_comb)


def get_sf_v_combinations(voices_parameters, max_aug_items,  max_n_sf, sfs_list):
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