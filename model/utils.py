import numpy as np
import itertools
import random

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

    range_items = range(min_n_voices_to_remove, max_n_voices_to_remove + 1)

    assert (len(prob) == len(
        range_items)), "The prob list must be the same length as the range(min_n_voices_to_remove, max_n_voices_to_remove)"
    assert (len(voice_idx) >= min_n_voices_to_remove and len(voice_idx) >= max_n_voices_to_remove), " " \
                                                                                                    "min_n_voices_to_remove <= len(voice_idx) <= max_n_voices_to_remove"

    voice_idx_comb = []
    weights = []

    for i, n_voices_to_remove in enumerate(range_items):
        _voice_idx_comb = list(itertools.combinations(voice_idx, n_voices_to_remove))
        voice_idx_comb.extend(_voice_idx_comb)

        _weights = list(np.repeat(prob[i], len(_voice_idx_comb)))
        weights.extend(_weights)

    if k is not None: # if there is no k, return all possible combinations
        voice_idx_comb = random.choices(voice_idx_comb, weights=weights, k=k)

    return voice_idx_comb