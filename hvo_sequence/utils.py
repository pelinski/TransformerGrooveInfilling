import numpy as np
from hvo_sequence.custom_dtypes import Tempo, Time_Signature
import math

def find_nearest(array, query):
    """
    Finds the closest entry in array to query. array must be sorted!
    @param array:                   a sorted array to search within
    @param query:                   value to find the nearest to
    @return index, array[index]:    the index in array closest to query, and the actual value
    """
    index = (np.abs(array-query)).argmin()
    return index, array[index]


def is_power_of_two(n):
    """
    Checks if a value is a power of two
    @param n:                               # value to check (must be int or float - otherwise assert error)
    @return:                                # True if a power of two, else false
    """
    if n is None:
        return False

    assert (isinstance(n, int) or isinstance(n, float)), "The value to check must be either int or float"

    if (isinstance(n, float) and n.is_integer()) or isinstance(n, int):
        # https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
        n = int(n)
        return (n & (n - 1) == 0) and n != 0
    else:
        return False


def find_pitch_and_tag(pitch_query, drum_mapping):
    """
    checks to which drum group the pitch belongs,
    then returns the index for group and the first pitch in group

    @param pitch_query:                 pitch_query for which the corresponding drum voice group is found
    @param drum_mapping:                dict of {'Drum Voice Tag': [midi numbers]}
    @return tuple of mapped_pitch,      index of the drum voice group the pitch belongs to
            instrument_tag,:            the first pitch in the corresponding drum group
                                        Note: Returns (None, None) if no matching group is found
            pitch_class_ix:             The ith group to which th pitch_query belongs
    """

    for ix, instrument_tag in enumerate(drum_mapping.keys()):
        if pitch_query in drum_mapping[instrument_tag]:
            mapped_pitch = drum_mapping[instrument_tag][0]
            return mapped_pitch, instrument_tag, ix

    # If pitch_query isn't in the pitch_class_list, return None, None, None
    return None, None, None


def create_grid_for_n_bars(n_bars, time_signature, tempo):

    # Creates n bars of grid lines according to the tempo and time_signature and the
    # requested beat_division_factors
    # ALso, returns the position of the beginning of the next bar/measure

    assert isinstance(time_signature, Time_Signature), "time_signature should be an instance of Time_Signature class"
    assert time_signature.is_ready_to_use, "There are missing fields in time_signature instance"
    assert isinstance(tempo, Tempo), "tempo should be an instance of Tempo class"
    assert tempo.is_ready_to_use, "There are missing fields in the tempo instance"

    # Calculate beat duration (beat defined based on signature denominator) --> not the perceived beat
    beat_dur = (60.0 / tempo.qpm) * 4.0 / time_signature.denominator

    # Calculate the number of beats
    n_beats = n_bars * time_signature.numerator

    # Empty grid
    grid_lines = np.array([])

    for ix, beat_div_factor in enumerate(time_signature.beat_division_factors):
        grid_lines = np.append(grid_lines, np.arange(n_beats * beat_div_factor) * beat_dur / beat_div_factor)

    beginning_of_next_bar = n_beats*beat_dur
    return np.unique(grid_lines), beginning_of_next_bar


def cosine_similarity(hvo_seq_a, hvo_seq_b):
    assert hvo_seq_a.hvo.shape[-1] == hvo_seq_b.hvo.shape[-1], "the two sequences must have the same last dimension"
    assert len(hvo_seq_a.tempos) == 1 and len(hvo_seq_a.time_signatures) == 1, \
        "Input A Currently doesn't support multiple tempos or time_signatures"
    assert len(hvo_seq_b.tempos) == 1 and len(hvo_seq_b.time_signatures) == 1, \
        "Input B Currently doesn't support multiple tempos or time_signatures"

    # Ensure a and b have same length by Padding the shorter sequence to match the longer one
    max_len = max(hvo_seq_a.hvo.shape[0], hvo_seq_b.hvo.shape[0])
    shape = max_len*hvo_seq_a.hvo.shape[-1]     # Flattened shape

    a = np.zeros(shape)
    b = np.zeros(shape)

    a[:(hvo_seq_a.hvo.shape[0]*hvo_seq_a.hvo.shape[1])] = hvo_seq_a.hvo.flatten()
    b[:hvo_seq_b.hvo.shape[0]*hvo_seq_b.hvo.shape[1]] = hvo_seq_b.hvo.flatten()

    return 1-np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def cosine_distance(hvo_seq_a, hvo_seq_b):
    return 1-cosine_similarity(hvo_seq_a, hvo_seq_b)


def fuzzy_Hamming_distance(velocity_grooveA, utiming_grooveA,
                           velocity_grooveB, utiming_grooveB,
                           beat_weighting=False):
    # Get fuzzy Hamming distance as velocity weighted Hamming distance, but with 1 metrical distance lookahead/back
    # and microtiming weighting
    # Microtiming must be in ms with nan whenever theres no hit
    assert velocity_grooveA.shape[0] == 32 and \
           velocity_grooveB.shape[0] == 32, "Currently only supports calculation on 2 bar " \
                                                    "loops in 4/4 and 16th note quantization"

    a = velocity_grooveA
    a_timing = utiming_grooveA
    b = velocity_grooveB
    b_timing = utiming_grooveB

    if beat_weighting is True:
        a = _weight_groove(a)
        b = _weight_groove(b)

    timing_difference = np.nan_to_num(a_timing - b_timing)

    x = np.zeros(a.shape)
    tempo = 120.0
    steptime_ms = 60.0 * 1000 / tempo / 4 # semiquaver step time in ms

    difference_weight = timing_difference / 125.
    difference_weight = 1+np.absolute(difference_weight)
    single_difference_weight = 400

    for j in range(a.shape[-1]):
        for i in range(31):
            if a[i,j] != 0.0 and b[i,j] != 0.0:
                x[i,j] = (a[i,j] - b[i,j]) * (difference_weight[i,j])
            elif a[i,j] != 0.0 and b[i,j] == 0.0:
                if b[(i+1) % 32, j] != 0.0 and a[(i+1) % 32, j] == 0.0:
                    single_difference = np.nan_to_num(a_timing[i,j]) - np.nan_to_num(b_timing[(i+1)%32,j]) + steptime_ms
                    if single_difference < 125.:
                        single_difference_weight = 1 + abs(single_difference_weight/steptime_ms)
                        x[i,j] = (a[i,j] - b[(i+1)%32,j]) * single_difference_weight
                    else:
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

                elif b[(i-1)%32,j] != 0.0 and a[(i-1)%32, j] == 0.0:
                    single_difference =  np.nan_to_num(a_timing[i,j]) - np.nan_to_num(b_timing[(i-1)%32,j]) - steptime_ms

                    if single_difference > -125.:
                        single_difference_weight = 1 + abs(single_difference_weight/steptime_ms)
                        x[i,j] = (a[i,j] - b[(i-1)%32,j]) * single_difference_weight
                    else:
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]
                else:
                    x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

            elif a[i,j] == 0.0 and b[i,j] != 0.0:
                if b[(i + 1) % 32, j] != 0.0 and a[(i + 1) % 32, j] == 0.0:
                    single_difference =  np.nan_to_num(a_timing[i,j]) - np.nan_to_num(b_timing[(i+1)%32,j]) + steptime_ms
                    if single_difference < 125.:
                        single_difference_weight = 1 + abs(single_difference_weight/steptime_ms)
                        x[i,j] = (a[i,j] - b[(i+1)%32,j]) * single_difference_weight
                    else:
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

                elif b[(i-1)%32,j] != 0.0 and a[(i-1)%32, j] == 0.0:
                    single_difference =  np.nan_to_num(a_timing[i,j]) - np.nan_to_num(b_timing[(i-1)%32,j]) - steptime_ms
                    if single_difference > -125.:
                        single_difference_weight = 1 + abs(single_difference_weight/steptime_ms)
                        x[i,j] = (a[i,j] - b[(i-1)%32,j]) * single_difference_weight

                    else:
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

                else: # if no nearby onsets, need to count difference between onset and 0 value.
                    x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

    fuzzy_distance = math.sqrt(np.dot(x.flatten(), x.flatten().T))
    return fuzzy_distance

def _weight_groove(_velocity_groove):
    # Metrical awareness profile weighting for hamming distance.
    # The rhythms in each beat of a bar have different significance based on GTTM

    # Repeat the awareness profile for each voice
    beat_awareness_weighting = np.array([1, 1, 1, 1,
                                         0.27, 0.27, 0.27, 0.27,
                                         0.22, 0.22, 0.22, 0.22,
                                         0.16, 0.16, 0.16, 0.16])

    # Match size of weighting factors and velocity groove
    if _velocity_groove.shape[0] > beat_awareness_weighting.shape[0]:
        pad_size = _velocity_groove.shape[0] - beat_awareness_weighting.shape[0]
        beat_awareness_weighting = np.pad(beat_awareness_weighting, (0, pad_size), mode='wrap').reshape(-1, 1)

    beat_awareness_weighting = beat_awareness_weighting[:_velocity_groove.shape[0], :]

    # Apply weight
    weighted_groove = _velocity_groove * beat_awareness_weighting

    return weighted_groove


def _reduce_part(part, metrical_profile):
    length = part.shape[0]
    for i in range(length):
        if part[i] <= 0.4:
            part[i] = 0
    for i in range(length):
        if part[i] != 0.:  # hit detected - must be figural or density transform - on pulse i.
            for k in range(-3, i):  # iterate through all previous events up to i.
                if part[k] != 0. and metrical_profile[k] < metrical_profile[i]:
                    # if there is a preceding event in a weaker pulse k (this is to be removed)

                    # groove[k,0] then becomes k, and can either be density of figural
                    previous_event_index = 0
                    for l in range(0, k):  # find strongest level pulse before k, with no events between m and k
                        if part[l] != 0.:  # find an event if there is one
                            previous_event_index = l
                        else:
                            previous_event_index = 0
                    m = max(metrical_profile[
                            previous_event_index:k])  # find the strongest level between previous event index and k.
                    # search for largest value in salience profile list between range l+1 and k-1. this is m.
                    if m <= k:  # density if m not stronger than k
                        part[k] = 0  # to remove a density transform just remove the note
                    if m > k:  # figural transform
                        part[m] = part[k]  # need to shift note forward - k to m.
                        part[k] = 0  # need to shift note forward - k to m.
        if part[i] == 0:
            for k in range(-3, i):
                if part[k] != 0. and metrical_profile[k] < metrical_profile[i]:  # syncopation detected
                    part[i] = part[k]
                    part[k] = 0.0
    return part

def _get_2bar_segments(part, steps_in_measure):
    """
    returns a list of np.array each element of which is a 2bar part
    Pads and replicates if len adjustment is needed
    """
    part = part.reshape(-1, 1) if part.ndim == 1 else part  # reshape to (n_steps, 1)

    # first make_sure_length_is_multiple of 16, if not append zero arrays
    if part.shape[0] % steps_in_measure != 0:
        pad_size = steps_in_measure - part.shape[0]
        part = np.pad(part, (0, pad_size), mode="constant")

    # match length to multiple 2 bars (if shorter repeat last bar)
    if part.shape[0] % (2 * steps_in_measure) != 0:
        part = np.append(part, part[-steps_in_measure:, :], axis=0)

    two_bar_segments = np.split(part, part.shape[0] // (steps_in_measure * 2))
    return two_bar_segments

def _get_kick_and_snare_syncopations(low, mid, high, i, metrical_profile, steps_in_measure=16):
    """
    Makes sure that the pattern is fitted and splitted into two bar measures
    then averages each 2bar segment's syncopation (calculated via get_monophonic_syncopation_for_2bar)

    :param part:
    :param metrical_profile:
    :param steps_in_measure:
    :return:
    """
    low_2bar_segs = _get_2bar_segments(low, steps_in_measure=steps_in_measure)
    mid_2bar_segs = _get_2bar_segments(mid, steps_in_measure=steps_in_measure)
    high_2bar_segs = _get_2bar_segments(high, steps_in_measure=steps_in_measure)

    kick_syncs_per_two_bar_segments = np.array([])
    snare_syncs_per_two_bar_segments = np.array([])

    for seg_ix, _ in enumerate(low_2bar_segs):
        kick_syncs_per_two_bar_segments = np.append(
            kick_syncs_per_two_bar_segments,
            _get_kick_syncopation_for_2bar(
                low_2bar_segs[seg_ix],
                mid_2bar_segs[seg_ix],
                high_2bar_segs[seg_ix],
                i,
                metrical_profile)
        )
        snare_syncs_per_two_bar_segments = np.append(
            snare_syncs_per_two_bar_segments,
            _get_snare_syncopation_for_2bar(
                low_2bar_segs[seg_ix],
                mid_2bar_segs[seg_ix],
                high_2bar_segs[seg_ix],
                i,
                metrical_profile)
        )

    return kick_syncs_per_two_bar_segments.mean(), snare_syncs_per_two_bar_segments.mean()


def _get_kick_syncopation_for_2bar(low, mid, high, i, metrical_profile):
    # Find instances  when kick syncopates against hi hat/snare on the beat.
    # For use in polyphonic syncopation feature

    kick_syncopation = 0
    k = 0
    next_hit = ""
    if low[i] == 1 and low[(i + 1) % 32] != 1 and low[(i + 2) % 32] != 1:
        for j in i + 1, i + 2, i + 3, i + 4:  # look one and two steps ahead only - account for semiquaver and quaver sync
            if mid[(j % 32)] == 1 and high[(j % 32)] != 1:
                next_hit = "Mid"
                k = j % 32
                break
            elif high[(j % 32)] == 1 and mid[(j % 32)] != 1:
                next_hit = "High"
                k = j % 32
                break
            elif high[(j % 32)] == 1 and mid[(j % 32)] == 1:
                next_hit = "MidAndHigh"
                k = j % 32
                break
            # if both next two are 0 - next hit == rest. get level of the higher level rest
        if mid[(i + 1) % 32] + mid[(i + 2) % 32] == 0.0 and high[(i + 1) % 32] + [(i + 2) % 32] == 0.0:
            next_hit = "None"

        if next_hit == "MidAndHigh":
            if metrical_profile[k] >= metrical_profile[i]:  # if hi hat is on a stronger beat - syncopation
                difference = metrical_profile[k] - metrical_profile[i]
                kick_syncopation = difference + 2
        elif next_hit == "Mid":
            if metrical_profile[k] >= metrical_profile[i]:  # if hi hat is on a stronger beat - syncopation
                difference = metrical_profile[k] - metrical_profile[i]
                kick_syncopation = difference + 2
        elif next_hit == "High":
            if metrical_profile[k] >= metrical_profile[i]:
                difference = metrical_profile[k] - metrical_profile[i]
                kick_syncopation = difference + 5
        elif next_hit == "None":
            if metrical_profile[k] > metrical_profile[i]:
                difference = max(metrical_profile[(i + 1) % 32], metrical_profile[(i + 2) % 32]) - metrical_profile[
                    i]
                kick_syncopation = difference + 6  # if rest on a stronger beat - one stream sync, high sync value
    return kick_syncopation


def _get_snare_syncopation_for_2bar(low, mid, high, i, metrical_profile):
    # Find instances  when snare syncopates against hi hat/kick on the beat
    # For use in polyphonic syncopation feature

    snare_syncopation = 0
    next_hit = ""
    k = 0
    if mid[i] == 1 and mid[(i + 1) % 32] != 1 and mid[(i + 2) % 32] != 1:
        for j in i + 1, i + 2, i + 3, i + 4:  # look one and 2 steps ahead only
            if low[(j % 32)] == 1 and high[(j % 32)] != 1:
                next_hit = "Low"
                k = j % 32
                break
            elif high[(j % 32)] == 1 and low[(j % 32)] != 1:
                next_hit = "High"
                k = j % 32
                break
            elif high[(j % 32)] == 1 and low[(j % 32)] == 1:
                next_hit = "LowAndHigh"
                k = j % 32
                break
        if low[(i + 1) % 32] + low[(i + 2) % 32] == 0.0 and high[(i + 1) % 32] + [(i + 2) % 32] == 0.0:
            next_hit = "None"

        if next_hit == "LowAndHigh":
            if metrical_profile[k] >= metrical_profile[i]:
                difference = metrical_profile[k] - metrical_profile[i]
                snare_syncopation = difference + 1  # may need to make this back to 1?)
        elif next_hit == "Low":
            if metrical_profile[k] >= metrical_profile[i]:
                difference = metrical_profile[k] - metrical_profile[i]
                snare_syncopation = difference + 1
        elif next_hit == "High":
            if metrical_profile[k] >= metrical_profile[i]:  # if hi hat is on a stronger beat - syncopation
                difference = metrical_profile[k] - metrical_profile[i]
                snare_syncopation = difference + 5
        elif next_hit == "None":
            if metrical_profile[k] > metrical_profile[i]:
                difference = max(metrical_profile[(i + 1) % 32], metrical_profile[(i + 2) % 32]) - metrical_profile[
                    i]
                snare_syncopation = difference + 6  # if rest on a stronger beat - one stream sync, high sync value
    return snare_syncopation


def get_monophonic_syncopation(part, metrical_profile, steps_in_measure=16):
    """
    Makes sure that the pattern is fitted and splitted into two bar measures
    then averages each 2bar segment's syncopation (calculated via get_monophonic_syncopation_for_2bar)

    :param part:
    :param metrical_profile:
    :param steps_in_measure:
    :return:
    """
    """    part = part.reshape(-1, 1) if part.ndim == 1 else part  # reshape to (n_steps, 1)
    
        # first make_sure_length_is_multiple of 16, if not append zero arrays
        if part.shape[0] % steps_in_measure != 0:
            pad_size = steps_in_measure - part.shape[0]
            part = np.pad(part, (0, pad_size), mode="constant")
    
        # match length to multiple 2 bars (if shorter repeat last bar)
        if part.shape[0] % (2 * steps_in_measure) != 0:
            part = np.append(part, part[-steps_in_measure:, :], axis=0)
    
        two_bar_segments = np.split(part, part.shape[0] // (steps_in_measure * 2))
    """
    two_bar_segments = _get_2bar_segments(part, steps_in_measure)
    syncs_per_two_bar_segments = np.array([])
    for segment in two_bar_segments:
        syncs_per_two_bar_segments = np.append(
            syncs_per_two_bar_segments,
            get_monophonic_syncopation_for_2bar(segment, metrical_profile)
        )

    return syncs_per_two_bar_segments.mean()

# todo: adapt to length variations (other than 2 bars)
def get_monophonic_syncopation_for_2bar(_2bar_part, metrical_profile):
    """
    Calculates monophonic syncopation levels for a 2 bar segment in 4-4 meter and 16th note quantization
    :param _2bar_part:
    :param metrical_profile:
    :return:
    """
    if all(np.isnan(_2bar_part)):
        return 0

    max_syncopation = 30.0
    syncopation = 0.0

    for i in range(len(_2bar_part)):
        if _2bar_part[i] != 0:
            if _2bar_part[(i + 1) % 32] == 0.0 and metrical_profile[(i + 1) % 32] > metrical_profile[i]:
                syncopation = float(syncopation + (
                    abs(metrical_profile[(i + 1) % 32] - metrical_profile[i])))  # * part[i])) #todo: velocity here?

            elif _2bar_part[(i + 2) % 32] == 0.0 and metrical_profile[(i + 2) % 32] > metrical_profile[i]:
                syncopation = float(syncopation + (
                    abs(metrical_profile[(i + 2) % 32] - metrical_profile[i])))  # * part[i]))

    return syncopation / max_syncopation


def get_weak_to_strong_ratio(velocity_groove):
    """
    returns the ratio of total weak onsets divided by all strong onsets
    strong onsets are onsets that occur on beat positions and weak onsets are the other ones
    """

    part = velocity_groove

    weak_hit_count = 0.0
    strong_hit_count = 0.0

    strong_positions = [0, 4, 8, 12, 16, 20, 24, 28]
    weak_positions = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17,
                      18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31]

    hits_count = np.count_nonzero(part)
    hit_indexes = np.nonzero(part)
    for i in range(hits_count):
        if len(hit_indexes) > 1:
            index = hit_indexes[0][i], hit_indexes[1][i]
        else:
            index = [hit_indexes[0][i]]
        if (index[0] % 32) in strong_positions:
            strong_hit_count += part[index]
        if (index[0] % 32) in weak_positions:
            weak_hit_count += part[index]

    if strong_hit_count>0:
        return weak_hit_count/strong_hit_count
    else:
        return 0


def _getmicrotiming_event_profile_1bar(microtiming_matrix, kick_ix, snare_ix, chat_ix, threshold):
    # Get profile of microtiming events for use in pushness/laidbackness/ontopness features
    # This profile represents the presence of specific timing events at certain positions in the pattern
    # Microtiming events fall within the following categories:
    #   Kick timing deviation - before/after metronome, before/after hihat, beats 1 and 3
    #   Snare timing deviation - before/after metronome, before/after hihat, beats 2 and 4
    # As such for one bar the profile contains 16 values.
    # The profile uses binary values - it only measures the presence of timing events, and the style features are
    # then calculated based on the number of events present that correspond to a certain timing feel.

    timing_to_grid_profile = np.zeros([8])
    timing_to_cymbal_profile = np.zeros([8])

    kick_timing_1 = microtiming_matrix[0, kick_ix]
    hihat_timing_1 = microtiming_matrix[0, chat_ix]
    snare_timing2 = microtiming_matrix[4, snare_ix]
    hihat_timing_2 = microtiming_matrix[4, chat_ix]
    kick_timing_3 = microtiming_matrix[8, kick_ix]
    hihat_timing_3 = microtiming_matrix[8, chat_ix]
    snare_timing4 = microtiming_matrix[12, snare_ix]
    hihat_timing_4 = microtiming_matrix[12, chat_ix]

    if kick_timing_1 > threshold:
        timing_to_grid_profile[0] = 1
    if kick_timing_1 < -threshold:
        timing_to_grid_profile[1] = 1
    if snare_timing2 > threshold:
        timing_to_grid_profile[2] = 1
    if snare_timing2 < -threshold:
        timing_to_grid_profile[3] = 1

    if kick_timing_3 > threshold:
        timing_to_grid_profile[4] = 1
    if kick_timing_3 < -threshold:
        timing_to_grid_profile[5] = 1
    if snare_timing4 > threshold:
        timing_to_grid_profile[6] = 1
    if snare_timing4 < -threshold:
        timing_to_grid_profile[7] = 1

    if kick_timing_1 > hihat_timing_1 + threshold:
        timing_to_cymbal_profile[0] = 1
    if kick_timing_1 < hihat_timing_1 - threshold:
        timing_to_cymbal_profile[1] = 1
    if snare_timing2 > hihat_timing_2 + threshold:
        timing_to_cymbal_profile[2] = 1
    if snare_timing2 < hihat_timing_2 - threshold:
        timing_to_cymbal_profile[3] = 1

    if kick_timing_3 > hihat_timing_3 + threshold:
        timing_to_cymbal_profile[4] = 1
    if kick_timing_3 < hihat_timing_3 - threshold:
        timing_to_cymbal_profile[5] = 1
    if snare_timing4 > hihat_timing_4 + threshold:
        timing_to_cymbal_profile[6] = 1
    if snare_timing4 < hihat_timing_4 - threshold:
        timing_to_cymbal_profile[7] = 1

    microtiming_event_profile_1bar = np.clip(timing_to_grid_profile + timing_to_cymbal_profile, 0, 1)

    return microtiming_event_profile_1bar
