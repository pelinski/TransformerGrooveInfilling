import numpy as np
from hvo_sequence.custom_dtypes import Tempo, Time_Signature


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


def create_grid_adapted_to_changes(grid_duration, ns_tempos, ns_time_signatures, beat_division_factors):

    # Creates the grid according to the features set for each tempo and time signature consistent segment
    def tempo_consistent_segment_boundaries():
        pass

    # Find tempo and time signature consistent segments

    """# Get the number of steps in each tempo and time signature consistent segment
    n_steps_per_segments = self.steps_per_segments

    # Get tempos and time signatures for each segment
    tempos, time_signatures = self.tempos_and_time_signatures_per_segments

    # Get and round up the number of bars per segments
    n_bars_per_segments = np.ceil(self.n_bars_per_segments)
    # print("n_bars_per_segments", n_bars_per_segments)

    # Keep track of the initial position of each segment grid section
    beginning_of_current_segment = 0

    # Variable for the final grid lines
    grid_lines = np.array([])

    for segment_ix, n_steps_per_segment in enumerate(n_steps_per_segments):
        # Create n-bars of grid lines
        segment_grid_lines, beginning_of_next_segment = create_grid_for_n_bars(
            n_bars=n_bars_per_segments[segment_ix],
            time_signature=time_signatures[segment_ix],
            tempo=tempos[segment_ix]
        )
        # print("len(segment_grid_lines)", len(segment_grid_lines))
        # print("beginning_of_next_segment", beginning_of_next_segment)
        # Trim the grid to fit required number of steps, also shift lines to start at the beginning of segment
        trimmed_moved_segment = segment_grid_lines[:n_steps_per_segment] + beginning_of_current_segment

        # Set the beginning of next segment
        if len(segment_grid_lines) > n_steps_per_segment:
            # in case the time sig or tempo are changed before end of measure
            # This ideally shouldn't happen but here for the sake of completeness
            beginning_of_current_segment = segment_grid_lines[n_steps_per_segment]
        else:
            beginning_of_current_segment = beginning_of_next_segment

        grid_lines = np.append(grid_lines, trimmed_moved_segment)
    # print("len(grid_lines)", len(grid_lines))
    return grid_lines"""


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
    # print(n_beats)

    # Empty grid
    grid_lines = np.array([])

    for ix, beat_div_factor in enumerate(time_signature.beat_division_factors):
        # print("np.arange(n_beats * beat_div_factor)", np.arange(n_beats * beat_div_factor)* beat_dur/ beat_div_factor)
        grid_lines = np.append(grid_lines, np.arange(n_beats * beat_div_factor) * beat_dur / beat_div_factor)

    beginning_of_next_bar = n_beats*beat_dur
    # print("len grid_lines 2", len(grid_lines))
    return np.unique(grid_lines), beginning_of_next_bar


