import numpy as np
import note_seq
from note_seq.protobuf import music_pb2
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from bokeh.plotting import output_file, show, save
from bokeh.io import output_notebook
from bokeh.models import Span, Label
import warnings

from hvo_sequence.utils import is_power_of_two, create_grid_for_n_bars, find_pitch_and_tag
from hvo_sequence.custom_dtypes import Tempo, Time_Signature


class HVO_Sequence(object):

    def __init__(self, drum_mapping=None):
        """

        @param drum_mapping:                        a dictionary of grouped midi notes and corresponding names
                                                    {..., "Snare": [38, 37, 40], ...}

        Note:                                       A user can only manually modify the following parameters:

                                                        time signature
                                                        tempo
                                                        beat division factors
                                                        pitch class groups and tags
                                                        hvo array of (time_steps, 3*n_voices)
        """

        # Create ESSENTIAL fields used for the sequence
        # NOTE: DO NOT MODIFY THE BELOW MANGLED VARIABLES DIRECTLY
        #       RATHER USE THE PROPERTY GETTER AND SETTERS TO ENSURE
        #       DATATYPE CONSISTENCY

        """
        ADD VERSION ACCORDING TO: https://semver.org/
        
        Given a version number MAJOR.MINOR.PATCH, increment the:
        
        MAJOR version when you make incompatible API changes,
        MINOR version when you add functionality in a backwards compatible manner, and
        PATCH version when you make backwards compatible bug fixes.
        """

        self.__version = "0.0.1"

        self.__time_signatures = list()
        self.__tempos = list()
        self.__drum_mapping = None
        self.__hvo = None

        self.__force_vo_reset = True

        # Use property setters to initiate properties (DON"T ASSIGN ABOVE so that the correct datatype is checked)
        if drum_mapping:
            self.drum_mapping = drum_mapping

    #   ----------------------------------------------------------------------
    #   Property getters and setter wrappers for ESSENTIAL class variables
    #   ----------------------------------------------------------------------

    @property
    def __version__(self):
        return self.__version

    @property
    def time_signatures(self):
        return self.__time_signatures

    def add_time_signature(self, time_step=None, numerator=None, denominator=None, beat_division_factors=None):
        time_signature = Time_Signature(time_step=time_step, numerator=numerator,
                                        denominator=denominator, beat_division_factors=beat_division_factors)
        self.time_signatures.append(time_signature)
        return time_signature

    @property
    def tempos(self):
        return self.__tempos

    def add_tempo(self, time_step=None, qpm=None):
        tempo = Tempo(time_step=time_step, qpm=qpm)
        self.tempos.append(tempo)
        return tempo

    @property
    def drum_mapping(self):
        if not self.__drum_mapping:
            warnings.warn("drum_mapping is not specified")
        return self.__drum_mapping

    @drum_mapping.setter
    def drum_mapping(self, drum_map):
        # Ensure drum map is a dictionary
        assert isinstance(drum_map, dict), "drum_mapping should be a dict" \
                                           "of {'Drum Voice Tag': [midi numbers]}"

        # Ensure the values in each key are non-empty list of ints between 0 and 127
        for key in drum_map.keys():
            assert isinstance(drum_map[key], list), "map[{}] should be a list of MIDI numbers " \
                                                    "(int between 0-127)".format(drum_map[key])
            if len(drum_map[key]) >= 1:
                assert all([isinstance(val, int) for val in drum_map[key]]), "Expected list of ints in " \
                                                                             "map[{}]".format(drum_map[key])
            else:
                assert False, "map[{}] is empty --> should be a list of MIDI numbers " \
                              "(int between 0-127)".format(drum_map[key])

        if self.hvo is not None:
            assert self.hvo.shape[1] % len(drum_map.keys()) == 0, \
                "The second dimension of hvo should be three times the number of drum voices, len(drum_mapping.keys())"

        # Now, safe to update the local drum_mapping variable
        self.__drum_mapping = drum_map

    @property
    def hvo(self):
        return self.__hvo

    @hvo.setter
    def hvo(self, x):
        # Ensure x is a numpy.ndarray of shape (number of steps, 3*number of drum voices)
        assert isinstance(x, np.ndarray), "Expected numpy.ndarray of shape (time_steps, 3 * number of voices), " \
                                          "but received {}".format(type(x))
        # if drum_mapping already exists, the dimension at each time-step should be 3 times
        #   the number of voices in the drum_mapping
        if self.drum_mapping:
            assert x.shape[1] / len(self.drum_mapping.keys()) == 3, \
                "The second dimension of hvo should be three times the number of drum voices, len(drum_mapping.keys())"

        # Now, safe to update the local hvo score array
        self.__hvo = x

    #   --------------------------------------------------------------
    #   Useful properties calculated from ESSENTIAL class variables
    #   --------------------------------------------------------------

    @property
    def number_of_voices(self):
        calculable = self.is_drum_mapping_available(print_missing=True)
        if not calculable:
            print("can't calculate the number of voices as the drum_mapping is missing")
        else:
            return int(self.hvo.shape[1] / 3)

    @property
    def hits(self):
        """ ndarray of dimensions m,n where m is the number of time steps and n the number of drums in the current
            drum mapping (e.g. 9 for the reduced mapping). The values of the array are 1 or 0, indicating whether a
            hit occurs at that time step for that drum (1) or not (0).
        """
        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_drum_mapping_available(print_missing=True)])
        if not calculable:
            print("can't get hits as there is no hvo score previously provided")
        else:
            return self.hvo[:, :self.number_of_voices]

    def __is_hit_array_valid(self, hit_array):
        valid = True
        if len(self.hvo[:, :self.number_of_voices]) != len(hit_array):
            valid = False
            print("hit array length mismatch")
        if np.min(hit_array) < 0 or np.max(hit_array) > 1:
            valid = False
            print("invalid hit values in array, they must be 0 or 1")
        return valid

    @hits.setter
    def hits(self, hit_array):
        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_drum_mapping_available(print_missing=True)])
        if not calculable:
            print("can't set hits as there is no hvo score previously provided")
        else:
            if self.__is_hit_array_valid(hit_array):
                #TODO: if hit is 0 - force remove vels and offsets?
                self.hvo[:, :self.number_of_voices] = hit_array

    @property
    def velocities(self):
        """ ndarray of dimensions m,n where m is the number of time steps and n the number of drums in the current
            drum mapping (e.g. 9 for the reduced mapping). The values of the array are continuous floating point
            numbers from 0 to 1 indicating the velocity.
        """
        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_drum_mapping_available(print_missing=True)])
        if not calculable:
            print("can't get velocities as there is no hvo score previously provided")
        else:
            return self.hvo[:, self.number_of_voices: 2 * self.number_of_voices]

    def __is_vel_array_valid(self, vel_array):
        valid = True
        if len(self.hvo[:, self.number_of_voices: 2 * self.number_of_voices]) != len(vel_array):
            valid = False
            print("velocity array length mismatch")
        if np.min(vel_array) < 0 or np.max(vel_array) > 1:
            valid = False
            print("invalid velocity values in array, they must be between 0 and 1")
        return valid

    @velocities.setter
    def velocities(self, vel_array):
        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_drum_mapping_available(print_missing=True)])
        if not calculable:
            print("can't set velocities as there is no hvo score previously provided")
        else:
            if self.__is_vel_array_valid(vel_array):
                self.hvo[:, self.number_of_voices: 2 * self.number_of_voices] = vel_array

    @property
    def offsets(self):
        """ ndarray of dimensions m,n where m is the number of time steps and n the number of drums in the current
            drum mapping (e.g. 9 for the reduced mapping). The values of the array are continuous floating point
            numbers from -0.5 to 0.5 indicating the offset respect to the beat grid line that each hit is on.
        """
        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_drum_mapping_available(print_missing=True)])
        if not calculable:
            print("can't get offsets/utimings as there is no hvo score previously provided")
        else:
            return self.hvo[:, 2 * self.number_of_voices:]

    def __is_offset_array_valid(self, offset_array):
        valid = True
        if len(self.hvo[:, 2 * self.number_of_voices:]) != len(offset_array):
            valid = False
            print("offset array length mismatch")
        if np.min(offset_array) < -0.5 or np.max(offset_array) > 0.5:
            valid = False
            print("invalid offset values in array, they must be between -0.5 and 0.5")
        return valid

    @offsets.setter
    def offsets(self, offset_array):
        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_drum_mapping_available(print_missing=True)])
        if not calculable:
            print("can't set offsets as there is no hvo score previously provided")
        else:
            if self.__is_offset_array_valid(offset_array):
                self.hvo[:, 2 * self.number_of_voices:] = offset_array

    #   ----------------------------------------------------------------------
    #   Utility methods for segment derivation
    #   EACH SEGMENT MEANS A PART THAT TEMPO AND TIME SIGNATURE IS CONSTANT
    #   ----------------------------------------------------------------------

    @property
    def tempo_consistent_segment_boundaries(self):
        #   Returns time boundaries within which the tempo is constant
        #   upper bound (infinite) is always at 100000 seconds
        #   Example: tempo change at 1.5 seconds
        #            method returns --> [0, 1.5, 100000] --> meaning that tempo is
        #                               constant between 0≤t<1.5, 1.5≤t,100000
        #   If no tempo changes in the track (i.e. consistent tempo across all times
        #            method returns --> [0, 1000000]
        calculable = self.is_tempos_available(print_missing=True)
        if not calculable:
            warnings.warn("Can't carry out request as Tempos are not specified")
            return None
        else:
            time_regions = [0, 100000]  # 100000 to denote infinite
            for ix, tempo in enumerate(self.tempos):
                if ix > 0:  # Force 1st tempo to be at 0 even if doesn't start at the very beginning
                    time_regions.append(tempo.time_step)
            return list(np.unique(time_regions))

    @property
    def tempo_consistent_segment_lower_bounds(self):
        boundaries = self.tempo_consistent_segment_boundaries
        if boundaries is not None:
            return boundaries[:-1]
        else:
            return None

    @property
    def tempo_consistent_segment_upper_bounds(self):
        boundaries = self.tempo_consistent_segment_boundaries
        if boundaries is not None and self.hvo is not None:
            boundaries = boundaries[1:]
            boundaries[-1] = len(self.hvo)
            return boundaries
        else:
            return None

    @property
    def time_signature_consistent_segment_boundaries(self):
        #   Returns time boundaries within which the time signature is constant
        #   upper bound (infinite) is always at 100000 seconds
        #   Example: time signature change at 1.5 seconds
        #            method returns --> [0, 1.5, 100000] --> meaning that tempo is
        #                               constant between 0≤t<1.5, 1.5≤t,100000
        #   If no time signature changes in the track (i.e. consistent time signature across all times
        #            method returns --> [0, 1000000]
        calculable = self.is_time_signatures_available(print_missing=True)
        if not calculable:
            warnings.warn("Can't carry out request as Time Signatures are not specified")
            return None
        else:
            time_regions = [0, 100000]  # 100000 to denote infinite
            for ix, time_signature in enumerate(self.time_signatures):
                if ix > 0:  # Force 1st tempo to be at 0 even if doesn't start at the very beginning
                    time_regions.append(time_signature.time_step)
            return list(np.unique(time_regions))

    @property
    def time_signature_consistent_segment_lower_bounds(self):
        boundaries = self.time_signature_consistent_segment_boundaries
        if boundaries is not None:
            return boundaries[:-1]
        else:
            return None

    @property
    def time_signature_consistent_segment_upper_bounds(self):
        boundaries = self.time_signature_consistent_segment_boundaries
        if boundaries is not None and self.hvo is not None:
            boundaries = boundaries[1:]
            boundaries[-1] = len(self.hvo)
            return boundaries
        else:
            return None

    @property
    def segment_boundaries(self):
        #   Returns time boundaries within which the time signature and tempo are constant
        #   upper bound (infinite) is always at 100000 seconds
        #   Example: time signature change at 1.5 seconds and tempo change at 2 seconds
        #            method returns --> [0, 1.5, 2, 100000] --> meaning that tempo is
        #                               constant between 0≤t<1.5, 1.5≤t,100000
        #   If no time signature or tempo changes in the track:
        #            method returns --> [0, 1000000]
        calculable = all([self.is_time_signatures_available(print_missing=True),
                          self.is_tempos_available(print_missing=True)])
        if not calculable:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            _time_regions = list()
            _time_regions.extend(self.tempo_consistent_segment_boundaries)
            _time_regions.extend(self.time_signature_consistent_segment_boundaries)
            return list(np.unique(_time_regions))

    @property
    def segment_lower_bounds(self):
        boundaries = self.segment_boundaries
        if boundaries is not None:
            return boundaries[:-1]
        else:
            return None

    @property
    def segment_upper_bounds(self):
        # Returns exclusive upper bounds of each segment
        boundaries = self.segment_boundaries

        if boundaries is not None and self.hvo is not None:
            upper_bounds = boundaries[1:]
            upper_bounds[-1] = len(self.hvo)
            return upper_bounds
        else:
            return None

    # IN THE DOCUMENTATION MENTION THAT EACH SEGMENT MEANS A PART THAT TEMPO AND TIME SIGNATURE IS CONSTANT
    @property
    def tempos_and_time_signatures_per_segments(self):
        # Returns two lists: 1. lists of tempos per segment
        #                     2. lists of time signature for each segment
        # Segments are defined as parts of the score where the tempo and time signature don't change
        calculable = all([self.is_time_signatures_available(print_missing=True),
                          self.is_tempos_available(print_missing=True)])
        if not calculable:
            warnings.warn("Can't carry out request as above fields are missing")
            return None, None
        else:
            tempos = list()
            time_signatures = list()

            """segment_bounds = self.segment_boundaries
            lower_bounds = segment_bounds[:-1]"""

            for segment_lower_bound in self.segment_lower_bounds:
                # Recursively find what the tempo and time signature is at the lower bound
                distance_from_tempo_boundaries = np.array(self.tempo_consistent_segment_boundaries) - \
                                                 segment_lower_bound
                tempo_index = np.where(distance_from_tempo_boundaries <= 0,
                                       distance_from_tempo_boundaries, -np.inf).argmax()
                distance_from_time_sig_boundaries = np.array(self.time_signature_consistent_segment_boundaries) - \
                                                    segment_lower_bound

                time_signature_index = np.where(distance_from_time_sig_boundaries <= 0,
                                                distance_from_time_sig_boundaries, -np.inf).argmax()

                tempos.append(self.tempos[tempo_index])
                time_signatures.append(self.time_signatures[time_signature_index])

            return tempos, time_signatures

    @property
    def number_of_segments(self):
        # Returns the number of segments in each of which the tempo and time signature is consistent
        calculable = all([self.is_time_signatures_available(print_missing=True),
                          self.is_tempos_available(print_missing=True)])
        if not calculable:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            segment_bounds = self.segment_boundaries
            return len(segment_bounds) - 1

    @property
    def beat_durations_per_segments(self):

        # Calculates the duration of each beat in seconds if time signature and qpm are available
        calculable = all([self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])
        if not calculable:
            warnings.warn("Beat durations per segments can be calculated as the above fields are missing")
            return None
        else:
            beat_durs_per_segment = list()
            segment_lower_boundaries = np.array(self.segment_lower_bounds)
            for ix, segment_lower_bound in enumerate(segment_lower_boundaries):
                tempo, time_signature = self.tempo_and_time_signature_at_step(segment_lower_bound)
                beat_durs_per_segment.append((60.0 / tempo.qpm) * 4.0 / time_signature.denominator)
            return beat_durs_per_segment

    @property
    def steps_per_beat_per_segments(self):
        # Calculates the total number of steps in each beat belonging
        # to each tempo & time signature consistent segment
        tempos_per_seg, time_signatures_per_seg = self.tempos_and_time_signatures_per_segments

        if tempos_per_seg is not None and time_signatures_per_seg is not None:
            steps_per_beat_per_segment = []
            for ix, time_signature_in_segment_ix in enumerate(time_signatures_per_seg):
                # calculate the number of steps per beat for corresponding beat_division_factors
                beat_divs = time_signature_in_segment_ix.beat_division_factors
                mock_beat_grid_lines = np.concatenate([np.arange(beat_div) / beat_div for beat_div in beat_divs])
                steps_per_beat_per_segment.append(len(np.unique(mock_beat_grid_lines)))

            return steps_per_beat_per_segment
        else:
            warnings.warn("Tempo or Time Signature missing")
            return None

    @property
    def steps_per_segments(self):
        # number of steps in each segment (tempo & time signature consistent segment)

        segment_lower_bounds = self.segment_lower_bounds
        segment_upper_bounds = self.segment_upper_bounds

        if segment_lower_bounds is not None and self.is_hvo_score_available(print_missing=True) is not None:
            return list(np.array(segment_upper_bounds) - np.array(segment_lower_bounds))
        else:
            warnings.warn("Tempo or Time Signature missing")
            return None

    @property
    def n_beats_per_segments(self):
        # Calculate the number of beats in each tempo and time signature consistent segment of score/sequence

        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])

        if calculable:
            n_beats_per_seg = list()

            steps_per_beat_per_segments = self.steps_per_beat_per_segments
            segment_boundaries = self.segment_boundaries
            # replace 100000 (upper bound for last segment) in segment_boundaries with actual length of hvo score
            segment_boundaries[-1] = self.total_number_of_steps

            for ix, steps_per_beat_per_segment_ix in enumerate(steps_per_beat_per_segments):
                total_steps_in_segment_ix = segment_boundaries[ix + 1] - segment_boundaries[ix]
                beats_in_segment_ix = total_steps_in_segment_ix / steps_per_beat_per_segment_ix
                n_beats_per_seg.append(beats_in_segment_ix)

            return n_beats_per_seg

        else:
            return None

    @property
    def n_bars_per_segments(self):
        # Returns the number of bars in each of the tempo and time signature consistent segments
        n_beats_per_segments = self.n_beats_per_segments
        _, time_signatures = self.tempos_and_time_signatures_per_segments

        if n_beats_per_segments is not None and time_signatures is not None:
            n_bars_per_segments = list()
            for segment_ix, n_beats_per_segment_ix in enumerate(n_beats_per_segments):
                n_bars_in_segment_ix = n_beats_per_segment_ix / time_signatures[segment_ix].numerator
                n_bars_per_segments.append(n_bars_in_segment_ix)
            return n_bars_per_segments
        else:
            warnings.warn("Can't execute request as above fields are missing")
            return None

    @property
    def segment_durations(self):
        beat_durations_per_segments = self.beat_durations_per_segments

        if beat_durations_per_segments is not None:

            n_beats_per_segments = self.n_beats_per_segments

            segment_durations = list()

            for segment_ix, beat_durations_in_segment_ix in enumerate(beat_durations_per_segments):
                segment_durations.append(beat_durations_in_segment_ix * self.n_beats_per_segments[segment_ix])

            return segment_durations
        else:
            return None

    def tempo_segment_index_at_step(self, step_ix):
        # gets the index of the tempo segment in which the step is located
        if self.is_tempos_available(print_missing=True) is None:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            # Get the boundaries where tempo or time signature changes
            tempo_boundaries = np.array(self.tempo_consistent_segment_boundaries)
            # find the distance between time stamp and the boundaries of the tempo segments
            tempo_boundaries_distance = np.array(tempo_boundaries) - step_ix
            # Figure out to which tempo segment the step belongs
            #  corresponding region will be identified with the index of the last negative value
            tempo_ix = np.where(tempo_boundaries_distance <= 0, tempo_boundaries_distance, -np.inf).argmax()
            return tempo_ix

    def time_signature_segment_index_at_step(self, step_ix):
        # gets the index of the time_signature segment in which the step is located
        if self.is_time_signatures_available(print_missing=True) is None:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            # Get the boundaries where tempo or time signature changes
            time_signature_boundaries = np.array(self.time_signature_consistent_segment_boundaries)
            # find the distance between time stamp and the boundaries of the tempo segments
            time_signature_boundaries_distance = np.array(time_signature_boundaries) - step_ix
            # Figure out to which tempo segment the step belongs
            #  corresponding region will be identified with the index of the last negative value
            time_signature_ix = np.where(time_signature_boundaries_distance <= 0,
                                         time_signature_boundaries_distance, -np.inf).argmax()
            return time_signature_ix

    def segment_index_at_step(self, step_ix):
        # gets the index of the tempo and time_signature consistent segments in which the step is located
        calculable = all([self.is_time_signatures_available(print_missing=True),
                          self.is_tempos_available(print_missing=True)])
        if calculable is None:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            """# Get the boundaries where tempo or time signature changes
            segment_boundaries = np.array(self.segment_boundaries)
            # find the distance between time stamp and the boundaries of the tempo segments
            segment_boundaries_distance = np.array(segment_boundaries) - step_ix
            # Figure out to which tempo segment the step belongs
            #  corresponding region will be identified with the index of the last negative value
            segment_ix = np.where(segment_boundaries_distance <= 0,
                                                   segment_boundaries_distance, -np.inf).argmax()"""

            # find the correct segment --> correct segment is where the step is larger/eq to the lower bound
            #       and smaller than upper bound
            lower_boundaries = np.array(self.segment_lower_bounds)
            upper_boundaries = np.array(self.segment_upper_bounds)
            check_sides = np.where(lower_boundaries <= step_ix, True, False
                                   ) * np.where(upper_boundaries > step_ix, True, False)
            segment_ix = np.argwhere(check_sides == True)[0, 0]

            return segment_ix

    def tempo_and_time_signature_at_step(self, step_ix):
        # Figures out which tempo and time signature consistent segment the step belongs to
        # and then returns the corresponding tempo and time signature
        distance_from_tempo_boundaries = np.array(self.tempo_consistent_segment_boundaries) - step_ix
        tempo_index = np.where(distance_from_tempo_boundaries <= 0,
                               distance_from_tempo_boundaries, -np.inf).argmax()
        distance_from_time_sig_boundaries = np.array(self.time_signature_consistent_segment_boundaries) - step_ix
        time_signature_index = np.where(distance_from_time_sig_boundaries <= 0,
                                        distance_from_time_sig_boundaries, -np.inf).argmax()
        return self.tempos[tempo_index], self.time_signatures[time_signature_index]

    def step_position_from_segment_beginning(self, step_ix):
        # Returns the position of an step with respect to the beginning of the corresponding
        # tempo and time signature consistent segment

        # Find corresponding segment index for step_ix
        segment_ix = self.segment_index_at_step(step_ix)

        if segment_ix is None:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            # Return the distance of the index from the lower bound of segment
            return step_ix - self.segment_lower_bounds[segment_ix]

    def step_position_from_time_signature_segment_beginning(self, step_ix):
        # Returns the position of an step with respect to the beginning of the corresponding
        # segment in which the time_signature is constant

        # Find corresponding segment index for step_ix
        time_signature_segment_ix = self.time_signature_segment_index_at_step(step_ix)

        if time_signature_segment_ix is None:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            # Return the distance of the index from the lower bound of segment
            return step_ix - self.time_signature_consistent_segment_lower_bounds[time_signature_segment_ix]

    """@property
    def bar_lens_per_segments(self):
        # Returns """

    """@property
    def bar_len(self):
        # Calculates the duration of each bar in seconds if time signature and qpm are available
        calculable = all([self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])
        if calculable:
            return self.beat_dur * self.time_signature["numerator"]
        else:
            return None"""

    @property
    def total_len(self):
        # Calculates the total length of score in seconds if hvo score, time signature and qpm are available
        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])
        if calculable:
            return self.grid_lines[-1] + 0.5 * (self.grid_lines[-1] - self.grid_lines[-2])
        else:
            return None

    @property
    def total_number_of_steps(self):
        # Calculates the total number of steps in the score/sequence
        calculable = self.is_hvo_score_available(print_missing=True)
        if calculable:
            return self.hvo.shape[0]
        else:
            return 0

    @property
    def grid_type_per_segments(self):
        # Returns a list of the type of grid per tempo and time signature consistent segment
        # Type at each segment can be:
        #   1. binary:  if the grid lines lie on 2^n divisions of each beat
        #   2. triplet: if the grid lines lie on divisions of each beat that are multiples of 3
        #   3. mix:     if the grid is a combination of binary and triplet

        calculable = all([self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])

        if calculable:
            grid_types_per_segments = list()
            _, time_signatures_per_seg = self.tempos_and_time_signatures_per_segments
            for ix, time_sig_in_seg_ix in enumerate(time_signatures_per_seg):
                po2_grid_flag, triplet_grid_flag = None, None
                for factor in time_sig_in_seg_ix.beat_division_factors:
                    if is_power_of_two(factor):
                        po2_grid_flag = True
                    if factor % 3 == 0:
                        triplet_grid_flag = True

                if po2_grid_flag and triplet_grid_flag:
                    grid_types_per_segments.append("mix")
                elif po2_grid_flag:
                    grid_types_per_segments.append("binary")
                else:
                    grid_types_per_segments.append("triplet")
            return grid_types_per_segments

        else:
            return None

    @property
    def is_grid_equally_distanced_per_segments(self):
        # for each tempo and time signature consistent segment
        #  Checks if the grid is uniformly placed (in case of binary or triplet grids)
        #       or if the grid is non-uniformly oriented (in case of mix grid)
        grid_type_per_segments = self.grid_type_per_segments
        if grid_type_per_segments is not None:
            is_grid_equally_distanced_per_segments = list()
            for grid_type_per_segment in grid_type_per_segments:
                is_grid_equally_distanced_per_segments.append(True if grid_type_per_segments == "mix" else False)
            return is_grid_equally_distanced_per_segments
        else:
            return None

    @property
    def major_and_minor_grid_line_indices(self):
        # Returns major and minor grid line indices (corresponding to 1st dimension of self.hvo and self.grid_lines)
        # Major lines lie on the beginning of each beat --> multiples of number of steps in each beat
        # Minor lines lie in between major gridlines    --> not multiples of number of steps in each beat

        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])

        if not calculable:
            warnings.warn("Above fields are required for calculating major/minor grid line positions")
            return None, None

        grids_with_types = self.grid_lines_with_types
        return grids_with_types["major_grid_line_indices"], grids_with_types["minor_grid_line_indices"]

    @property
    def major_and_minor_grid_lines(self):
        # Returns major and minor grid lines
        # Major lines lie on the beginning of each beat --> multiples of number of steps in each beat
        # Minor lines lie in between major gridlines    --> not multiples of number of steps in each beat

        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])

        if not calculable:
            warnings.warn("Above fields are required for calculating major/minor grid line positions")
            return None, None

        grids_with_types = self.grid_lines_with_types
        return grids_with_types["major_grid_lines"], grids_with_types["minor_grid_lines"]

    @property
    def downbeat_indices(self):
        # Returns the indices of the grid_lines where a downbeat occurs.
        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])

        if not calculable:
            warnings.warn("Above fields are required for calculating downbeat grid line positions")
            return None, None

        grids_with_types = self.grid_lines_with_types
        return grids_with_types["downbeat_grid_line_indices"]

    @property
    def downbeat_positions(self):
        # Returns the indices of the grid_lines where a downbeat occurs.
        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])

        if not calculable:
            warnings.warn("Above fields are required for calculating downbeat grid line positions")
            return None, None

        grids_with_types = self.grid_lines_with_types
        return grids_with_types["downbeat_grid_lines"]

    @property
    def starting_measure_indices(self):
        # A wrapper for downbeat_positions (for the sake of code readability)
        return self.downbeat_indices

    @property
    def starting_measure_positions(self):
        # A wrapper for downbeat_positions (for the sake of code readability)
        return self.downbeat_positions

    @property
    def grid_lines_with_types(self):

        """

        """
        major_grid_lines = [
            0]  # Happens at the beggining of beats! --> Beat Pos depends on Time_signature only (If Time_sig changes before an expected beat position, force reset beat position)
        minor_grid_lines = []  # Any Index that's not major
        downbeat_grid_lines = [
            0]  # every nth major_ix where n is time sig numerator in segment (downbeat always measured from the beginning of a time signature time stamp)

        major_grid_line_indices = [0]
        minor_grid_line_indices = []
        downbeat_grid_line_indices = [0]

        grid_lines = [0]

        ts_consistent_lbs = self.time_signature_consistent_segment_lower_bounds
        ts_consistent_ubs = self.time_signature_consistent_segment_upper_bounds

        current_step = 0
        for ts_consistent_seg_ix, (ts_lb, ts_up) in enumerate(zip(ts_consistent_lbs, ts_consistent_ubs)):
            major_grid_lines.append(grid_lines[-1])
            major_grid_line_indices.append(len(grid_lines))
            downbeat_grid_lines.append(grid_lines[-1])
            downbeat_grid_line_indices.append(len(grid_lines))

            # Figure out num_steps in each beat as well as the ratios of beat_dur for each time increase
            time_sig = self.time_signatures[self.time_signature_segment_index_at_step(ts_lb)]

            delta_t_ratios = np.array([])
            for beat_div_factor in time_sig.beat_division_factors:
                delta_t_ratios = np.append(delta_t_ratios, np.arange(0, 1, 1.0 / beat_div_factor))
            delta_t_ratios = np.unique(np.append(delta_t_ratios, 1))
            delta_t_ratios = delta_t_ratios[1:] - delta_t_ratios[:-1]
            steps_per_beat_in_seg = len(delta_t_ratios)

            for step_ix in range(ts_lb - ts_lb, ts_up - ts_lb):  # For each ts, re-start counting from 0
                actual_step_ix = step_ix if ts_consistent_seg_ix == 0 else step_ix + len(grid_lines) - 1
                tempo = self.tempos[self.tempo_segment_index_at_step(actual_step_ix)]
                beat_duration_at_step = (60.0 / tempo.qpm) * 4.0 / time_sig.denominator
                grid_lines.append(grid_lines[-1] + delta_t_ratios[step_ix % steps_per_beat_in_seg] * \
                                  beat_duration_at_step)
                current_step = current_step + 1
                if (step_ix + 1) % (steps_per_beat_in_seg) == 0:
                    major_grid_lines.append(grid_lines[-1])
                    major_grid_line_indices.append(current_step)
                    if (step_ix + 1) % (time_sig.numerator * steps_per_beat_in_seg) == 0:
                        downbeat_grid_lines.append(grid_lines[-1])
                        downbeat_grid_line_indices.append(current_step)
                else:
                    minor_grid_lines.append(grid_lines[-1])
                    minor_grid_line_indices.append(current_step)

        output = {
            "grid_lines": grid_lines,
            "major_grid_lines": major_grid_lines,
            "minor_grid_lines": minor_grid_lines,
            "downbeat_grid_lines": downbeat_grid_lines,
            "major_grid_line_indices": major_grid_line_indices,
            "minor_grid_line_indices": minor_grid_line_indices,
            "downbeat_grid_line_indices": downbeat_grid_line_indices
        }

        return output

    @property
    def grid_lines(self):

        """

        """
        return np.array(self.grid_lines_with_types["grid_lines"])

    #   ----------------------------------------------------------------------
    #   Utility methods to check whether required properties are
    #       available for carrying out a request
    #   All methods have two args: print_missing, print_available
    #       set either to True to get additional info for debugging
    #
    #   Assuming that the local variables haven't been modified directly,
    #   No need to check the validity of data types if they are available
    #       as this is already done in the property.setters
    #   ----------------------------------------------------------------------

    def is_time_signatures_available(self, print_missing=False, print_available=False):
        # Checks whether time_signatures are already specified and necessary fields are filled
        time_signatures_ready_to_use = list()
        if self.time_signatures is not None:
            for time_signature in self.time_signatures:
                time_signatures_ready_to_use.append(time_signature.is_ready_to_use)

        if not all(time_signatures_ready_to_use):
            for ix, ready_status in enumerate(time_signatures_ready_to_use):
                if ready_status is not True:
                    print("There are missing fields in Time_Signature {}: {}".format(ix, self.time_signatures[ix]))
            return False
        else:
            return True

    def is_tempos_available(self, print_missing=False, print_available=False):
        # Checks whether tempos are already specified and necessary fields are filled
        tempos_ready_to_use = list()
        if self.tempos is not None:
            for tempo in self.tempos:
                tempos_ready_to_use.append(tempo.is_ready_to_use)

        if not all(tempos_ready_to_use):
            for ix, ready_status in enumerate(tempos_ready_to_use):
                if ready_status is not True:
                    print("There are missing fields in Tempo {}: {}".format(ix, self.tempos[ix]))
            return False
        else:
            return True

    def is_drum_mapping_available(self, print_missing=False, print_available=False):
        # Checks whether drum_mapping is already specified
        if not self.is_drum_mapping_available:
            if print_missing:
                print("\n|---- drum_mapping is not specified. Currently empty ")
            return False
        else:
            if print_available:
                print("\n|---- drum_mapping is available and specified as {}".format(self.drum_mapping))
            return True

    def is_hvo_score_available(self, print_missing=False, print_available=False):
        # Checks whether hvo score array is already specified
        if not self.is_drum_mapping_available:
            if print_missing:
                print("\n|---- HVO score is not specified. Currently empty ")
            return False
        else:
            if print_available:
                print("\n|---- HVO score is available and specified as {}".format(self.hvo))
            return True

    #   --------------------------------------------------------------
    #   Utilities to modify voices in the sequence
    #   --------------------------------------------------------------

    @property
    def force_vo_reset(self):
        return self.__force_vo_reset

    @force_vo_reset.setter
    def force_vo_reset(self, x):

        # Ensure x is a boolean
        assert isinstance(x, bool), "Expected boolean " \
                                    "but received {}".format(type(x))

        # Now, safe to update the local force_vo_reset boolean
        self.__force_vo_reset = x

    def reset_voices(self, voice_idx=None, reset_hits=True, reset_velocity=True, reset_offsets=True):
        """
        @param voice_idx:                   voice index or indexes (can be single value or list of values) according
                                            to the drum mapping
        @param reset_hits:                  if True, resets hits in every voice in voice_idx. can be a list of
                                            booleans where each index corresponds to a voice in voice_idx list
        @param reset_velocity:              if True, resets velocities in every voice in voice_idx. can be a list of
                                            booleans where each index corresponds to a voice in voice_idx list
        @param reset_offsets:               if True, resets offsets in every voice in voice_idx. can be a list of
                                            booleans where each index corresponds to a voice in voice_idx list
        """

        if voice_idx is None:
            warnings.warn("Pass a voice index or a list of voice indexes to be reset")
            return None

        # for consistency, turn voice_idx int into list
        if isinstance(voice_idx, int):
            voice_idx = [voice_idx]

        # props list lengths must be equal to voice_idx length
        if isinstance(reset_hits, list) and len(reset_hits) != len(voice_idx):
            warnings.warn("Reset_hits must be boolean or list of booleans of length equal to voice_idx")
            return None
        if isinstance(reset_velocity, list) and len(reset_velocity) != len(voice_idx):
            warnings.warn("Reset_velocities must be boolean or list of booleans of length equal to voice_idx")
            return None
        if isinstance(reset_offsets, list) and len(reset_offsets) != len(voice_idx):
            warnings.warn("Reset_offsets must be boolean or list of booleans of length equal to voice_idx")
            return None

        n_inst = len(self.drum_mapping)  # number of instruments in the mapping
        n_frames = self.hvo.shape[0]  # number of frames

        # iterate voices in voice_idx list
        for i, _voice_idx in enumerate(voice_idx):

            if _voice_idx not in range(n_inst):
                warnings.warn("Instrument index not in drum mapping")
                return None

            h_idx = _voice_idx  # hits
            v_idx = _voice_idx + n_inst  # velocity
            o_idx = _voice_idx + 2 * n_inst  # offset

            _reset_hits = reset_hits
            _reset_velocity = reset_velocity
            _reset_offsets = reset_offsets

            # if props are given as list (one condition for each voice), assign value to _reset_prop
            if isinstance(reset_hits, list):
                _reset_hits = reset_hits[i]
            if isinstance(reset_velocity, list):
                _reset_velocity = reset_velocity[i]
            if isinstance(reset_offsets, list):
                _reset_offsets = reset_offsets[i]

            if self.force_vo_reset and _reset_hits:
                if not _reset_velocity or not _reset_offsets:
                    _reset_velocity = True
                    _reset_offsets = True
                    warnings.warn("Forcing velocity and offset reset for voice {}."\
                                  " Deactivate setting force_vo_reset property to False".format(_voice_idx))

            # reset voice
            if _reset_hits:
                self.hvo[:, h_idx] = np.zeros(n_frames)
            if _reset_velocity:
                self.hvo[:, v_idx] = np.zeros(n_frames)
            if _reset_offsets:
                self.hvo[:, o_idx] = np.zeros(n_frames)

    #   -------------------------------------------------------------
    #   Method to flatten all voices to a single tapped sequence
    #   -------------------------------------------------------------

    def flatten_voices(self, get_velocities=True, reduce_dim=False, voice_idx=0):

        """ Flatten all voices into a single tapped sequence. If there are several voices hitting at the same
        time step, the loudest one will be selected and its offset will be kept, however velocity is discarded
        (set to the maximum).
        
        Parameters
        ----------
        get_velocities: bool
            When set to True the function will return an hvo array with the hits, velocities and
            offsets of the voice with the hit that has maximum velocity at each time step, when
            set to False it will do the same operation but only return the hits and offsets, discarding
            the velocities at the end.
        reduce_dim: bool
            When set to False the hvo array returned will have the same number of voices as the original
            hvo, with the tapped sequenced in the selected voice_idx and the rest of the voices set to 0.
            When True, the hvo array returned will have only one voice.
        voice_idx : int
            The index of the voice where the tapped sequence will be stored. 0 by default.
            If reduce_dim is True, this index is disregarded as the flat hvo will only have
            one voice.
        """

        # Store number of voices
        n_voices_keep = self.number_of_voices

        if not reduce_dim:
            # Make sure voice index is within range
            assert (self.number_of_voices > voice_idx >= 0), "invalid voice index"
        else:
            # Overwrite voice index
            voice_idx = 0
            # Overwrite number of voices: we only want to keep one
            n_voices_keep = 1

        # Get number of time steps
        time_steps = self.hvo.shape[0]

        # Copying to new arrays since we don't want to modify existing internal hvo
        _hits = self.hits.copy()
        _velocities = self.velocities.copy()
        _offsets = self.offsets.copy()

        for i in np.arange(time_steps):

            # initialize 3 temporary arrays
            new_hits, new_velocities, new_offsets = np.zeros((3, n_voices_keep))

            # if there is any hit at that timestep
            if np.any(_hits[i, :] == 1):

                # get index of voice with max velocity
                _idx_keep = np.argmax(_velocities[i, :])

                # copy the hit, velocity and offset of the voice in that timestep with the maximum velocity
                new_hits[voice_idx] = 1
                new_velocities[voice_idx] = _velocities[i, _idx_keep]
                new_offsets[voice_idx] = _offsets[i, _idx_keep]

            # copy time step into bigger array
            _hits[i, :] = new_hits
            _velocities[i, :] = new_velocities
            _offsets[i, :] = new_offsets

        if reduce_dim:
            # if we want to return only 1 voice (instead of e.g. 9 with all the others to 0)
            # we remove the other dimensions and transform it into a 2-dim array so that the
            # concatenate after will join the arrays in the right axis
            _hits = np.array([_hits[:, voice_idx]]).T
            _velocities = np.array([_velocities[:, voice_idx]]).T
            _offsets = np.array([_offsets[:, voice_idx]]).T

        # concatenate arrays
        flat_hvo = np.concatenate((_hits, _velocities, _offsets), axis=1) if get_velocities else np.concatenate((_hits, _offsets), axis=1)
        return flat_hvo




    #   --------------------------------------------------------------
    #   Utilities to import/export different score formats such as
    #       1. NoteSequence, 2. HVO array, 3. Midi
    #   --------------------------------------------------------------

    def to_note_sequence(self, midi_track_n=9):
        """
        Exports the hvo_sequence to a note_sequence object

        @param midi_track_n:    the midi track channel used for the drum scores
        @return:
        """
        convertible = all([self.is_hvo_score_available(print_missing=True),
                           self.is_tempos_available(print_missing=True),
                           self.is_time_signatures_available(print_missing=True)])
        if not convertible:
            warnings.warn("Above fields need to be provided so as to convert the hvo_sequence into a note_sequence")
            return None

        # Create a note sequence instance
        ns = music_pb2.NoteSequence()

        # get the number of allowed drum voices
        n_voices = len(self.__drum_mapping.keys())

        # find nonzero hits tensor of [[position, drum_voice]]
        pos_instrument_tensors = np.transpose(np.nonzero(self.__hvo[:, :n_voices]))

        # Set note duration as 1/2 of the smallest grid distance
        note_duration = np.min(self.grid_lines[1:] - self.grid_lines[:-1]) / 2.0

        # Add notes to the NoteSequence object
        for drum_event in pos_instrument_tensors:  # drum_event -> [grid_position, drum_voice_class]
            grid_pos = drum_event[0]  # grid position
            drum_voice_class = drum_event[1]  # drum_voice_class in range(n_voices)

            # Grab the first note for each instrument group
            pitch = list(self.__drum_mapping.values())[drum_voice_class][0]
            velocity = self.__hvo[grid_pos, drum_voice_class + n_voices]  # Velocity of the drum event
            utiming_ratio = self.__hvo[  # exact timing of the drum event (rel. to grid)
                grid_pos, drum_voice_class + 2 * n_voices]

            utiming = 0
            if utiming_ratio < 0:
                # if utiming comes left of grid, figure out the grid resolution left of the grid line
                if grid_pos > 0:
                    utiming = (self.grid_lines[grid_pos] - self.grid_lines[grid_pos - 1]) * \
                              utiming_ratio
                else:
                    utiming = 0  # if utiming comes left of beginning,  snap it to the very first grid (loc[0]=0)
            elif utiming_ratio > 0:
                if grid_pos < (self.total_number_of_steps - 2):
                    utiming = (self.grid_lines[grid_pos + 1] -
                               self.grid_lines[grid_pos]) * utiming_ratio
                else:
                    utiming = (self.grid_lines[grid_pos] -
                               self.grid_lines[grid_pos - 1]) * utiming_ratio
                    # if utiming_ratio comes right of the last grid line, use the previous grid resolution for finding
                    # the utiming value in ms

            start_time = self.grid_lines[grid_pos] + utiming  # starting time of note in sec

            end_time = start_time + note_duration  # ending time of note in sec

            ns.notes.add(pitch=pitch, start_time=start_time.item(), end_time=end_time.item(),
                         is_drum=True, instrument=midi_track_n, velocity=int(velocity.item() * 127))

        ns.total_time = self.total_len

        for tempo in self.tempos:
            ns.tempos.add(
                time=self.grid_lines[tempo.time_step],
                qpm=tempo.qpm
            )

        for time_sig in self.time_signatures:
            ns.time_signatures.add(
                time=self.grid_lines[time_sig.time_step],
                numerator=time_sig.numerator,
                denominator=time_sig.denominator
            )

        return ns

    def save_hvo_to_midi(self, filename="misc/temp.mid", midi_track_n=9):
        """
            Exports to a  midi file

            @param filename:            filename/path for saving the midi
            @param midi_track_n:        midi track for

            @return pm:                 the pretty_midi object
        """
        convertible = all([self.is_hvo_score_available(print_missing=True),
                           self.is_tempos_available(print_missing=True),
                           self.is_time_signatures_available(print_missing=True)])
        if not convertible:
            warnings.warn("Above fields need to be provided so as to convert the hvo_sequence into a note_sequence")
            return None

        ns = self.to_note_sequence(midi_track_n=midi_track_n)
        pm = note_seq.note_sequence_to_pretty_midi(ns)
        pm.write(filename)
        return pm

    #   --------------------------------------------------------------
    #   Utilities to Synthesize the hvo score
    #   --------------------------------------------------------------

    def synthesize(self, sr=44100, sf_path="../hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"):
        """
        Synthesizes the hvo_sequence to audio using a provided sound font
        @param sr:                          sample rate
        @param sf_path:                     path to the soundfont samples
        @return:                            synthesized audio sequence
        """
        synthesizable = all([self.is_hvo_score_available(print_missing=True),
                             self.is_tempos_available(print_missing=True),
                             self.is_time_signatures_available(print_missing=True)])
        if not synthesizable:
            warnings.warn("Above fields need to be provided so as to synthesize the sequence")
            return None

        ns = self.to_note_sequence(midi_track_n=9)
        pm = note_seq.note_sequence_to_pretty_midi(ns)
        audio = pm.fluidsynth(fs=sr, sf2_path=sf_path)
        return audio

    def save_audio(self, filename="misc/temp.wav", sr=44100,
                   sf_path="../hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"):
        """
        Synthesizes and saves the hvo_sequence to audio using a provided sound font
        @param filename:                    filename/path used for saving the audio
        @param sr:                          sample rate
        @param sf_path:                     path to the soundfont samples
        @return:                            synthesized audio sequence
        """
        synthesizable = all([self.is_hvo_score_available(print_missing=True),
                             self.is_tempos_available(print_missing=True),
                             self.is_time_signatures_available(print_missing=True)])
        if not synthesizable:
            warnings.warn("Above fields need to be provided so as to synthesize the sequence")
            return None

        ns = self.to_note_sequence(midi_track_n=9)
        pm = note_seq.note_sequence_to_pretty_midi(ns)
        audio = pm.fluidsynth(sf2_path=sf_path, fs=sr)
        sf.write(filename, audio, sr, 'PCM_24')
        return audio

    #   --------------------------------------------------------------
    #   Utilities to plot the score
    #   --------------------------------------------------------------

    def to_html_plot(self, filename="misc/temp.html", show_figure=False,
                     show_tempo=True, tempo_font_size="8pt",
                     show_time_signature=True, time_signature_font_size="8pt",
                     minor_grid_color="black", minor_line_width=0.1,
                     major_grid_color="blue", major_line_width=0.5,
                     downbeat_color="blue", downbeat_line_width=2,
                     width=800, height=400):
        """
        Creates a bokeh plot of the hvo sequence
        @param filename:                    path to save the html plot
        @param show_figure:                 If True, opens the plot as soon as it's generated
        @return:                            html_figure object generated by bokeh
        """
        plottable = all([self.is_hvo_score_available(print_missing=True),
                         self.is_tempos_available(print_missing=True),
                         self.is_time_signatures_available(print_missing=True)])

        if not plottable:
            warnings.warn("Above fields need to be provided so as to synthesize the sequence")
            return None

        ns = self.to_note_sequence(midi_track_n=9)
        # Create the initial piano roll
        _html_fig = note_seq.plot_sequence(ns, show_figure=False)
        _html_fig.title.text = filename.split("/")[-1]  # add title

        # Add y-labels corresponding to instrument names rather than midi note ("kick", "snare", ...)
        unique_pitches = set([note.pitch for note in ns.notes])

        # Find corresponding drum tags
        drum_tags = []
        for p in unique_pitches:
            _, tag, _ = find_pitch_and_tag(p, self.__drum_mapping)
            drum_tags.append(tag)

        _html_fig.xgrid.grid_line_color = None
        _html_fig.ygrid.grid_line_color = None

        _html_fig.yaxis.ticker = list(unique_pitches)
        _html_fig.yaxis.major_label_overrides = dict(zip(unique_pitches, drum_tags))

        """
        ax2 = LinearAxis(x_range_name="foo", axis_label="blue circles")
        ax2.axis_label_text_color = "navy"
        _html_fig.add_layout(ax2, 'left')"""

        # Add beat and beat_division grid lines
        major_grid_lines, minor_grid_lines = self.major_and_minor_grid_lines

        grid_lines = self.grid_lines

        minor_grid_ = []
        for t in minor_grid_lines:
            minor_grid_.append(Span(location=t, dimension='height',
                                    line_color=minor_grid_color, line_width=minor_line_width))
            _html_fig.add_layout(minor_grid_[-1])

        major_grid_ = []
        for t in major_grid_lines:
            major_grid_.append(Span(location=t, dimension='height',
                                    line_color=major_grid_color, line_width=major_line_width))
            _html_fig.add_layout(major_grid_[-1])

        downbeat_grid_ = []
        for t in self.starting_measure_positions:
            downbeat_grid_.append(Span(location=t, dimension='height',
                                       line_color=downbeat_color, line_width=downbeat_line_width))
            _html_fig.add_layout(downbeat_grid_[-1])

        if show_tempo:
            my_label = []
            tempo_lower_b = self.tempo_consistent_segment_lower_bounds
            for ix, tempo in enumerate(self.tempos):
                my_label.append(Label(x=grid_lines[tempo_lower_b[ix]], y=list(unique_pitches)[-1] + 2,
                                      text="qpm {:.1f}".format(tempo.qpm)))
                my_label[-1].text_font_size = tempo_font_size
                my_label[-1].angle = 1.57
                _html_fig.add_layout(my_label[-1])

        if show_time_signature:
            my_label2 = []
            time_signature_lower_b = self.time_signature_consistent_segment_lower_bounds
            for ix, ts in enumerate(self.time_signatures):
                my_label2.append(Label(x=grid_lines[time_signature_lower_b[ix]], y=list(unique_pitches)[-1] + 0.5,
                                       text="{}/{}".format(ts.numerator, ts.denominator)))
                my_label2[-1].text_font_size = time_signature_font_size
                my_label2[-1].angle = 1.57
                _html_fig.add_layout(my_label2[-1])

        _html_fig.width = width
        _html_fig.height = height

        # Plot the figure if requested
        if show_figure:
            show(_html_fig)

        # Save the plot
        output_file(filename)  # Set name used for saving the figure
        save(_html_fig)  # Save to file

        return _html_fig

    #   --------------------------------------------------------------
    #   Utilities to compute, plot and save Spectrograms
    #   --------------------------------------------------------------

    def stft(self, sr=44100, sf_path="../hvo_sequence/soundfonts/Standard_Drum_Kit.sf2", n_fft=2048, hop_length=128,
             win_length=1024, window='hamming', plot=False, plot_filename="misc/temp_spec.png", plot_title="STFT", 
             width=800, height=400, font_size=12, colorbar=False ):
        """
        Computes the Short-time Fourier transform.
        @param sr:                          sample rate of the audio file from which the STFT is computed
        @param sf_path:                     path to the soundfont samples
        @param n_fft:                       length of the windowed signal after padding to closest power of 2
        @param hop_length:                  number of samples between successive STFT frames
        @param win_length:                  window length in samples. must be equal or smaller than n_fft
        @param window:                      window type specification (see scipy.signal.get_window) or function
        @param plot                         if True, plots and saves plot 
        @param plot_filename:               filename for saved figure
        @param plot_title:                  plot title
        @param width:                       figure width in pixels
        @param height:                      figure height in pixels
        @param font_size:                   font size in pt
        @param colorbar:                    if True, display colorbar
        @return:                            STFT ndarray
        """

        # Check inputs
        if not win_length <= n_fft:
            warnings.warn("Window size must be equal or smaller than FFT size.")
            return None

        if not hop_length > 0:
            warnings.warn("Hop size must be greater than 0.")
            return None

        # Get audio signal
        y = self.save_audio(sr=sr, sf_path=sf_path)

        # Get STFT
        sy = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        stft = np.abs(sy)

        if plot:
            # Plot STFT
            # Plot params
            plt.rcParams['font.size'] = font_size

            px = 1 / plt.rcParams['figure.dpi']  # pixel to inch conversion factor
            [width_i, height_i] = [width * px, height * px]  # width and height in inches

            plt.rcParams.update({'figure.autolayout': True})  # figure layout
            plt.tight_layout()

            # Plot spectogram and save
            fig, ax = plt.subplots(figsize=(width_i, height_i))
            ax.set_title(plot_title)


            spec = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), y_axis='log', x_axis='time', ax=ax)

            if colorbar:
                fig.colorbar(spec, ax=ax, format="%+2.0f dB")

            fig.savefig(plot_filename)


        return stft

    def mel_spectrogram(self, sr=44100, sf_path="../hvo_sequence/soundfonts/Standard_Drum_Kit.sf2", n_fft=2048,
                        hop_length=128, win_length=1024, window='hamming', n_mels=24, fmin=0, fmax=22050, plot=False,
                        plot_filename="misc/temp_mel_spec.png", plot_title="'Mel-frequency spectrogram'", width=800, 
                        height=400, font_size=12, colorbar=False):
        
        """
        Computes mel spectrogram.
        @param sr:                          sample rate of the audio file from which the STFT is computed
        @param sf_path:                     path to the soundfont samples
        @param n_fft:                       length of the windowed signal after padding to closest power of 2
        @param hop_length:                  number of samples between successive STFT frames
        @param win_length:                  window length in samples. must be equal or smaller than n_fft
        @param window:                      window type specification (see scipy.signal.get_window) or function
        @param n_mels:                      number of mel bands
        @param fmin:                        lowest frequency in Hz
        @param fmax:                        highest frequency in Hz
        @param plot_filename:               filename for saved figure
        @param plot_title:                  plot title
        @param width:                       figure width in pixels
        @param height:                      figure height in pixels
        @param font_size:                   font size in pt
        @param colorbar:                    if True, display colorbar
        @return:                            mel spectrogram ndarray
        """

        # Check inputs
        if not win_length <= n_fft:
            warnings.warn("Window size must be equal or smaller than FFT size.")
            return None

        if not hop_length > 0:
            warnings.warn("Hop size must be greater than 0.")
            return None

        if not n_mels > 0:
            warnings.warn("Number of mel bands must be greater than 0.")
            return None

        if not fmin >= 0 or not fmax > 0:
            warnings.warn("Frequency must be greater than 0.")
            return None

        # Get audio signal
        y = self.save_audio(sr=sr, sf_path=sf_path)

        # Get mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                                  window=window, n_mels=n_mels, fmin=fmin, fmax=fmax)
        
        if plot:
            # Plot mel spectrogram
            # Plot specs
            plt.rcParams['font.size'] = font_size

            px = 1 / plt.rcParams['figure.dpi']  # pixel to inch conversion factor
            [width_i, height_i] = [width * px, height * px]  # width and height in inches

            plt.rcParams.update({'figure.autolayout': True})  # figure layout
            plt.tight_layout()

            # Plot spectogram and save
            fig, ax = plt.subplots(figsize=(width_i, height_i))
            ax.set_title(plot_title)

            spec = librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), y_axis='mel', x_axis='time', ax=ax)

            if colorbar:
                fig.colorbar(spec, ax=ax, format="%+2.0f dB")

            fig.savefig(plot_filename)

        return mel_spec
