import numpy as np
import note_seq
from note_seq.protobuf import music_pb2
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from bokeh.plotting import output_file, show, save
from bokeh.models import Span, Label
import warnings
from scipy import stats
from scipy.signal import find_peaks
import math

from hvo_sequence.utils import is_power_of_two, find_pitch_and_tag, cosine_similarity, cosine_distance
from hvo_sequence.utils import _weight_groove, _reduce_part, fuzzy_Hamming_distance
from hvo_sequence.utils import _get_kick_and_snare_syncopations, get_monophonic_syncopation
from hvo_sequence.utils import get_weak_to_strong_ratio, _getmicrotiming_event_profile_1bar

from hvo_sequence.custom_dtypes import Tempo, Time_Signature
from hvo_sequence.drum_mappings import Groove_Toolbox_5Part_keymap, Groove_Toolbox_3Part_keymap

from hvo_sequence.metrical_profiles import WITEK_SYNCOPATION_METRICAL_PROFILE_4_4_16th_NOTE
from hvo_sequence.metrical_profiles import Longuet_Higgins_METRICAL_PROFILE_4_4_16th_NOTE
from hvo_sequence.metrical_profiles import RHYTHM_SALIENCE_PROFILE_4_4_16th_NOTE

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

        self.__version = "0.2.0"

        self.__time_signatures = list()
        self.__tempos = list()
        self.__drum_mapping = None
        self.__hvo = None

        self.__force_vo_reset = True

        # Use property setters to initiate properties (DON"T ASSIGN ABOVE so that the correct datatype is checked)
        if drum_mapping:
            self.drum_mapping = drum_mapping

    #   ----------------------------------------------------------------------
    #   Essential properties (which require getters and setters)
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

        if self.__hvo is not None:
            # Return 'synced' hvo array, meaning for hits == 0, velocities and offsets are returned as 0
            # without modifying the actual values stored internally
            n_voices = int(self.__hvo.shape[1] / 3)

            hits_tmp = self.__hvo[:, :n_voices]
            velocities_tmp = self.__hvo[:, n_voices:2*n_voices]
            offsets_tmp = self.__hvo[:, 2*n_voices:]

            return np.concatenate((hits_tmp, velocities_tmp * hits_tmp, offsets_tmp * hits_tmp), axis=1)

        return None

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
    #   Utilities to modify hvo sequence
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
                    warnings.warn("Forcing velocity and offset reset for voice {}." \
                                  " Deactivate setting force_vo_reset property to False".format(_voice_idx))

            # reset voice
            if _reset_hits:
                self.__hvo[:, h_idx] = np.zeros(n_frames)
            if _reset_velocity:
                self.__hvo[:, v_idx] = np.zeros(n_frames)
            if _reset_offsets:
                self.__hvo[:, o_idx] = np.zeros(n_frames)

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
        flat_hvo = np.concatenate((_hits, _velocities, _offsets), axis=1) if get_velocities else np.concatenate(
            (_hits, _offsets), axis=1)
        return flat_hvo

    @property
    def hits(self):
        """ ndarray of dimensions m,n where m is the number of time steps and n the number of drums in the current
            drum mapping (e.g. 9 for the reduced mapping). The values of the array are 1 or 0, indicating whether a
            hit occurs at that time step for that drum (1) or not (0).
        """
        calculable = all([self.is_hvo_score_available(),
                          self.is_drum_mapping_available()])
        return self.__hvo[:, :self.number_of_voices] if calculable else None

    def __is_hit_array_valid(self, hit_array):
        """
        checks to see if hit array is binary and second dimension matches number of voices
        :param hit_array:
        :return:
        """
        valid = True
        if len(self.__hvo[:, :self.number_of_voices]) != len(hit_array):
            valid = False
            print("hit array length mismatch")
        if not np.all(np.logical_or(np.asarray(hit_array) == 0, np.asarray(hit_array) == 1)):
            valid = False
            print("invalid hit values in array, they must be 0 or 1")
        return valid

    @hits.setter
    def hits(self, hit_array):
        assert self.__is_hit_array_valid(hit_array), "Hit array is invalid! Must be binary and second dimension " \
                                                     "must match the number of voices in drum_mapping"
        if self.hvo is None:  # if no hvo score is available set velocities to one at hit and offsets to zero
            self.hvo = np.concatenate((hit_array, hit_array, np.zeros_like(hit_array)), axis=1)
        else:
            self.hvo[:, :self.number_of_voices] = hit_array

    @property
    def velocities(self):
        """ ndarray of dimensions m,n where m is the number of time steps and n the number of drums in the current
            drum mapping (e.g. 9 for the reduced mapping). The values of the array are continuous floating point
            numbers from 0 to 1 indicating the velocity.
        """
        calculable = all([self.is_hvo_score_available(),
                          self.is_drum_mapping_available()])
        if not calculable:
            print("can't get velocities as there is no hvo score previously provided")
        else:
            # Note that the value returned is the internal one - even if a hit is 0 at an index,
            # velocity at that same index might not be 0
            return self.__hvo[:, self.number_of_voices: 2 * self.number_of_voices]

    def __is_vel_array_valid(self, vel_array):
        valid = True
        if vel_array.shape[1] != self.number_of_voices:
            warnings.warn('Second dimension of vel_array must match the number of keys in drum_mapping')
            return False
        if np.min(vel_array) < 0 or np.max(vel_array) > 1:
            warnings.warn("Velocity values must be between 0 and 1")
            return False
        if self.is_hvo_score_available():
            if vel_array.shape[0] != len(self.hvo[:, self.number_of_voices: 2 * self.number_of_voices]):
                warnings.warn("velocity array length mismatch")
                return False
        return True

    @velocities.setter
    def velocities(self, vel_array):
        assert self.__is_vel_array_valid(vel_array), "velocity array is incorrect! either time step mismatch " \
                                                     "or second number of voices mismatch or values outside [0, 1]"

        if self.hvo is None:  # if hvo empty, set correspong hits to one and offsets to zero
            self.hvo = np.concatenate((np.where(vel_array > 0, 1, 0),
                                       vel_array,
                                       np.zeros_like(vel_array)), axis=1)
        else:
            self.hvo[:, self.number_of_voices: 2 * self.number_of_voices] = vel_array

    @property
    def offsets(self):
        """ ndarray of dimensions m,n where m is the number of time steps and n the number of drums in the current
            drum mapping (e.g. 9 for the reduced mapping). The values of the array are continuous floating point
            numbers from -0.5 to 0.5 indicating the offset respect to the beat grid line that each hit is on.
        """
        calculable = all([self.is_hvo_score_available(),
                          self.is_drum_mapping_available()])
        if not calculable:
            print("can't get offsets/utimings as there is no hvo score previously provided")
            return None
        else:
            # Note that the value returned is the internal one - even if a hit is 0 at an index,
            # offset at that same index might not be 0
            return self.hvo[:, 2 * self.number_of_voices:]

    def __is_offset_array_valid(self, offset_array):
        if self.is_hvo_score_available() is False:
            warnings.warn("hvo field is empty: Can't set offsets without hvo field")
            return False

        if offset_array.shape[1] != self.number_of_voices:
            warnings.warn("offset array length mismatch")
            return False

        if offset_array.mean() < -0.5 or offset_array.mean() > 0.5:
            warnings.warn("invalid offset values in array, they must be between -0.5 and 0.5")
            return False

        return True

    @offsets.setter
    def offsets(self, offset_array):
        calculable = all([self.is_hvo_score_available(),
                          self.is_drum_mapping_available()])
        if not calculable:
            print("can't set offsets as there is no hvo score previously provided")
        else:
            if self.__is_offset_array_valid(offset_array):
                self.hvo[:, 2 * self.number_of_voices:] = offset_array

    #   ----------------------------------------------------------------------
    #   Utility methods to check whether required properties are
    #       available for carrying out a request
    #
    #   Assuming that the local variables haven't been modified directly,
    #   No need to check the validity of data types if they are available
    #       as this is already done in the property.setters
    #   ----------------------------------------------------------------------

    def is_time_signatures_available(self):
        # Checks whether time_signatures are already specified and necessary fields are filled

        if len(self.time_signatures) == 0:
            warnings.warn("Time Signature missing")
            return False

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

    def is_tempos_available(self):
        # Checks whether tempos are already specified and necessary fields are filled
        tempos_ready_to_use = list()
        if self.tempos is not None:
            for tempo in self.tempos:
                tempos_ready_to_use.append(tempo.is_ready_to_use)
        else:
            warnings.warn("No tempos specified")
            return False

        if not all(tempos_ready_to_use):
            for ix, ready_status in enumerate(tempos_ready_to_use):
                if ready_status is not True:
                    print("There are missing fields in Tempo {}: {}".format(ix, self.tempos[ix]))
            return False
        else:
            return True

    def is_drum_mapping_available(self):
        # Checks whether drum_mapping is already specified
        if self.is_drum_mapping_available is None:
            warnings.warn("Drum mapping is missing")
            return False
        else:
            return True

    def is_hvo_score_available(self):
        # Checks whether hvo score array is already specified
        if self.is_drum_mapping_available is None or self.hvo is None:
            print("Either HVO score or drum_mapping are missing")
            return False
        else:
            return True

    def is_ready_for_use(self):
        state = all([
            self.is_hvo_score_available(),
            self.is_time_signatures_available(),
            self.is_hvo_score_available(),
            self.is_drum_mapping_available()
        ])
        return state

    #   -------------------------------------------------------------
    #   Method to get hvo in a flexible way
    #   -------------------------------------------------------------

    def get(self, hvo_str, offsets_in_ms=False, use_NaN_for_non_hits=False):
        """
        Flexible method to get hits, velocities and offsets in the desired order, or zero arrays with the same
        dimensions as one of those vectors. The velocities and offsets are synced to the hits, so whenever a hit is 0,
        velocities and offsets will be 0 as well.

        Parameters
        ----------
        hvo_str: str
            String formed with the characters 'h', 'v', 'o' and '0' in any order. It's not necessary to use all of the
            characters and they can be repeated. E.g. 'ov', will return the offsets and velocities, 'h0h' will return
            the hits, a 0-vector and the hits again, again and '000' will return a hvo-sized 0 matrix.

        offsets_in_ms: bool
            If true, the queried offsets will be provided in ms deviations from grid, otherwise, will be
            provided in terms of ratios
        """
        assert self.is_hvo_score_available(), "No hvo score available, update this field"

        assert isinstance(hvo_str, str), 'hvo_str must be a string'
        hvo_str = hvo_str.lower()
        hvo_arr = []

        # Get h, v, o
        h = self.hvo[:, :self.number_of_voices]
        v = self.hvo[:, self.number_of_voices:self.number_of_voices*2]
        o = self.get_offsets_in_ms() if offsets_in_ms else self.hvo[:, self.number_of_voices*2:]
        zero = np.zeros_like(h)

        # replace vels and offsets with no associated hit to np.nan if use_NaN_for_non_hits set to True
        if use_NaN_for_non_hits is not False:
            v[h == 0] = -1000000
            v = np.where(v == -1000000, np.nan, v)
            o[h == 0] = -1000000
            o = np.where(o == -1000000, np.nan, o)

        # Concatenate parts
        for c in hvo_str:
            assert (c == 'h' or c == 'v' or c == 'o' or c == '0'), 'hvo_str not valid'
            concat_arr = zero if c == '0' else h if c == 'h' else v if c == 'v' else o
            hvo_arr = concat_arr if len(hvo_arr) == 0 else np.concatenate((hvo_arr, concat_arr), axis=1)

        return hvo_arr

    def get_with_different_drum_mapping(self, hvo_str, tgt_drum_mapping,
                                        offsets_in_ms=False, use_NaN_for_non_hits=False):
        """
        similar to self.get() except that it maps the extracted hvo sequence to a provided target mapping

        if multiple velocities/offsets are to be grouped together, only the position of the loudest velocity is used

        :param hvo_str: str
            String formed with the characters 'h', 'v', 'o' and '0' in any order. It's not necessary to use all of the
            characters and they can be repeated. E.g. 'ov', will return the offsets and velocities, 'h0h' will return
            the hits, a 0-vector and the hits again, again and '000' will return a hvo-sized 0 matrix.
        :param tgt_drum_mapping:        Alternative mapping to use
        :param offsets_in_ms:           True/False, specifies if offsets should be in ms
        :param use_NaN_for_non_hits:    True/False, specifies if np.nan should be used instead of 0 wherever a hit is
                                        missing
        :return:
            the sequence associated with hvo_str mapped to a target drum map

        """

        def get_tgt_map_index_for_src_map(src_map, tgt_map):
            """
            Finds the corresponding voice group index for
            :param src_map:   a drum_mapping dictionary
            :param tgt_map:     a drum_mapping dictionary
            :return: list of indices in src to be grouped together. Each element in returned list is the corresponding voice group to be used in
                     tgt_map for each of the voice groups in the src_map

                    Example:
                    if src_map = ROLAND_REDUCED_MAPPING and tgt_map = Groove_Toolbox_5Part_keymap
                    the return will be [[0], [1], [2, 8], [3, 7], [4, 5, 6]]
                    this means that kicks are to be mapped to kick
                                    snares are to be mapped to snares
                                    c_hat and rides are to be mapped to the same group (closed)
                                    o_hat and crash are to be mapped to the same group (open)
                                    low. mid. hi Toms are to be mapped to the same group (toms)

            """
            # Find the corresponding index in tgt mapping for each element in src map
            src_ix_to_tgt_ix_map = np.array([])
            for src_voice_ix, src_voice_midi_list in enumerate(src_map.values()):
                corresponding_tgt_indices = []
                for tgt_voice_ix, tgt_voice_midi_list in enumerate(tgt_map.values()):
                    if src_voice_midi_list[0] in tgt_voice_midi_list:
                        corresponding_tgt_indices.append(tgt_voice_ix)
                src_ix_to_tgt_ix_map = np.append(src_ix_to_tgt_ix_map,
                                                 max(corresponding_tgt_indices, key=corresponding_tgt_indices.count))

            n_voices_tgt = len(tgt_map.keys())

            grouped_voices = [np.argwhere(src_ix_to_tgt_ix_map==tgt_ix).reshape(-1)
                               for tgt_ix in range(n_voices_tgt)]

            return grouped_voices

        # Find src indices in src_map corresponding to tgt
        grouped_voices = get_tgt_map_index_for_src_map(self.drum_mapping, tgt_drum_mapping)

        # Get non-reduced score with the existing mapping
        hvo = self.get("hvo", offsets_in_ms, use_NaN_for_non_hits=False)
        h_src, v_src, o_src = np.split(hvo, 3, axis=1)

        # Create empty placeholders for hvo and zero
        h_tgt = np.zeros((h_src.shape[0], len(tgt_drum_mapping.keys())))
        if "v" in hvo_str or "o" in hvo_str:
            v_tgt = np.zeros((h_src.shape[0], len(tgt_drum_mapping.keys())))
            o_tgt = np.zeros((h_src.shape[0], len(tgt_drum_mapping.keys())))
        if "0" in hvo_str:
            zero_tgt = np.zeros((h_src.shape[0], len(tgt_drum_mapping.keys())))

        # use the groups of indices in grouped_voices to map the src sequences to tgt sequence
        for ix, voice_group in enumerate(grouped_voices):
            if len(voice_group) > 0:
                h_tgt[:, ix] = np.any(h_src[:, voice_group], axis=1)
                if "v" in hvo_str or "o" in hvo_str:
                    v_max_indices = np.nanargmax(v_src[:, voice_group], axis=1) # use loudest velocity
                    v_tgt[:, ix] = v_src[:, voice_group][range(len(v_max_indices)), v_max_indices]
                    o_tgt[:, ix] = o_src[:, voice_group][range(len(v_max_indices)), v_max_indices]


        # replace vels and offsets with no associated hit to np.nan if use_NaN_for_non_hits set to True
        if use_NaN_for_non_hits is not False and ("v" in hvo_str or "o" in hvo_str):
            v_tgt[h_tgt == 0] = -1000000
            v_tgt = np.where(v_tgt == -1000000, np.nan, v_tgt)
            o_tgt[h_tgt == 0] = -1000000
            o_tgt = np.where(o_tgt == -1000000, np.nan, o_tgt)

        # Concatenate parts according to hvo_str
        hvo_arr = []
        for c in hvo_str:
            assert (c == 'h' or c == 'v' or c == 'o' or c == '0'), 'hvo_str not valid'
            concat_arr = zero_tgt if c == '0' else h_tgt if c == 'h' else v_tgt if c == 'v' else o_tgt
            hvo_arr = concat_arr if len(hvo_arr) == 0 else np.concatenate((hvo_arr, concat_arr), axis=1)

        return hvo_arr

    def get_offsets_in_ms(self, use_NaN_for_non_hits=False):
        """
        Gets the offset portion of hvo and converts the values to ms using the associated grid

        :return:    the offsets in hvo tensor in ms
        """
        convertible = all([self.is_tempos_available(),
                           self.is_time_signatures_available()])

        if not convertible:
            warnings.warn("Above fields need to be provided so as to get the offsets in ms")
            return None

        # get the number of allowed drum voices
        n_voices = len(self.__drum_mapping.keys())

        # find nonzero hits tensor of [[position, drum_voice]]
        pos_instrument_tensors = np.transpose(np.nonzero(self.__hvo[:, :n_voices]))

        # create an empty offsets array
        offsets = np.zeros_like(self.__hvo[:, :n_voices])

        # Initialize the offsets to np.nan if use_NaN_for_non_hits is True
        if use_NaN_for_non_hits is not False:
            offsets[:] = np.nan

        # Add notes to the NoteSequence object
        for drum_event in pos_instrument_tensors:  # drum_event -> [grid_position, drum_voice_class]
            grid_pos = drum_event[0]  # grid position
            drum_voice_class = drum_event[1]  # drum_voice_class in range(n_voices)

            # Grab the first note for each instrument group
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

            offsets[drum_event[0], drum_event[1]] = utiming*1000

        return offsets

    def get_bar_beat_hvo(self, hvo_str="hvo"):
        """
        returns the score organized as an array of shape (bar_idx, beat_idx, step_in_beat, len(hvo_str)*n_voices)

        Note: If the number of bars is not an integer multiple, the sequence will be padded with -1's to denote the
        padding

        :param hvo_str:     String formed with the characters 'h', 'v' and 'o' in any order. It's not necessary
                            to use all of the characters and they can be repeated. E.g. 'ov' or 'hvoh'

        :return:            Score (possibly padded w/ -1s) shaped as
                            (bar_idx, beat_idx, step_in_beat, len(hvo_str)*n_voices)
        """

        assert len(self.time_signatures) == 1, "This feature is not currently available for " \
                                               "hvo scores with time signature change"

        n_bars = int(np.ceil(sum(self.n_bars_per_segments)))
        beats_per_bar = self.time_signatures[0].numerator
        steps_per_beat_per_bar = self.steps_per_beat_per_segments[0]

        total_steps = int(n_bars)*steps_per_beat_per_bar*beats_per_bar

        hvo_arr = self.get(hvo_str)

        dim_at_step = hvo_arr.shape[-1]

        # Create an empty padded sequence
        reshaped_hvo = np.zeros((total_steps, dim_at_step))-1

        # fill sequence with existing score
        reshaped_hvo[:hvo_arr.shape[0], :hvo_arr.shape[1]] = hvo_arr

        # Reshape to (bar_index, beat_index, step_index, len(hvo_str)*n_voices)
        reshaped_hvo = reshaped_hvo.reshape((n_bars, beats_per_bar, steps_per_beat_per_bar, dim_at_step))

        return reshaped_hvo


    #   ----------------------------------------------------------------------
    #            Calculated properties
    #   Useful properties calculated from ESSENTIAL class variables
    #   EACH SEGMENT MEANS A PART THAT TEMPO AND TIME SIGNATURE IS CONSTANT
    #   ----------------------------------------------------------------------

    @property
    def number_of_voices(self):
            return None if self.is_drum_mapping_available() is False else int(self.hvo.shape[1] / 3)

    @property
    def tempo_consistent_segment_boundaries(self):
        #   Returns time boundaries within which the tempo is constant
        #   upper bound (infinite) is always at 100000 seconds
        #   Example: tempo change at 1.5 seconds
        #            method returns --> [0, 1.5, 100000] --> meaning that tempo is
        #                               constant between 0≤t<1.5, 1.5≤t,100000
        #   If no tempo changes in the track (i.e. consistent tempo across all times
        #            method returns --> [0, 1000000]
        calculable = self.is_tempos_available()
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
        calculable = self.is_time_signatures_available()
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
        calculable = all([self.is_time_signatures_available(),
                          self.is_tempos_available()])
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

    @property
    def tempos_and_time_signatures_per_segments(self):
        # Returns two lists: 1. lists of tempos per segment
        #                     2. lists of time signature for each segment
        # Segments are defined as parts of the score where the tempo and time signature don't change
        calculable = all([self.is_time_signatures_available(),
                          self.is_tempos_available()])
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
        calculable = all([self.is_time_signatures_available(),
                          self.is_tempos_available()])
        if not calculable:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            segment_bounds = self.segment_boundaries
            return len(segment_bounds) - 1

    @property
    def beat_durations_per_segments(self):

        # Calculates the duration of each beat in seconds if time signature and qpm are available
        calculable = all([self.is_tempos_available(),
                          self.is_time_signatures_available()])
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

        if segment_lower_bounds is not None and self.is_hvo_score_available() is not None:
            return list(np.array(segment_upper_bounds) - np.array(segment_lower_bounds))
        else:
            warnings.warn("Tempo or Time Signature missing")
            return None

    @property
    def n_beats_per_segments(self):
        # Calculate the number of beats in each tempo and time signature consistent segment of score/sequence

        calculable = all([self.is_hvo_score_available(),
                          self.is_tempos_available(),
                          self.is_time_signatures_available()])

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

    @property
    def total_len(self):
        # Calculates the total length of score in seconds if hvo score, time signature and qpm are available
        calculable = all([self.is_hvo_score_available(),
                          self.is_tempos_available(),
                          self.is_time_signatures_available()])
        if calculable:
            return self.grid_lines[-1] + 0.5 * (self.grid_lines[-1] - self.grid_lines[-2])
        else:
            return None

    @property
    def total_number_of_steps(self):
        # Calculates the total number of steps in the score/sequence
        calculable = self.is_hvo_score_available()
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

        calculable = all([self.is_tempos_available(),
                          self.is_time_signatures_available()])

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

        calculable = all([self.is_hvo_score_available(),
                          self.is_tempos_available(),
                          self.is_time_signatures_available()])

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

        calculable = all([self.is_hvo_score_available(),
                          self.is_tempos_available(),
                          self.is_time_signatures_available()])

        if not calculable:
            warnings.warn("Above fields are required for calculating major/minor grid line positions")
            return None, None

        grids_with_types = self.grid_lines_with_types
        return grids_with_types["major_grid_lines"], grids_with_types["minor_grid_lines"]

    @property
    def downbeat_indices(self):
        # Returns the indices of the grid_lines where a downbeat occurs.
        calculable = all([self.is_hvo_score_available(),
                          self.is_tempos_available(),
                          self.is_time_signatures_available()])

        if not calculable:
            warnings.warn("Above fields are required for calculating downbeat grid line positions")
            return None, None

        grids_with_types = self.grid_lines_with_types
        return grids_with_types["downbeat_grid_line_indices"]

    @property
    def downbeat_positions(self):
        # Returns the indices of the grid_lines where a downbeat occurs.
        calculable = all([self.is_hvo_score_available(),
                          self.is_tempos_available(),
                          self.is_time_signatures_available()])

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
        assert all([self.is_tempos_available(), self.is_time_signatures_available(), self.is_hvo_score_available()]), \
            "Can't calculate grid lines as either no tempos, no time signature or no hvo score is specified"

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
    #   Utility methods to get STEP specific information
    #   ----------------------------------------------------------------------

    def tempo_segment_index_at_step(self, step_ix):
        # gets the index of the tempo segment in which the step is located
        if self.is_tempos_available() is None:
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
        if self.is_time_signatures_available() is None:
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
        calculable = all([self.is_time_signatures_available(),
                          self.is_tempos_available()])
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

    #   --------------------------------------------------------------
    #   Utilities to import/export/Convert different score formats such as
    #       1. NoteSequence, 2. HVO array, 3. Midi
    #   --------------------------------------------------------------

    def to_note_sequence(self, midi_track_n=9):
        """
        Exports the hvo_sequence to a note_sequence object

        @param midi_track_n:    the midi track channel used for the drum scores
        @return:
        """
        if self.is_ready_for_use() is False:
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

        if self.is_ready_for_use() is False:
            return None

        ns = self.to_note_sequence(midi_track_n=midi_track_n)
        pm = note_seq.note_sequence_to_pretty_midi(ns)
        pm.write(filename)
        return pm

    def convert_to_alternate_mapping(self, tgt_drum_mapping):

        if self.is_ready_for_use() is False:
            return None

        hvo_seq_tgt = HVO_Sequence(tgt_drum_mapping)

        # Copy the tempo and time signature fields to new hvo
        for tempo in self.tempos:
            hvo_seq_tgt.add_tempo(tempo.time_step, tempo.qpm)
        for ts in self.time_signatures:
            hvo_seq_tgt.add_time_signature(ts.time_step, ts.numerator, ts.denominator, ts.beat_division_factors)

        hvo_tgt = self.get_with_different_drum_mapping("hvo", tgt_drum_mapping=tgt_drum_mapping)
        hvo_seq_tgt.hvo = hvo_tgt

        return hvo_seq_tgt

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

        if self.is_ready_for_use() is False:
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

        if self.is_ready_for_use() is False:
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
                     save_figure=False,
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

        if self.is_ready_for_use() is False:
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
        if save_figure:
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

        if self.is_ready_for_use() is False:
            return None

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

        if self.is_ready_for_use() is False:
            return None

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

    #   -------------------------------------------------------------
    #   Extract Rhythmical and Microtiming Features for Evaluation
    #   -------------------------------------------------------------

    # ######################################################################
    #
    #           Rhythmic Features::Statistical Features Related
    #
    # ######################################################################

    def get_number_of_active_voices(self):
        """gets total number of active instruments in patter"""
        h = self.hits
        h = np.where(h == 0.0, np.nan, h)
        noi = 0
        for voice_ix in range(h.shape[1]):
            # check if voice part is empty
            if all(np.isnan(h[:, voice_ix])) is not True:
                noi += 1
        return noi

    def get_total_step_density(self):
        """ calculates the ratio of total steps in which there is at least one hit """
        hits = self.hits
        return np.clip(np.count_nonzero(hits, axis=1), 0, 1).sum()/hits.shape[0]

    def get_average_voice_density(self):
        """ average of number of instruments divided by total number of voices over all steps """
        hits = self.hits
        return np.count_nonzero(hits, axis=1).sum()/hits.size

    def get_hit_density_for_voice(self, voice_ix):
        return np.count_nonzero(self.hits[:, voice_ix])/ self.hvo.shape[0]

    def get_velocity_intensity_mean_stdev_for_voice(self, voice_ix):
        # Calculates mean and std of velocities for a single voice
        # first get all non-zero hits. then divide by number of hits
        if self.is_ready_for_use() is False:
            return None
        v = self.get("v", use_NaN_for_non_hits=True)[:, voice_ix]
        if all(np.isnan(v)) is True:
            return 0, 0
        else:
            return np.nanmean(v), np.nanstd(v)

    def get_offset_mean_stdev_for_voice(self, voice_ix, offsets_in_ms=False):
        # Calculates mean and std of offsets for a single voice
        # first get all non-zero hits. then divide by number of hits
        if self.is_ready_for_use() is False:
            return None
        o = self.get("o", offsets_in_ms=offsets_in_ms, use_NaN_for_non_hits=True)[:, voice_ix]
        if all(np.isnan(o)) is True:
            return 0, 0
        else:
            return np.nanmean(o), np.nanstd(o)

    def get_lowness_midness_hiness(self, low_mid_hi_drum_map=Groove_Toolbox_3Part_keymap):
        """
        "Share of the total density of patterns that belongs to each
        of the different instrument categories. Computed as the
        quotient between the densities per instrument category
        and the total density" [2]

        [2] Drum rhythm spaces: From polyphonic similarity to generative maps by Daniel Gomez Marin, 2020
        """
        lmh_hits = self.get_with_different_drum_mapping("h", tgt_drum_mapping=low_mid_hi_drum_map)
        total_hits = np.count_nonzero(self.hits)
        lowness = np.count_nonzero(lmh_hits[:, 0])/total_hits
        midness = np.count_nonzero(lmh_hits[:, 1])/total_hits
        hiness = np.count_nonzero(lmh_hits[:, 2])/total_hits
        return lowness, midness, hiness

    def get_velocity_score_symmetry(self):
        # Get total symmetry of pattern. Defined as the number of onsets that appear in the same positions in the first
        # and second halves of the pattern, divided by total number of onsets in the pattern.
        # symmetry is calculated using velocity section of hvo

        if self.is_ready_for_use() is False:
            return None

        v = self.get("v", use_NaN_for_non_hits=True)
        assert v.shape[0] % 2 == 0, "symmetry can't be calculated as the length of score needs to be a multiple of 2"

        # Find difference between splits
        part1, part2 = np.split(v, 2)
        diff = np.abs(part1 - part2)
        diff = diff[~np.isnan(diff)]    # Remove non hit locations (denoted with np.nan)
        # get symmetry level
        symmetry_level = (1 - diff)

        return np.nanmean(symmetry_level)

    #todo easily adaptable to alternative grids if implementation is changed
    def get_total_weak_to_strong_ratio(self):
        """
        returns the ratio of total weak onsets divided by all strong onsets
        strong onsets are onsets that occur on beat positions and weak onsets are the other ones
        """
        return get_weak_to_strong_ratio(self.get("v"))

    def get_polyphonic_velocity_mean_stdev(self):
        # Get average loudness for any single part or group of parts. Will return 1 for binary loop, otherwise calculate
        # based on velocity mode chosen (transform or regular)

        # first get all non-zero hits. then divide by number of hits
        if self.is_ready_for_use() is False:
            return None
        v = self.get("v", use_NaN_for_non_hits=True)
        if all(np.isnan(v.flatten())) is True:
            return 0, 0
        else:
            return np.nanmean(v), np.nanstd(v)

    def get_polyphonic_offset_mean_stdev(self, offsets_in_ms=False):
        # Get average loudness for any single part or group of parts. Will return 1 for binary loop, otherwise calculate
        # based on velocity mode chosen (transform or regular)

        # first get all non-zero hits. then divide by number of hits
        if self.is_ready_for_use() is False:
            return None
        o = self.get("o", offsets_in_ms=offsets_in_ms, use_NaN_for_non_hits=True)
        if all(np.isnan(o.flatten())) is True:
            return 0, 0
        else:
            return np.nanmean(o), np.nanstd(o)

    # ######################################################################
    #   Rhythmic Features::Syncopation from GrooveToolbox
    #
    #        The following code is mostly from the GrooveToolbox
    #              https://github.com/fredbru/GrooveToolbox
    #        Some additional functions have been implemented here
    #         to adapt hvo_sequence representation to the groove and
    #            utiming representations used in the GrooveToolbox
    #
    # Reference:    Yang, Li-Chia, and Alexander Lerch. "On the evaluation
    #               of generative models in music." Neural Computing
    #               and Applications 32.9 (2020): 4773-4784.
    # ######################################################################

    # todo : use get_monophonic_syncopation_for_voice() to get syncopation of tapped pattern

    def get_monophonic_syncopation_for_voice(self, voice_index):
        # Using Longuet-Higgins  and  Lee 1984 metric profile, get syncopation of 1 monophonic line.
        # Assumes it's a drum loop - loops round.
        # Normalized against maximum syncopation: syncopation score of pattern with all pulses of lowest metrical level
        # at maximum amplitude (=30 for 2 bar 4/4 loop)

        if self.is_ready_for_use() is True:
            assert len(self.time_signatures) == 1 and self.time_signatures[0].denominator == 4, \
                "currently Can't calculate syncopation for patterns with multiple time signatures and " \
                "time signature denominators other than 4"
        else:
            return None

        metrical_profile = Longuet_Higgins_METRICAL_PROFILE_4_4_16th_NOTE

        part = self.hits[:, voice_index]

        return get_monophonic_syncopation(part, metrical_profile)

    # todo: error of size mismatch here
    def get_combined_syncopation(self):
        # Calculate syncopation as summed across all kit parts.
        # Tested - working correctly (12/3/20)

        combined_syncopation = 0.0

        for i in range(self.number_of_voices):
            combined_syncopation += self.get_monophonic_syncopation_for_voice(voice_index=i)

        return combined_syncopation

    def get_witek_polyphonic_syncopation(self, low_mid_hi_drum_map = Groove_Toolbox_3Part_keymap):
        # Calculate syncopation using Witek syncopation distance - modelling syncopation between instruments
        # Works on semiquaver and quaver levels of syncopation
        # at maximum amplitude (=30 for 2 bar 4/4 loop)
        # todo: Normalize...?

        metrical_profile = WITEK_SYNCOPATION_METRICAL_PROFILE_4_4_16th_NOTE

        max_syncopation = 30.0

        # Get hits score reduced to low mid high groups
        lmh_hits = self.get_with_different_drum_mapping("h", tgt_drum_mapping=low_mid_hi_drum_map)
        low = lmh_hits[:, 0]
        mid = lmh_hits[:, 1]
        high = lmh_hits[:, 2]

        total_syncopation = 0

        for i in range(len(low)):
            kick_syncopation, snare_syncopation = _get_kick_and_snare_syncopations(low, mid, high, i, metrical_profile)
            total_syncopation += kick_syncopation * low[i]
            total_syncopation += snare_syncopation * mid[i]

        return total_syncopation / max_syncopation

    # todo: error of size mismatch here
    def get_low_mid_hi_syncopation_info(self, low_mid_hi_drum_map=Groove_Toolbox_3Part_keymap):
        """
            calculates monophonic syncopation of low/mid/high voice groups
            also weighted by their number of corresponding onsets

            Details of definitions here:
            [2] Drum rhythm spaces: From polyphonic similarity to generative maps by Daniel Gomez Marin, 2020
        :return:
            dictionary of all values calculated
        """

        metrical_profile = WITEK_SYNCOPATION_METRICAL_PROFILE_4_4_16th_NOTE

        # Get hits score reduced to low mid high groups
        lmh_hits = self.get_with_different_drum_mapping("h", tgt_drum_mapping=low_mid_hi_drum_map)

        lowsync = get_monophonic_syncopation(lmh_hits[:, 0], metrical_profile)
        midsync = get_monophonic_syncopation(lmh_hits[:, 1], metrical_profile)
        hisync = get_monophonic_syncopation(lmh_hits[:, 2], metrical_profile)
        lowsyness = (lowsync / np.count_nonzero(lmh_hits[:, 0])) if np.count_nonzero(lmh_hits[:, 0]) > 0 else 0
        midsyness = (midsync / np.count_nonzero(lmh_hits[:, 1])) if np.count_nonzero(lmh_hits[:, 1]) > 0 else 0
        hisyness = (hisync / np.count_nonzero(lmh_hits[:, 2])) if np.count_nonzero(lmh_hits[:, 2]) > 0 else 0

        return {"lowsync": lowsync, "midsync": midsync, "hisync": hisync,
                "lowsyness": lowsyness, "midsyness": midsyness, "hisyness": hisyness}

    def get_complexity_for_voice(self, voice_index):
        """ Calculated following Sioros and Guedes (2011) as
        combination of density and syncopation"""

        hit_density = self.get_hit_density_for_voice(voice_index)
        sync = self.get_monophonic_syncopation_for_voice(voice_index)
        return math.sqrt(pow(sync, 2) + pow(hit_density, 2))

    # todo: error of size mismatch here
    def get_total_complexity(self):
        hit_density = self.get_total_step_density()
        sync = self.get_combined_syncopation()
        return math.sqrt(pow(sync, 2) + pow(hit_density, 2))

    # ######################################################################
    #      Rhythmic Features::Autocorrelation Related from GrooveToolbox
    #
    #        The following code is mostly from the GrooveToolbox
    #              https://github.com/fredbru/GrooveToolbox
    #        Some additional functions have been implemented here
    #         to adapt hvo_sequence representation to the groove and
    #            utiming representations used in the GrooveToolbox
    #
    # Reference:    Yang, Li-Chia, and Alexander Lerch. "On the evaluation
    #               of generative models in music." Neural Computing
    #               and Applications 32.9 (2020): 4773-4784.
    # ######################################################################

    def get_total_autocorrelation_curve(self, hvo_str="v", offsets_in_ms=False):
        """
        Returns the autocorrelation of hvo score (according to hvo_str)
        the autocorrelation is calculated per voice and then added up per step

        :param hvo_str: str
            String formed with the characters 'h', 'v', 'o' and '0' in any order. It's not necessary to use all of the
            characters and they can be repeated. E.g. 'ov', will return the offsets and velocities, 'h0h'
            set offsets_in_ms to True if 'o' should be in milliseconds
        :return:
            autocorrelation curve for all parts summed.
        """

        if self.is_ready_for_use() is False:
            return None

        def autocorrelation(x):
            result = np.correlate(x, x, mode='full')
            return result[result.size // 2:]

        score = self.get(hvo_str=hvo_str, offsets_in_ms=offsets_in_ms)

        total_autocorrelation_curve = 0.0
        for i in range(score.shape[1]):
            total_autocorrelation_curve = total_autocorrelation_curve + autocorrelation(score[:, i])

        return total_autocorrelation_curve

    def get_velocity_autocorrelation_features(self):
        # Calculate autocorrelation curve

        if self.is_ready_for_use() is False:
            return None

        acorr = self.get_total_autocorrelation_curve(hvo_str="v")
        vels = self.get("v")

        # Create an empty dict to store features
        autocorrelation_features = dict()

        # Feature 1:
        # Calculate skewness of autocorrelation curve
        autocorrelation_features["skewness"] = stats.skew(acorr)

        # Feature 2:
        # Maximum of autocorrelation curve
        autocorrelation_features["max"] = acorr.max()

        # Feature 3:
        # Calculate acorr centroid Like spectral centroid - weighted mean of frequencies
        # in the signal, magnitude = weights.
        centroid_sum = 0
        total_weights = 0

        for i in range(acorr.shape[0]):
            # half wave rectify
            addition = acorr[i] * i  # sum all periodicity's in the signal
            if addition >= 0:
                total_weights += acorr[i]
                centroid_sum += addition

        if total_weights != 0:
            autocorrelation_centroid = centroid_sum / total_weights
        else:
            autocorrelation_centroid = vels.shape[0] / 2
        autocorrelation_features["centroid"] = autocorrelation_centroid

        # Feature 4:
        # Autocorrelation Harmonicity adapted from Lartillot et al. 2008
        total_autocorrelation_curve = acorr

        alpha = 0.15
        rectified_autocorrelation = acorr
        for i in range(total_autocorrelation_curve.shape[0]):
            if total_autocorrelation_curve[i] < 0:
                rectified_autocorrelation[i] = 0

        # weird syntax due to 2.x/3.x compatibility issues here todo: rewrite for 3.x
        peaks = np.asarray(find_peaks(rectified_autocorrelation))
        peaks = peaks[0] + 1  # peaks = lags

        inharmonic_sum = 0.0
        inharmonic_peaks = []
        for i in range(len(peaks)):
            remainder1 = 16 % peaks[i]
            if remainder1 > 16 * alpha and remainder1 < 16 * (1 - alpha):
                inharmonic_sum += rectified_autocorrelation[peaks[i] - 1]  # add magnitude of inharmonic peaks
                inharmonic_peaks.append(rectified_autocorrelation[i])

        if float(rectified_autocorrelation.max()) != 0:
            harmonicity = math.exp((-0.25 * len(peaks) * inharmonic_sum / float(rectified_autocorrelation.max())))
        else:
            harmonicity = np.nan
        autocorrelation_features["harmonicity"] = harmonicity
        return autocorrelation_features

    # ######################################################################
    #               Micro-timing Features from GrooveToolbox
    #
    #        The following code is mostly from the GrooveToolbox
    #              https://github.com/fredbru/GrooveToolbox
    #        Some additional functions have been implemented here
    #         to adapt hvo_sequence representation to the groove and
    #            utiming representations used in the GrooveToolbox
    #
    # Reference:    Yang, Li-Chia, and Alexander Lerch. "On the evaluation
    #               of generative models in music." Neural Computing
    #               and Applications 32.9 (2020): 4773-4784.
    # ######################################################################

    def swingness(self, use_tapped_pattern=False, mode=1):
        """
        Two modes here for calculating Swing implemented

        algorithm 1: same as groovetoolbox
            "Swung onsets are detected as significantly delayed second eighth-notes, approximating the typically
            understood 2:1 eighth-note swing ratio. Although musically these are considered as eighth notes,
            they fall into sixteenth note positions when quantized,
            with !!!significant negative (ahead of the position) deviations.!!!
            The ‘swingness’ feature first records whether these timing deviations occur or not,
            returning 0 for no swing or 1 for swing. This is then weighted by the number of swung onsets to model
            perceptual salience of the swing." [1]


        [1] Bruford, Fred, et al. "Multidimensional similarity modelling of complex drum loops
                                using the GrooveToolbox." (2020): 263-270.

        algorithm 2: Based on Ableton 16th note swing definitions
            The implementation here is slightly different!
            here we look out the utiming_ratio (not in ms) at each second and fourth time step within each
            quantization time_grid. (similar to 16th note swings in ableton and many other DAWs/Drum Machines)
            In this context, a +.5 swing at 2nd or 4th time steps means that the note is 100 percent swung
            while a value of <=0 means no swing
            In other words, we measure average of delayed timings on 2and 4th time steps in each beat (i.e. grid(1::2))

            maximum swing in this method is

        :param use_tapped_pattern:
        :param mode: int
                         0 --> groovetoolbox method
                         1 --> similar to DAW Swing

        :return: a tuple of swingness

        """

        if self.is_ready_for_use() is False:
            return None

        assert len(self.time_signatures) == 1, "Currently Swing calculation doesn't support time signature change"
        assert self.grid_type_per_segments[0] == "binary" and self.time_signatures[0].denominator == 4, \
            "Currently Swing calculation can only be done for binary grids with time signature denominator of 4"

        if mode == 0:
            # get micro-timings in ms and use np.nan whenever no note is played
            microtiming_matrix = self.get("o", offsets_in_ms=True, use_NaN_for_non_hits=True)
            n_steps = microtiming_matrix.shape[0]

            # Calculate average_timing_matrix
            average_timings_per_step = np.array([])

            # Get the mean of micro-timings per step
            for timings_at_step in microtiming_matrix:
                average_timings_per_step = np.append(
                    average_timings_per_step,
                    np.nanmean(timings_at_step) if not np.all(np.isnan(timings_at_step)) else 0
                )

            # get indices for swing positions = delayed 8th notes (timing on [fourth step, 8th, ..] 16th note grid)
            swung_note_positions = list(range(n_steps))[3::4]

            swing_count = 0.0
            j = 0
            for i in swung_note_positions:
                if average_timings_per_step[i] < -25.0:
                    swing_count += 1
                j += 1

            swing_count = np.clip(swing_count, 0, len(swung_note_positions))

            if swing_count > 0:
                swingness = (1 + (swing_count / len(swung_note_positions) / 10))  # todo: weight swing count
            else:
                swingness = 0.0

            return swingness

        elif mode == 1:
            # Get offsets at required swing steps
            microtiming_matrix = self.get("o", offsets_in_ms=False, use_NaN_for_non_hits=False)

            # look at offsets at 2nd, 4th steps in each beat (corresponding to  grid line indices 1, 3, 5, 7, ... )
            offset_at_swing_steps = microtiming_matrix[1::2, :]

            # get average of positive offsets at swing steps
            offset_at_swing_steps = offset_at_swing_steps[offset_at_swing_steps > 0]

            # return mean of swung offsets or zero if none
            # max swing should be 1 (but max offset is 0.5) hence mult by 2
            return offset_at_swing_steps.mean()*2 if offset_at_swing_steps.size > 0 else 0

    def laidbackness(self,
                     kick_key_in_drum_mapping="KICK",
                     snare_key_in_drum_mapping="SNARE",
                     hihat_key_in_drum_mapping="HH_CLOSED",
                     threshold=12.0
                    ):

        """
        Calculates how 'pushed' (or laidback) the loop is, based on number of pushed events /
        number of possible pushed events

        pushedness or laidbackness are calculated by looking at the timing of kick/snare/hat combinations with respect
        to each other

        :param kick_key_in_drum_mapping:
        :param snare_key_in_drum_mapping:
        :param hihat_key_in_drum_mapping:
        :param threshold:
        :return: laidbackness - pushedness
        """

        if self.is_ready_for_use() is False:
            return None

        assert len(self.time_signatures) == 1, " time signature missing or multiple time signatures." \
                                               "Currently  calculation doesn't support time signature change"
        assert self.grid_type_per_segments[0] == "binary" and self.time_signatures[0].denominator == 4, \
            "Currently  calculation can only be done for binary grids with time signature denominator of 4"

        # Get micro-timings in ms
        microtiming_matrix = self.get("o", offsets_in_ms=True)

        n_bars = int(np.ceil(microtiming_matrix.shape[0] / 16))

        for bar_n in range(n_bars):
            microtiming_event_profile_in_bar = self._getmicrotiming_event_profile_1bar(
                microtiming_matrix[bar_n:(bar_n + 1) * 16, :],
                kick_key_in_drum_mapping=kick_key_in_drum_mapping,
                snare_key_in_drum_mapping=snare_key_in_drum_mapping,
                hihat_key_in_drum_mapping=hihat_key_in_drum_mapping,
                threshold=threshold
            )
            if bar_n == 0:
                microtiming_event_profile = microtiming_event_profile_in_bar
            else:
                microtiming_event_profile = np.append(microtiming_event_profile, microtiming_event_profile_in_bar)

        # Calculate how 'pushed' the loop is, based on number of pushed events / number of possible pushed events
        push_events = microtiming_event_profile[1::2]
        push_event_count = np.count_nonzero(push_events)
        total_push_positions = push_events.shape[0]
        pushed_events = push_event_count / total_push_positions

        # Calculate how 'laid-back' the loop is,
        # based on the number of laid back events / number of possible laid back events
        laidback_events = microtiming_event_profile[0::2]
        laidback_event_count = np.count_nonzero(laidback_events)
        total_laidback_positions = laidback_events.shape[0]
        laidback_events = laidback_event_count / float(total_laidback_positions)

        return laidback_events - pushed_events

    def _getmicrotiming_event_profile_1bar(self,
                                           microtiming_matrix,
                                           kick_key_in_drum_mapping="KICK",
                                           snare_key_in_drum_mapping="SNARE",
                                           hihat_key_in_drum_mapping="HH_CLOSED",
                                           threshold=12.0):
        """
        Same implementation as groovetoolbox
        :param microtiming_matrix:                  offsets matrix for maximum of 1 bar in 4/4
        :param kick_key_in_drum_mapping:
        :param snare_key_in_drum_mapping:
        :param hihat_key_in_drum_mapping:
        :return:
            microtiming_event_profile_1bar
        """

        if self.is_ready_for_use() is False:
            return None

        assert len(self.time_signatures) == 1, "Currently Swing calculation doesn't support time signature change"
        assert self.grid_type_per_segments[0] == "binary" and self.time_signatures[0].denominator == 4, \
            "Currently Swing calculation can only be done for binary grids with time signature denominator of 4"

        # Ensure duration is 16 steps
        if microtiming_matrix.shape[0] < 16:
            pad_len = 16-microtiming_matrix.shape[0]
            for i in range(pad_len):
                microtiming_matrix = np.append(microtiming_matrix, np.zeros((1, microtiming_matrix.shape[1])))
        microtiming_matrix = microtiming_matrix[:16, :]

        # Get 2nd dimension indices corresponding to kick, snare and hi-hats using the keys provided
        kick_ix = list(self.drum_mapping.keys()).index(kick_key_in_drum_mapping)
        snare_ix = list(self.drum_mapping.keys()).index(snare_key_in_drum_mapping)
        chat_ix = list(self.drum_mapping.keys()).index(hihat_key_in_drum_mapping)

        microtiming_event_profile_1bar = _getmicrotiming_event_profile_1bar(microtiming_matrix, kick_ix, snare_ix, chat_ix, threshold )

        return microtiming_event_profile_1bar

    def get_timing_accuracy(self):
        # Calculate timing accuracy of the loop

        if self.is_ready_for_use() is False:
            return None

        def get_average_timing_deviation(microtiming_matrix):
            # Get vector of average microtiming deviation at each metrical position

            average_timing_matrix = np.zeros([microtiming_matrix.shape[0]])
            for i in range(microtiming_matrix.shape[0]):
                row_sum = 0.0
                hit_count = 0.0
                row_is_empty = np.all(np.isnan(microtiming_matrix[i, :]))
                if row_is_empty:
                    average_timing_matrix[i] = np.nan
                else:
                    for j in range(microtiming_matrix.shape[1]):
                        if np.isnan(microtiming_matrix[i, j]):
                            pass
                        else:
                            row_sum += microtiming_matrix[i, j]
                            hit_count += 1.0
                    average_timing_matrix[i] = row_sum / hit_count
            return average_timing_matrix

        # Get micro-timings in ms
        microtiming_matrix = self.get("o", offsets_in_ms=True)

        average_timing_matrix = get_average_timing_deviation(microtiming_matrix)

        swung_note_positions = list(range(average_timing_matrix.shape[0]))[3::4]
        nonswing_timing = 0.0
        nonswing_note_count = 0
        triplet_positions = 1, 5, 9, 13, 17, 21, 25, 29

        for i in range(average_timing_matrix.shape[0]):
            if i not in swung_note_positions and i not in triplet_positions:
                if ~np.isnan(average_timing_matrix[i]):
                    nonswing_timing += abs(np.nan_to_num(average_timing_matrix[i]))
                    nonswing_note_count += 1
        timing_accuracy = nonswing_timing / float(nonswing_note_count)

        return timing_accuracy

    # ######################################################################
    #                   Similarity/Distance Measures
    #
    #        The following code is partially from the GrooveToolbox
    #              https://github.com/fredbru/GrooveToolbox
    #        Some additional functions have been implemented here
    #         to adapt hvo_sequence representation to the groove and
    #            utiming representations used in the GrooveToolbox
    #
    # Reference:    Yang, Li-Chia, and Alexander Lerch. "On the evaluation
    #               of generative models in music." Neural Computing
    #               and Applications 32.9 (2020): 4773-4784.
    # ######################################################################

    def calculate_all_distances_with(self, hvo_seq_b):
        distances_dictionary = {
            "l1_distance_hvo": self.calculate_l1_distance_with(hvo_seq_b),
            "l1_distance_h": self.calculate_l1_distance_with(hvo_seq_b, "h"),
            "l1_distance_v": self.calculate_l1_distance_with(hvo_seq_b, "v"),
            "l1_distance_o": self.calculate_l1_distance_with(hvo_seq_b, "o"),
            "l2_distance_hvo": self.calculate_l2_distance_with(hvo_seq_b),
            "l2_distance_h": self.calculate_l2_distance_with(hvo_seq_b, "h"),
            "l2_distance_v": self.calculate_l2_distance_with(hvo_seq_b, "v"),
            "l2_distance_o": self.calculate_l2_distance_with(hvo_seq_b, "o"),
            "cosine_distance": self.calculate_cosine_distance_with(hvo_seq_b),
            "cosine_similarity": self.calculate_cosine_similarity_with(hvo_seq_b),
            "hamming_distance_all_voices_not_weighted": self.calculate_hamming_distance_with(
                hvo_seq_b, reduction_map=None, beat_weighting=False),
            "hamming_distance_all_voices_weighted": self.calculate_hamming_distance_with(
                hvo_seq_b, reduction_map=None, beat_weighting=True),
            "hamming_distance_low_mid_hi_not_weighted": self.calculate_hamming_distance_with(
                hvo_seq_b, reduction_map=Groove_Toolbox_3Part_keymap, beat_weighting=False),
            "hamming_distance_low_mid_hi_weighted": self.calculate_hamming_distance_with(
                hvo_seq_b, reduction_map=Groove_Toolbox_3Part_keymap, beat_weighting=True),
            "hamming_distance_5partKit_not_weighted": self.calculate_hamming_distance_with(
                hvo_seq_b, reduction_map=Groove_Toolbox_5Part_keymap, beat_weighting=False),
            "hamming_distance_5partKit_weighted": self.calculate_hamming_distance_with(
                hvo_seq_b, reduction_map=Groove_Toolbox_5Part_keymap, beat_weighting=True),
            "fuzzy_hamming_distance_not_weighted": self.calculate_fuzzy_hamming_distance_with(
                hvo_seq_b, beat_weighting=False),
            "fuzzy_hamming_distance_weighted": self.calculate_fuzzy_hamming_distance_with(
                hvo_seq_b, beat_weighting=True),
            "structural_similarity_distance": self.calculate_structural_similarity_distance_with(hvo_seq_b)
        }
        return distances_dictionary

    def calculate_l1_distance_with(self, hvo_seq_b, hvo_str="hvo"):
        """
        :param hvo_seq_b:   Sequence to find l1 norm of euclidean distance with
        :param hvo_str:     String formed with the characters 'h', 'v' and 'o' in any order. It's not necessary
                            to use all of the characters and they can be repeated. E.g. 'ov' or 'hvoh'
        :return:            l1 norm of euclidean distance with hvo_seq_b
        """
        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        a = self.get(hvo_str).flatten()
        b = hvo_seq_b.get(hvo_str).flatten()
        return np.linalg.norm((a - b), ord=1)

    def calculate_l2_distance_with(self, hvo_seq_b, hvo_str="hvo"):
        """
        :param hvo_seq_b:   Sequence to find l2 norm of euclidean distance with
        :param hvo_str:     String formed with the characters 'h', 'v' and 'o' in any order. It's not necessary
                            to use all of the characters and they can be repeated. E.g. 'ov' or 'hvoh'
        :return:            l2 norm of euclidean distance with hvo_seq_b
        """

        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        a = self.get(hvo_str).flatten()
        b = hvo_seq_b.get(hvo_str).flatten()
        return np.linalg.norm((a - b), ord=2)

    def calculate_cosine_similarity_with(self, hvo_seq_b):
        """
        Calculates cosine similarity with secondary sequence
        Calculates the cosine of the angle between flattened hvo scores (flatten into 1d)

        :param hvo_seq_b: a secondary hvo_sequence to measure similarity with
        :return:
            a value between -1 and 1 --> 1. 0 when sequences are equal, 0 when they are "perpendicular"
        """

        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        return cosine_similarity(self, hvo_seq_b)

    def calculate_cosine_distance_with(self, hvo_seq_b):
        # returns 1 - cosine_similarity
        # returns 0 when equal and 1 when "perpendicular"

        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        return cosine_distance(self, hvo_seq_b)

    def calculate_hamming_distance_with(self, hvo_seq_b, reduction_map=None, beat_weighting=False):
        """
        Calculates the vanilla hamming distance between the current hvo_seq and a target sequence

        :param hvo_seq_b:       target sequence from which the distance is measured
        :param reduction_map:   None:       Calculates distance as is
                                reduction_map: an alternative drum mapping to reduce the score
                                               options available in drum_mappings.py:
                                                1.      Groove_Toolbox_3Part_keymap (low, mid, hi)
                                                2.      Groove_Toolbox_5Part_keymap (kick, snare, closed, open, tom)

        :param beat_weighting:  If true, weights time steps using a 4/4 metrical awareness weights
        :return:
            distance:           abs value of distance between sequences
        """
        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        if reduction_map is not None:
            groove_a = self.get_with_different_drum_mapping("v", reduction_map)
            groove_b = hvo_seq_b.get_with_different_drum_mapping("v", reduction_map)
        else:
            groove_a = self.get("v")
            groove_b = hvo_seq_b.get("v")

        if beat_weighting is True:
            groove_a = _weight_groove(groove_a)
            groove_b = _weight_groove(groove_b)


        x = (groove_a.flatten() - groove_b.flatten())

        _weighted_Hamming_distance = math.sqrt(np.dot(x, x.T))

        return _weighted_Hamming_distance

    def calculate_fuzzy_hamming_distance_with(self, hvo_seq_b, beat_weighting=False):
        """
        Calculates the vanilla hamming distance between the current hvo_seq and a target sequence

        :param hvo_seq_b:       target sequence from which the distance is measured

        :param beat_weighting:  If true, weights time steps using a 4/4 metrical awareness weights
        :return:
            distance:           abs value of distance between sequences
        """

        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        velocity_grooveA = self.get("v")
        utiming_grooveA = self.get("o", offsets_in_ms = True, use_NaN_for_non_hits = True)
        velocity_grooveB = hvo_seq_b.get("v")
        utiming_grooveB = hvo_seq_b.get("o", offsets_in_ms=True, use_NaN_for_non_hits=True)

        fuzzy_dist = fuzzy_Hamming_distance(velocity_grooveA, utiming_grooveA,
                                            velocity_grooveB, utiming_grooveB,
                                            beat_weighting=beat_weighting)
        return fuzzy_dist

    def calculate_structural_similarity_distance_with(self, hvo_seq_b):
        """
        Calculates the vanilla hamming distance between the current hvo_seq and a target sequence

        :param hvo_seq_b:       target sequence from which the distance is measured

        :return:
            distance:           abs value of distance between sequences
        """
        
        if self.is_ready_for_use() is False or hvo_seq_b.is_ready_for_use() is False:
            return None

        groove_b = hvo_seq_b.get_reduced_velocity_groove()
        groove_a = self.get_reduced_velocity_groove()

        assert(groove_a.shape == groove_b.shape), "can't calculate structural similarity difference as " \
                                                  "the dimensions are different"

        x = (groove_b.flatten() - groove_a.flatten())

        structural_similarity_distance = math.sqrt(np.dot(x, x.T))

        return structural_similarity_distance

    def get_reduced_velocity_groove(self):
        # Remove ornamentation from a groove to return a simplified representation of the rhythm structure
        # change salience profile for different metres etc

        velocity_groove = self.get("v")

        metrical_profile_4_4 = [0, -2, -1, -2, 0, -2, -1, -2, -0, -2, -1, -2, -0, -2, -1, -2,
                            0, -2, -1, -2, 0, -2, -1, -2, 0, -2, -1, -2, 0, -2, -1, -2]

        reduced_groove = np.zeros(velocity_groove.shape)
        for i in range(velocity_groove.shape[1]):  # number of parts to reduce
            reduced_groove[:, i] = _reduce_part(velocity_groove[:, i], metrical_profile_4_4)

        rows_to_remove = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31]
        reduced_groove = np.delete(reduced_groove, rows_to_remove, axis=0)

        return reduced_groove

    def is_performance(self, velocity_threshold=0.3, offset_threshold=0.1):
        """
        By looking at the unique velocities and offsets, approximate whether the hvo_sequence comes from a
        performance MIDI file or not

        :param velocity_threshold:      threshold from 0 to 1 indicating what percentage of the velocities different
                                        from 0 and 1 must be unique to consider the sequence to be performance.

        :param offset_threshold:        threshold from 0 to 1 indicating what percentage of the offsets different
                                        from 0 must be unique to consider the sequence to be performance.
        :return:
            is_performance:             boolean value returning whether the sequence is or not from a performance
        """

        assert (0 <= velocity_threshold <= 1 and 0 <= offset_threshold <= 1), "Invalid threshold"
        is_perf = False

        nonzero_nonone_vels = self.velocities[(self.velocities != 0) & (self.velocities != 1)]
        unique_velocities = np.unique(nonzero_nonone_vels)
        unique_vel_perc = 0 if len(nonzero_nonone_vels) == 0 else len(unique_velocities) / len(nonzero_nonone_vels)

        nonzero_offsets = self.offsets[np.nonzero(self.offsets)[0]]
        unique_offsets = np.unique(nonzero_offsets)
        unique_off_perc = 0 if len(nonzero_offsets) == 0 else len(unique_offsets) / len(nonzero_offsets)

        if unique_vel_perc >= velocity_threshold and unique_off_perc >= offset_threshold:
            is_perf = True

        return is_perf
