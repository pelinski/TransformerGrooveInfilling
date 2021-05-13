import pickle
import note_seq
import soundfile as sf
import numpy as np
import pretty_midi

from hvo_sequence.utils import find_nearest, find_pitch_and_tag
from hvo_sequence.hvo_seq import HVO_Sequence


def empty_like(other_hvo_sequence):
    """
    Creates an HVO_Sequence instance with the same fields as other_hvo_sequence. However, the hvo array for the
    returned sequence will be None (i.e. empty)

    :param other_hvo_sequence:      a HVO_Sequence instance
    :return:                        a HVO_Sequence instance same as other_hvo_sequence except the hvo field
    """

    new_hvo_seq = HVO_Sequence()



def note_sequence_to_hvo_sequence(ns, drum_mapping, beat_division_factors=[4], max_n_bars=None):
    """
            # Note_Sequence importer. Converts the note sequence to hvo format
            @param ns:                          Note_Sequence drum score
            @param beat_division_factors:       the number of divisions for each beat (must be a list)
            @param max_n_bars:                  maximum number of bars to import
            @return:
            """

    # Get tempo and time signature time stamps
    tempo_and_time_signature_time_stamps = list()
    tempo_time_stamps = np.array([x.time for x in ns.tempos])
    time_signature_time_stamps = np.array([x.time for x in ns.time_signatures])
    tempo_and_time_signature_time_stamps = np.unique(np.append(tempo_time_stamps, time_signature_time_stamps, axis=0))

    segment_lower_bounds = np.unique(np.array(tempo_and_time_signature_time_stamps))
    segment_upper_bounds = np.append(segment_lower_bounds[1:], ns.total_time)

    def get_tempo_and_time_signature_at_step(ix):
        _time_sig_ix = np.where((time_signature_time_stamps-ix)<=0,
                                time_signature_time_stamps-ix, -np.inf).argmax()
        _tempo_ix = np.where((tempo_time_stamps - ix) <= 0,
                             tempo_time_stamps - ix, -np.inf).argmax()
        _ns_time_sig = ns.time_signatures[_time_sig_ix]
        _ns_tempo = ns.tempos[_tempo_ix]
        return _ns_tempo, _ns_time_sig

    grid_lines = np.array([])

    for segment_ix, (lower_b, upper_b) in enumerate(zip(segment_lower_bounds, segment_upper_bounds)):
        ns_tempo, ns_time_sig = get_tempo_and_time_signature_at_step(lower_b)
        segment_grid_lines = np.array([])
        for beat_div in beat_division_factors:
            beat_duration_in_segment = (60.0 / ns_tempo.qpm) * 4.0 / ns_time_sig.denominator
            step_duration = beat_duration_in_segment/beat_div
            segment_grid_lines = np.append(segment_grid_lines, np.arange(lower_b, upper_b, step_duration))
        grid_lines = np.append(grid_lines, np.unique(segment_grid_lines))

    def snap_time_stamp_to_grid(time_stamp):
        time_step, _ = find_nearest(grid_lines, time_stamp)
        return int(time_step)

    # Create an empty HVO_Sequence instance
    hvo_seq = HVO_Sequence(drum_mapping=drum_mapping)

    # Add tempos and time signatures to hvo_seq instance
    for tempo in ns.tempos:
        hvo_seq.add_tempo(snap_time_stamp_to_grid(tempo.time), tempo.qpm)
    for time_sig in ns.time_signatures:
        hvo_seq.add_time_signature(
            time_step=snap_time_stamp_to_grid(time_sig.time),
            numerator=time_sig.numerator,
            denominator=time_sig.denominator,
            beat_division_factors=beat_division_factors
        )

    # Create an empty hvo_array
    hvo_seq.hvo = np.zeros((len(grid_lines), 3*len(drum_mapping.keys())))

    for ns_note in ns.notes:
        hvo_seq.hvo = place_note_in_hvo(ns_note=ns_note, hvo=hvo_seq.hvo, grid=grid_lines, drum_mapping=drum_mapping)
    return hvo_seq


def place_note_in_hvo(ns_note, hvo, grid, drum_mapping):
    """
    updates the entries in hvo corresponding to features available in ns_note

    @param ns_note:                 note_sequence.note to place in hvo matrix
    @param hvo:                     hvo matrix created for the note_sequence score
    @param grid:                    grid corresponding to hvo matrix (list of time stamps in seconds)
    @param drum_mapping:            dict of {'Drum Voice Tag': [midi numbers]}
    @return hvo:                    hvo matrix containing information from ns_note
    """

    grid_index, utiming = get_grid_position_and_utiming_in_hvo(ns_note.start_time, grid)

    _, _, pitch_group_ix = find_pitch_and_tag(ns_note.pitch, drum_mapping)

    n_drum_voices = len(drum_mapping.keys())  # Get the number of reduced drum classes

    # if pitch was in the pitch_class_list
    # also check if the corresponding note is already filled, only update if the velocity is louder
    if (pitch_group_ix is not None) and ((ns_note.velocity / 127.0) > hvo[grid_index, n_drum_voices]):
        hvo[grid_index, pitch_group_ix] = 1  # Set hit  to 1
        hvo[grid_index, pitch_group_ix + n_drum_voices] = ns_note.velocity / 127.0  # Set velocity (0, 1)
        hvo[grid_index, pitch_group_ix + n_drum_voices * 2] = utiming  # Set utiming (-.5, 0.5)

    return hvo


def get_grid_position_and_utiming_in_hvo(start_time, grid):
    """
    Finds closes grid line and the utiming deviation from the grid for a queried onset time in sec

    @param start_time:                  Starting position of a note
    @param grid:                        Grid lines (list of time stamps in sec)
    @return tuple of grid_index,        the index of the grid line closes to note
            and utiming:                utiming ratio in (-0.5, 0.5) range
    """
    grid_index, grid_sec = find_nearest(grid, start_time)

    utiming = start_time - grid_sec                         # utiming in sec

    if utiming < 0:                                         # Convert to a ratio between (-0.5, 0.5)
        if grid_index == 0:
            utiming = 0
        else:
            utiming = utiming / (grid[grid_index] - grid[grid_index-1])
    else:
        if grid_index == (grid.shape[0]-1):
            utiming = utiming / (grid[grid_index] - grid[grid_index-1])
        else:
            utiming = utiming / (grid[grid_index+1] - grid[grid_index])

    return grid_index, utiming


#   ------------------- Pickle Loader --------------------

def get_pickled_note_sequences(pickle_path, item_list=None):
    """
    loads a pickled file of note_sequences, also allows for grabbing only specific items from the pickled set
    @param pickle_path:                     # path to the pickled set
    @param item_list:                       # list of items to grab, leave as None to get all
    @return note_sequences:                 # either all the items in set (when item_list == None)
                                            # or a list of sequences (when item_list = [item indices]
                                            # or a single item (when item_list is an int)
    """
    # load pickled items
    note_sequence_pickle_file = open(pickle_path, 'rb')
    note_sequences = pickle.load(note_sequence_pickle_file)

    # get queried items or all
    if item_list:                            # check if specific items are queried
        if isinstance(item_list, list):      # Grab queried items
            note_sequences = [note_sequences[item] for item in item_list]
        else:                                # in case a single item (as integer) requested
            note_sequences = note_sequences[item_list]

    return note_sequences


def get_pickled_hvos(pickle_path, item_list=None):
    """
    loads a pickled file of hvos, also allows for grabbing only specific items from the pickled set
    @param pickle_path:                     # path to the pickled set
    @param item_list:                       # list of items to grab, leave as None to get all
    @return hvos:                           # either all the items in set (when item_list == None)
                                            # or a list of hvos (when item_list = [item indices]
                                            # or a single item (when item_list is an int)
    """
    # load pickled items
    hvo_pickle_file = open(pickle_path, 'rb')
    hvos = pickle.load(hvo_pickle_file)

    # get queried items or all
    if item_list:                            # check if specific items are queried
        if isinstance(item_list, list):      # Grab queried items
            hvos = [hvos[item] for item in item_list]
        else:                                # in case a single item (as integer) requested
            hvos = hvos[item_list]

    return hvos


#   --------------- Data type Convertors --------------------

def get_reduced_pitch(pitch_query, pitch_class_list):
    """
    checks to which drum group the pitch belongs,
    then returns the index for group and the first pitch in group

    @param pitch_query:                 pitch_query for which the corresponding drum voice group is found
    @param pitch_class_list:            list of grouped pitches sharing same tags      [..., [50, 48], ...]
    @return tuple of voice_group_ix,    index of the drum voice group the pitch belongs to
            and pitch_group[0]:         the first pitch in the corresponding drum group
                                        Note: Returns (None, None) if no matching group is found
    """

    for voice_group_ix, pitch_group in enumerate(pitch_class_list):
        if pitch_query in pitch_group:
            return voice_group_ix, pitch_group[0]

    # If pitch_query isn't in the pitch_class_list, return None, None
    return None, None


def unique_pitches_in_note_sequence(ns):
    """
    Returns unique pitches existing in a note sequence score
    @param ns: note sequence object
    @return: list of unique pitches
    """
    unique_pitches = set([note.pitch for note in ns.notes])
    return unique_pitches


#   -------------------- Midi Converters -----------------------

def save_note_sequence_to_midi(ns, filename="temp.mid"):
    pm = note_seq.note_sequence_to_pretty_midi(ns)
    pm.write(filename)


#   -------------------- Audio Synthesizers --------------------

def note_sequence_to_audio(ns, sr=44100, sf_path="../test/soundfonts/Standard_Drum_Kit.sf2"):
    """
    Synthesize a note_sequence score to an audio vector using a soundfont
    if you want to save the audio, use save_note_sequence_to_audio()
    @param ns:                  note_sequence score
    @param sr:                  sample_rate for generating audio
    @param sf_path:             soundfont for synthesizing to audio
    @return audio:              the generated audio
    """
    pm = note_seq.note_sequence_to_pretty_midi(ns)
    audio = pm.fluidsynth(fs=sr, sf2_path=sf_path)
    return audio


def save_note_sequence_to_audio(ns, filename="temp.wav", sr=44100,
                                sf_path="../test/soundfonts/Standard_Drum_Kit.sf2"):
    """
    Synthesize and save a note_sequence score to an audio vector using a soundfont
    @param ns:                  note_sequence score
    @param filename:            filename/path for saving the synthesized audio
    @param sr:                  sample_rate for generating audio
    @param sf_path:             soundfont for synthesizing to audio
    @return audio:              returns audio, in addition to saving it
    """
    """
        Synthesize a note_sequence score to an audio vector using a soundfont

        @param ns:                  note_sequence score
        @param sr:                  sample_rate for generating audio
        @param sf_path:             soundfont for synthesizing to audio
        @return audio:              the generated audio
        """

    pm = note_seq.note_sequence_to_pretty_midi(ns)
    audio = pm.fluidsynth(sf2_path=sf_path)
    sf.write(filename, audio, sr, 'PCM_24')
    return audio

