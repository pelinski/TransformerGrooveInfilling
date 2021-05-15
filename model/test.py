import sys
sys.path.append('../../hvo_sequence/')
sys.path.append('../dev/')
sys.path.append('../../preprocessed_dataset/')


from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING
from dev.input_features.onsets import input_features_extractor
from dataset_loader import GrooveMidiDataset
import numpy as np
from Subset_Creators.subsetters import GrooveMidiSubsetter


if __name__ == "__main__":

    # Create an instance of a HVO_Sequence
    hvo_seq = HVO_Sequence(drum_mapping=ROLAND_REDUCED_MAPPING)

    # Add two time_signatures
    hvo_seq.add_time_signature(0, 4, 4, [4])

    # Add two tempos
    hvo_seq.add_tempo(0, 50)

    # Create a random hvo
    hits = np.random.randint(0, 2, (16, 9))
    vels = hits * np.random.rand(16, 9)
    offs = hits * (np.random.rand(16, 9) -0.5)

    #hvo_seq.get('0')
    #print(hvo_seq.hvo)

    # Add hvo score to hvo_seq instance
    hvo_bar = np.concatenate((hits, vels, offs), axis=1)
    hvo_seq.hvo = np.concatenate((hvo_bar, hvo_bar), axis=0)

    # Reset voices
    if True:
        hvo_reset, hvo_out_voices = hvo_seq.reset_voices([0])
        print(hvo_seq.hvo[:10,0],hvo_seq.hvo[:10,9],hvo_seq.hvo[:10,2*9])
        print(hvo_reset.hvo[:10,0],hvo_reset.hvo[:10,9],hvo_reset.hvo[:10,2*9])
        print(hvo_out_voices.hvo[:10,0],hvo_out_voices.hvo[:10,9],hvo_out_voices.hvo[:10,2*9])

    #mso
    if True:
        mso = hvo_reset.mso(sf_path='../soundfonts/filtered_soundfonts/7mb_vinyl_drums1.sf2')
        y = hvo_reset.synthesize(sf_path='../soundfonts/filtered_soundfonts/7mb_vinyl_drums1.sf2')
        mso_2 = input_features_extractor(y, qpm=hvo_reset.tempos[0].qpm)
        print(mso == mso_2)


    ## GMD
    if True:
        # load subset
        filters = {"beat_type": ["beat"],
                   "master_id": ["drummer1/eval_session/10"]}
        # subset creator
        pickle_source_path = '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.3.0/Processed_On_13_05_2021_at_12_56_hrs'
        subset_name = 'GrooveMIDI_processed_test'
        metadata_csv_filename = 'metadata.csv'
        hvo_pickle_filename = 'hvo_sequence_data.obj'

        gmd_subsetter = GrooveMidiSubsetter(
            pickle_source_path=pickle_source_path,
            subset=subset_name,
            hvo_pickle_filename=hvo_pickle_filename,
            list_of_filter_dicts_for_subsets=[filters],
        )
        _, subset_list = gmd_subsetter.create_subsets()

        subset_info = {"pickle_source_path": pickle_source_path,
                       "subset": subset_name,
                       "metadata_csv_filename": metadata_csv_filename,
                       "hvo_pickle_filename": hvo_pickle_filename,
                       "filters": filters}

        mso_parameters = {
            "sr": 44100,
            "n_fft": 1024,
            "win_length": 1024,
            "hop_length": 441,
            "n_bins_per_octave": 16,
            "n_octaves": 9,
            "f_min": 40,
            "mean_filter_size": 22
        }

        # load gmd
        gmd = GrooveMidiDataset(subset=subset_list[0], subset_info=subset_info, mso_parameters=mso_parameters)
        # check that input corresponds to mso
        _in, _, _ = gmd.__getitem__(1)
        hvo_in = gmd.get_hvo_sequence(1)
        hvo_in_reset, _ = hvo_in.reset_voices(voice_idx=gmd.get_voices_idx(1))
        mso = hvo_in_reset.mso(sf_path=gmd.get_soundfont(1))
        print(_in == mso)

