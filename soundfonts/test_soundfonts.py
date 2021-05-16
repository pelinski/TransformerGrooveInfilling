
import os
import pickle
import sys

sys.path.append('../../hvo_sequence/')
sys.path.append('../../preprocessed_dataset/')

import numpy as np
import librosa
import shutil

from hvo_sequence.hvo_seq import HVO_Sequence


# get example
source_path = '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.3.0' \
        '/Processed_On_13_05_2021_at_12_56_hrs'
print(os.path.join(source_path, "GrooveMIDI_processed_train", "hvo_data.obj"))
train_file = open(os.path.join(source_path, "GrooveMIDI_processed_train", "hvo_sequence_data.obj"),'rb')
train_set = pickle.load(train_file)
dataset_size = len(train_set)
ix =  int(np.random.random_sample()*dataset_size)
example = train_set[0]

# select pack
pack = 'pack2'
sf_path_root = os.path.join('./all_soundfonts', pack)
sf_list = os.listdir(sf_path_root)
print(sf_list)

# filter items
sounds_path = os.path.join('./sounds', pack)
if not os.path.exists(sounds_path):
    os.makedirs(sounds_path)

for idx,sf in enumerate(sf_list):
    print(idx)
    sf_path = os.path.join(sf_path_root, sf)
    sounds_path = os.path.join('./sounds', pack)
    if sf.split('.')[0] == "": continue  # .DS_Store
    filename = os.path.join(sounds_path, str(idx) + "_" + sf.split('.')[-2] + '.wav')
    print(sf_path)
    audio = example.synthesize(sr=44100, sf_path=sf_path)
    if len(librosa.onset.onset_detect(audio)) > 1:
        example.save_audio(filename=filename, sr=44100, sf_path=sf_path)
        shutil.copy(sf_path, './filtered_soundfonts')


