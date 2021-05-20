import torch
import sys

sys.path.append('../../preprocessed_dataset/')
sys.path.append('../../BaseGrooveTransformers/models/')

from Subset_Creators.subsetters import GrooveMidiSubsetter
from dataset_loader import GrooveMidiDataset

from torch.utils.data import DataLoader
from transformer import GrooveTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
####################################

filters = {"beat_type": ["beat"], "time_signature": ["4-4"], "master_id": ["drummer9/session1/9"]}

# LOAD SMALL TRAIN SUBSET
pickle_source_path = '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2/Processed_On_17_05_2021_at_22_32_hrs'
subset_name = 'GrooveMIDI_processed_train'
metadata_csv_filename = 'metadata.csv'
hvo_pickle_filename = 'hvo_sequence_data.obj'

gmd_subsetter = GrooveMidiSubsetter(pickle_source_path=pickle_source_path, subset=subset_name,
                                    hvo_pickle_filename=hvo_pickle_filename, list_of_filter_dicts_for_subsets=[filters])

_, subset_list = gmd_subsetter.create_subsets()

subset_info = {"pickle_source_path": pickle_source_path, "subset": subset_name, "metadata_csv_filename":
    metadata_csv_filename, "hvo_pickle_filename": hvo_pickle_filename, "filters": filters}

mso_parameters = {"sr": 44100, "n_fft": 1024, "win_length": 1024, "hop_length": 441, "n_bins_per_octave": 16,
                  "n_octaves": 9, "f_min": 40, "mean_filter_size": 22}

voices_parameters = {"voice_idx": [2],  # closed hihat
                     "min_n_voices_to_remove": 1,
                     "max_n_voices_to_remove": 1,
                     "prob": [1],
                     "k": 1}

train_data = GrooveMidiDataset(subset=subset_list[0], subset_info=subset_info, mso_parameters=mso_parameters,
                               max_aug_items=100, voices_parameters=voices_parameters,
                               sf_path="../../TransformerGrooveInfilling/soundfonts/filtered_soundfonts/", max_n_sf=1)

print("data len", train_data.__len__())

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

####################################

# TRANSFORMER MODEL PARAMETERS
d_model = 128
nhead = 8
dim_feedforward = d_model * 10
dropout = 0.1
num_encoder_layers = 1
num_decoder_layers = 1
max_len = 32

embedding_size_src = 16
embedding_size_tgt = 27


TM = GrooveTransformer(d_model, embedding_size_src, embedding_size_tgt, nhead, dim_feedforward, dropout,
                       num_encoder_layers, num_decoder_layers, max_len).to(device)

# TRAINING PARAMETERS
learning_rate = 1e-3
batch_size = 64
epochs = 5

BCE = torch.nn.BCEWithLogitsLoss(reduction='sum')
MSE = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(TM.parameters(), lr=learning_rate)

def calculate_loss(pred_h,pred_v,pred_o, y):
    div = int(y.shape[2]/3)
    y_h, y_v, y_o = torch.split(y, div, 2)
    BCE_h = BCE(pred_h, y_h)
    MSE_v = MSE(pred_v, y_v)
    MSE_o = MSE(pred_o, y_o)
    print("BCE hits", BCE_h)
    print("MSE vels", MSE_v)
    print("MSE offs", MSE_o)
    return BCE_h + MSE_v + MSE_o

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X, y, idx) in enumerate(dataloader):
        print(X.shape, y.shape) # da Nx32xembedding_size
        X = X.permute(1,0,2)  # reorder dimensions to 32xNx embedding_size
        y = y.permute(1,0,2)  # reorder dimensions


        # y_shifted
        y_s = torch.zeros([1, y.shape[1], y.shape[2]])
        y_s = torch.cat((y_s, y[:-1,:,:]), dim=0)

        # prediction
        h, v, o = model(X, y_s)

        #loss
        loss = loss_fn(h,v,o, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    epochs = 100000
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, TM, calculate_loss, optimizer)
        print("Done!")

