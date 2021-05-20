import sys
import torch

sys.path.append('/')
from dev.outdated.transformer import Transformer


d_model = 16
nhead = 2
dim_feedforward = d_model*10
dropout = 0.1
num_encoder_layers = 5
num_decoder_layers = 6


TM = Transformer(d_model, nhead, dim_feedforward, dropout, num_encoder_layers, num_decoder_layers)

N = 24
src_len = 12
tgt_len = 14

src = torch.rand(src_len,N,d_model)
tgt = torch.rand(tgt_len, N, d_model)

print(TM.forward(src,tgt).shape)