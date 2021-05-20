import torch

class Transformer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_encoder_layers, num_decoder_layers):
        super(Transformer, self).__init__()

        norm_encoder = torch.nn.LayerNorm(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.Encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm_encoder)

        norm_decoder = torch.nn.LayerNorm(d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.Decoder = torch.nn.TransformerDecoder(decoder_layer, num_decoder_layers, norm_decoder)

    def forward(self, src, tgt):
        memory = self.Encoder(src)  # no mask in the encoder
        mask = (torch.triu(torch.ones(tgt.shape[0], tgt.shape[0])) == 1)  # future mask
        out = self.Decoder(tgt, memory, tgt_mask=mask)

        return out