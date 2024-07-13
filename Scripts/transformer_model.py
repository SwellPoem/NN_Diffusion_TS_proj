# Description: This file contains the implementation of the Transformer model
import torch
from torch import nn
from Scripts.model_classes import ConvolutionalMLP, LearnablePositionalEncoder
from Scripts.encoder import Encoder
from Scripts.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, n_feat, n_channel, n_layer_enc=5, n_layer_dec=14, embd_dim=1024, n_heads=16, attn_drop_prob=0.1, resid_drop_prob=0.1, mlp_hidden_times=4, block_activation='GELU', max_len=2048, conv_params=None):
        super().__init__()
        self.emb = ConvolutionalMLP(n_feat, embd_dim, resid_drop_prob=resid_drop_prob)      # Embedding layer
        self.inverse = ConvolutionalMLP(embd_dim, n_feat, resid_drop_prob=resid_drop_prob)      # Inverse layer

        #set the convolutional parameters
        #if not provided, set the kernel size and padding based on the input size
        #if provided, use the provided parameters
        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        #convolutional layers to combine the trend and seasonality
        self.combine_s = nn.Conv1d(embd_dim, n_feat, kernel_size=kernel_size, stride=1, padding=padding, padding_mode='circular', bias=False)
        self.combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0, padding_mode='circular', bias=False)

        #encoder and positional encoder
        self.encoder = Encoder(n_layer_enc, embd_dim, n_heads, attn_drop_prob, resid_drop_prob, mlp_hidden_times, block_activation)
        self.pos_enc = LearnablePositionalEncoder(embd_dim, dropout=resid_drop_prob, max_len=max_len)

        #decoder and positional encoder
        self.decoder = Decoder(n_channel, n_feat, embd_dim, n_heads, n_layer_dec, attn_drop_prob, resid_drop_prob, mlp_hidden_times, block_activation, condition_dim=embd_dim)
        self.pos_dec = LearnablePositionalEncoder(embd_dim, dropout=resid_drop_prob, max_len=max_len)

    def forward(self, input, t, padding_masks=None, return_res=False):
        #firts input goes through the embedding layer
        emb = self.emb(input)
        #then through positional encoder
        inp_enc = self.pos_enc(emb)
        #pass the encoder output to the encoder
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)

        #the encoded input is passed to the positional decoder and then to the decoder
        inp_dec = self.pos_dec(emb)
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks)

        #calculate the resoiduals
        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        #compute season error and trend of the series
        season_error = self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        trend = self.combine_m(mean) + res_m + trend

        #return also the residuals if requested in addition to the trend and season error
        if return_res:
            return trend, self.combine_s(season.transpose(1, 2)).transpose(1, 2), res - res_m
        #otherwise return only the trend and season error
        return trend, season_error


if __name__ == '__main__':
    pass