# Description: this class is the Decoder class, which is a stack of DecoderBlocks. 
# The DecoderBlock is similar to the EncoderBlock, but with a few differences:
# has an additional DecoderCrossAttention layer used to combine the encoder output with the decoder output,
# has a TrendLayer and a FourierLayer used to extract trend and seasonal components from the input data. 
import torch
from torch import nn
from Scripts.model_classes import AdaptiveLayerNorm, GELU2, MultiHeadAttention, TrendLayer, FourierLayer, DecoderCrossAttention

    
class DecoderBlock(nn.Module):
    def __init__(self, n_channel, n_feat, embd_dim=1024, n_head=16, attn_drop_prob=0.1, resid_drop_prob=0.1, mlp_hidden_times=4, activation='GELU', condition_dim=1024):
        super().__init__()
        
        self.norm_layer1 = AdaptiveLayerNorm(embd_dim)
        self.norm_layer2 = nn.LayerNorm(embd_dim)

        self.self_attention = MultiHeadAttention(embd_dim=embd_dim, n_head=n_head, attn_drop_prob=attn_drop_prob, resid_drop_prob=resid_drop_prob)
        self.cross_attention = DecoderCrossAttention(embd_dim=embd_dim, condition_dim=condition_dim, n_head=n_head, attn_drop_prob=attn_drop_prob,resid_drop_prob=resid_drop_prob)
        
        self.norm_layer3 = AdaptiveLayerNorm(embd_dim)

        assert activation in ['GELU', 'GELU2']
        act = nn.GELU() if activation == 'GELU' else GELU2()

        # new trend and seasonal components
        self.trend_layer = TrendLayer(n_channel, n_channel, embd_dim, n_feat, activation=act)
        self.seasonal_layer = FourierLayer(d_model=embd_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embd_dim, mlp_hidden_times * embd_dim),
            act,
            nn.Linear(mlp_hidden_times * embd_dim, embd_dim),
            nn.Dropout(resid_drop_prob),
        )

        self.projection = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(embd_dim, n_feat)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        a, _ = self.self_attention(self.norm_layer1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, _ = self.cross_attention(self.norm_layer3(x, timestep), encoder_output, mask=mask)
        x = x + a
        x1, x2 = self.projection(x).chunk(2, dim=1)
        trend, season = self.trend_layer(x1), self.seasonal_layer(x2)
        x = x + self.mlp(self.norm_layer2(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season
    

class Decoder(nn.Module):
    def __init__(self, n_channel, n_feat, embd_dim=1024, n_head=16, n_layer=10, attn_drop_prob=0.1, resid_drop_prob=0.1, mlp_hidden_times=4, block_activation='GELU', condition_dim=512):
      super(Decoder, self).__init__()
      self.d_model = embd_dim
      self.n_feat = n_feat
      self.blocks = nn.Sequential(*[DecoderBlock(n_feat=n_feat,
                                                 n_channel=n_channel,
                                                 embd_dim=embd_dim,
                                                 n_head=n_head,
                                                 attn_drop_prob=attn_drop_prob,
                                                 resid_drop_prob=resid_drop_prob,
                                                 mlp_hidden_times=mlp_hidden_times,
                                                 activation=block_activation,
                                                 condition_dim=condition_dim) for _ in range(n_layer)])
      
    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, c, _ = x.shape
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device)
        trend = torch.zeros((b, c, self.n_feat), device=x.device)
        for idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = self.blocks[idx](x, enc, t, mask=padding_masks, label_emb=label_emb)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season