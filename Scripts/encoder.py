# Description: This file contains the Encoder class, which is a stack of EncoderBlocks.
from torch import nn
from Scripts.model_classes import AdaptiveLayerNorm, GELU2, MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, embd_dim=1024, n_head=16, attn_drop_prob=0.1, resid_drop_prob=0.1, mlp_hidden_times=4, activation='GELU'):
        super().__init__()

        self.layer1 = AdaptiveLayerNorm(embd_dim)
        self.layer2 = nn.LayerNorm(embd_dim)
        self.attn = MultiHeadAttention(embd_dim=embd_dim, n_head=n_head, attn_drop_prob=attn_drop_prob, resid_drop_prob=resid_drop_prob)
        
        assert activation in ['GELU', 'GELU2']
        act = nn.GELU() if activation == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(embd_dim, mlp_hidden_times * embd_dim),
                act,
                nn.Linear(mlp_hidden_times * embd_dim, embd_dim),
                nn.Dropout(resid_drop_prob),
            )
        
    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.layer1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.layer2(x))
        return x, att


class Encoder(nn.Module):
    def __init__(self, n_layer=14, embd_dim=1024, n_head=16, attn_drop_prob=0., resid_drop_prob=0., mlp_hidden_times=4,block_activation='GELU'):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(embd_dim=embd_dim,
                                                   n_head=n_head,
                                                   attn_drop_prob=attn_drop_prob,
                                                   resid_drop_prob=resid_drop_prob,
                                                   mlp_hidden_times=mlp_hidden_times,
                                                   activation=block_activation) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for idx in range(len(self.blocks)):
            x, _ = self.blocks[idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x