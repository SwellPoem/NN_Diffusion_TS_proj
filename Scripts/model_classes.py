# Description: This script contains the classes that are used in the encoder and decoder of the transformer_model.py script.
import math
import torch
from torch import nn
from einops import rearrange, reduce, repeat

# add positional information to the input data in the form of learnable embeddings
# useful for the transformer model since the order of the input data is not inherently captured
# input: embedding_dim, dropout, max_len
# output: sequence with positional embeddings
class LearnablePositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoder, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.positional_embeddings = nn.Parameter(torch.empty(1, max_len, embedding_dim))
        nn.init.uniform_(self.positional_embeddings, -0.02, 0.02)

    def forward(self, sequence):
        sequence = sequence + self.positional_embeddings
        return self.dropout_layer(sequence)


# apply a 1D convolution to the input data, which can be useful for extracting local features from the data
# dropout -> used for regularization to prevent overfitting
# transposition operations -> used to ensure that the convolution is applied to the correct dimension of the input data
# input: input_dim, output_dim, dropout
# output: output tensor after applying convolution
class ConvolutionalMLP(nn.Module):
    def __init__(self, input_dim, output_dim, resid_drop_prob=0.):
        super(ConvolutionalMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=resid_drop_prob),
        )

    def forward(self, input_tensor):
        transposed_input = input_tensor.transpose(1, 2)
        output = self.layers(transposed_input)
        return output.transpose(1, 2)
    

# adaptive layer normalization operation
# input: embedding_dim
# output: normalized input tensor
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super(AdaptiveLayerNorm, self).__init__()
        self.positional_embedding = PositionalEmbedding(embedding_dim)
        self.activation = nn.SiLU()
        self.linear_layer = nn.Linear(embedding_dim, embedding_dim * 2)
        self.layer_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, input_tensor, time_step, label_embedding=None):
        pos_emb = self.positional_embedding(time_step)
        if label_embedding is not None:
            pos_emb += label_embedding
        pos_emb = self.linear_layer(self.activation(pos_emb)).unsqueeze(1)
        scale, shift = pos_emb.chunk(2, dim=2)
        normalized_input = self.layer_norm(input_tensor)
        return normalized_input * (1 + scale) + shift
    

# generate a positional embedding that varies sinusoidally with the position
# model can take into account the order of the data
# used in the previous class
# input: embedding_dim
# output: positional embeddings
class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, input_tensor):
        device = input_tensor.device
        half_dim = self.embedding_dim // 2
        log_term = math.log(10000) / (half_dim - 1)
        scale = torch.exp(torch.arange(half_dim, device=device) * -log_term)
        scaled_input = input_tensor.unsqueeze(1) * scale.unsqueeze(0)
        sinusoidal_embedding = torch.cat((scaled_input.sin(), scaled_input.cos()), dim=-1)
        return sinusoidal_embedding
    

# transpose the dimensions of a tensor
# input: dim
# output: transposed tensor
class TensorTranspose(nn.Module):
    """ A module that transposes the dimensions of a tensor. """
    def __init__(self, dim1, dim2):
        super(TensorTranspose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input_tensor):
        return input_tensor.transpose(self.dim1, self.dim2)
    

# GELU2 function
# modified GELU activation function
# not use of tanh, but instead use the sigmoid function
# computationally cheaper
class GELU2(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    

#series decomposition operation on the input data
#separate a time series into a trend component and a residual component
# input: window_size
# output: residual, trend
class SeriesDecomposition(nn.Module):
    def __init__(self, window_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_average = MovingAverage(window_size, stride=1)

    def forward(self, input_tensor):
        trend = self.moving_average(input_tensor)
        residual = input_tensor - trend
        return residual, trend
    

#calculate the moving average of the input data
#used in the previous class
# input: window_size, step_size
# output: moving average of the input data
class MovingAverage(nn.Module):
    def __init__(self, window_size, step_size):
        super(MovingAverage, self).__init__()
        self.window_size = window_size
        self.average_pool = nn.AvgPool1d(kernel_size=window_size, stride=step_size, padding=0)

    def forward(self, input_tensor):
        padding_front = input_tensor[:, 0:1, :].repeat(1, self.window_size - 1 - (self.window_size - 1) // 2, 1)
        padding_end = input_tensor[:, -1:, :].repeat(1, (self.window_size - 1) // 2, 1)
        padded_input = torch.cat([padding_front, input_tensor, padding_end], dim=1)
        
        # calculate the moving average
        moving_average = self.average_pool(padded_input.permute(0, 2, 1)).permute(0, 2, 1)
        return moving_average
    

# multihead attention mechanism
# input: embd_dim, n_head, attn_drop_prob, resid_drop_prob
# output: output, avg_attention_scores
class MultiHeadAttention(nn.Module):    #decoder
    def __init__(self, embd_dim, n_head, attn_drop_prob=0.1, resid_drop_prob=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embd_dim % n_head == 0

        # Key, query, and value projections for all heads
        self.key_proj = nn.Linear(embd_dim, embd_dim)
        self.query_proj = nn.Linear(embd_dim, embd_dim)
        self.value_proj = nn.Linear(embd_dim, embd_dim)

        # Dropout layers
        self.attention_dropout = nn.Dropout(attn_drop_prob)
        self.resid_drop_prob = nn.Dropout(resid_drop_prob)

        # Output projection
        self.output_proj = nn.Linear(embd_dim, embd_dim)

        self.n_head = n_head

    def forward(self, x, mask=None):
        batch_size, seq_length, embd_dim = x.size()

        # Compute key, query, and value
        key = self.key_proj(x).view(batch_size, seq_length, self.n_head, embd_dim // self.n_head).transpose(1, 2)
        query = self.query_proj(x).view(batch_size, seq_length, self.n_head, embd_dim // self.n_head).transpose(1, 2)
        value = self.value_proj(x).view(batch_size, seq_length, self.n_head, embd_dim // self.n_head).transpose(1, 2)

        # Compute attention scores
        attention_scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.attention_dropout(attention_scores)

        # Apply attention scores to values
        output = attention_scores @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, embd_dim)

        # Compute average attention scores
        avg_attention_scores = attention_scores.mean(dim=1, keepdim=False)

        # Apply output projection and dropout
        output = self.resid_drop_prob(self.output_proj(output))

        return output, avg_attention_scores
    

# multihead cross-attention mechanism designed for attention between two different sequences
# input: embd_dim, condition_dim, n_head, attn_drop_prob, resid_drop_prob
# output: output, avg_attention_scores
class DecoderCrossAttention(nn.Module):   #decoder
    def __init__(self, embd_dim, condition_dim, n_head, attn_drop_prob=0.1, resid_drop_prob=0.1):
        super(DecoderCrossAttention, self).__init__()
        assert embd_dim % n_head == 0

        # Key, query, and value projections for all heads
        self.key_proj = nn.Linear(condition_dim, embd_dim)
        self.query_proj = nn.Linear(embd_dim, embd_dim)
        self.value_proj = nn.Linear(condition_dim, embd_dim)

        # Dropout layers
        self.attention_dropout = nn.Dropout(attn_drop_prob)
        self.resid_drop_prob = nn.Dropout(resid_drop_prob)

        # Output projection
        self.output_proj = nn.Linear(embd_dim, embd_dim)

        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        batch_size, seq_length, embd_dim = x.size()
        _, encoder_seq_length, _ = encoder_output.size()

        # Compute key, query, and value
        key = self.key_proj(encoder_output).view(batch_size, encoder_seq_length, self.n_head, embd_dim // self.n_head).transpose(1, 2)
        query = self.query_proj(x).view(batch_size, seq_length, self.n_head, embd_dim // self.n_head).transpose(1, 2)
        value = self.value_proj(encoder_output).view(batch_size, encoder_seq_length, self.n_head, embd_dim // self.n_head).transpose(1, 2)

        # Compute attention scores
        attention_scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.attention_dropout(attention_scores)

        # Apply attention scores to values
        output = attention_scores @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, embd_dim)

        # Compute average attention scores
        avg_attention_scores = attention_scores.mean(dim=1, keepdim=False)

        # Apply output projection and dropout
        output = self.resid_drop_prob(self.output_proj(output))

        return output, avg_attention_scores
    

# NEW BLOCK FOR THE TREND OF THE SERIES
# TrendBlock before
# trend estimation block
# input: input_dim, output_dim, input_features, output_features, activation
# output: trend_values
class TrendLayer(nn.Module):    #decoder
    def __init__(self, input_dim, output_dim, input_features, output_features, activation):
        super(TrendLayer, self).__init__()
        polynomial_degree = 3

        self.trend_estimator = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=polynomial_degree, kernel_size=3, padding=1),
            activation,
            TensorTranspose(1, 2),
            nn.Conv1d(input_features, output_features, 3, stride=1, padding=1)
        )

        linear_space = torch.arange(1, output_dim + 1, 1) / (output_dim + 1)
        self.polynomial_space = torch.stack([linear_space ** float(degree + 1) for degree in range(polynomial_degree)], dim=0)

    def forward(self, input_tensor):
        batch_size, channels, height = input_tensor.shape
        x = self.trend_estimator(input_tensor).transpose(1, 2)
        trend_values = torch.matmul(x.transpose(1, 2), self.polynomial_space.to(x.device))
        trend_values = trend_values.transpose(1, 2)
        return trend_values
    

# NEW BLOCK FOR THE SEASONALITY AND ERROR OF THE SERIES
# describe the seasonality of a time series using the inverse discrete Fourier transform
# input: d_model, min_freq, freq_factor
# output: time_series
class FourierLayer(nn.Module):    #decoder
    def __init__(self, d_model, min_freq=1, freq_factor=1):
        super().__init__()
        self.d_model = d_model
        self.freq_factor = freq_factor
        self.min_freq = min_freq

    def forward(self, input_tensor):
        batch_size, time_step, d_model = input_tensor.shape
        f_tensor = torch.fft.rfft(input_tensor, dim=1)

        if time_step % 2 == 0:
            f_tensor = f_tensor[:, self.min_freq:-1]
            f_values = torch.fft.rfftfreq(time_step)[self.min_freq:-1]
        else:
            f_tensor = f_tensor[:, self.min_freq:]
            f_values = torch.fft.rfftfreq(time_step)[self.min_freq:]

        f_tensor, index_tuple = self.top_K(f_tensor)
        f_values = repeat(f_values, 'f -> batch_size f d_model', batch_size=f_tensor.size(0), d_model=f_tensor.size(2)).to(f_tensor.device)
        f_values = rearrange(f_values[index_tuple], 'batch_size f d_model -> batch_size f () d_model').to(f_tensor.device)
        return self.compute_time_series(f_tensor, f_values, time_step)

    def compute_time_series(self, f_tensor, f_values, time_step):
        f_tensor = torch.cat([f_tensor, f_tensor.conj()], dim=1)
        f_values = torch.cat([f_values, -f_values], dim=1)
        time_values = rearrange(torch.arange(time_step, dtype=torch.float), 't -> () () t ()').to(f_tensor.device)

        amplitude = rearrange(f_tensor.abs(), 'batch_size f d_model -> batch_size f () d_model')
        phase = rearrange(f_tensor.angle(), 'batch_size f d_model -> batch_size f () d_model')
        time_series = amplitude * torch.cos(2 * math.pi * f_values * time_values + phase)
        return reduce(time_series, 'batch_size f t d_model -> batch_size t d_model', 'sum')

    def top_K(self, f_tensor):
        length = f_tensor.shape[1]
        top_k = int(self.freq_factor * math.log(length))
        values, indices = torch.topk(f_tensor.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(f_tensor.size(0)), torch.arange(f_tensor.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        f_tensor = f_tensor[index_tuple]
        return f_tensor, index_tuple