import os
import time
import logging
import signal

logging.basicConfig(level="WARNING")

import fairseq
import parselmouth
import pyworld
import torch
import numpy as np


LRELU_SLOPE = 0.1
HERE = os.path.dirname(__file__)


if not torch.cuda.is_available():
    device = "cpu"
    is_half = False
else:
    device = "cuda:0"
    gpu_name = torch.cuda.get_device_name(int(device.split(":")[-1]))
    is_half = False if "16" in gpu_name or "MX" in gpu_name else True
device = torch.device(device)


if not os.path.exists(os.path.join(HERE, "hubert_base.pt")):
    with open(os.path.join(HERE, "hubert_base.pt"), "wb") as of:
        with open(os.path.join(HERE, "hubert_base_1.py"), "rb") as f: of.write(f.read())
        with open(os.path.join(HERE, "hubert_base_2.py"), "rb") as f: of.write(f.read())
hubert_model = fairseq.checkpoint_utils.load_model_ensemble_and_task([os.path.join(HERE, "hubert_base.pt")], suffix="")[0][0]
hubert_model = hubert_model.to(device).half() if is_half else hubert_model.to(device).float()
hubert_model.eval()


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1: m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    return [item for sublist in pad_shape[::-1] for item in sublist]


def sequence_mask(length, max_length=None):
    if max_length is None: max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int, in_act = n_channels[0], input_a + input_b
    return torch.tanh(in_act[:,:n_channels_int,:]) * torch.sigmoid(in_act[:,n_channels_int:,:])


def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:,:,:segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        ret[i] = x[i,:,idx_str:idx_str+segment_size]
    return ret


def slice_segments2(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:,:segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        ret[i] = x[i, idx_str:idx_str+segment_size]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0,
                 window_size=None, heads_share=True, block_length=None,
                 proximal_bias=False, proximal_init=False):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None
        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = torch.nn.Parameter(
                torch.randn(n_heads_rel, window_size*2+1, self.k_channels)*rel_stddev)
            self.emb_rel_v = torch.nn.Parameter(
                torch.randn(n_heads_rel, window_size*2+1, self.k_channels)*rel_stddev)
        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        scores = torch.matmul(query / np.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query/np.sqrt(self.k_channels), key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                block_mask = (torch.ones_like(scores).triu(-self.block_length).tril(self.block_length))
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings)
        return output.transpose(2, 3).contiguous().view(b, d, t_t), p_attn

    def _matmul_with_relative_values(self, x, y):
        return torch.matmul(x, y.unsqueeze(0))

    def _matmul_with_relative_keys(self, x, y):
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = torch.nn.functional.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length,pad_length], [0, 0]]))
        else: padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:,slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = torch.nn.functional.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        return x_flat.view([batch,heads,length+1,2*length-1])[:,:,:length,length-1:]

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        x_flat = torch.nn.functional.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        return x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)

class LayerNorm(torch.nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return torch.nn.functional.layer_norm(
            x.transpose(1, -1), (self.channels,),
            self.gamma, self.beta, self.eps).transpose(1, -1)


class FFN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels,
                 kernel_size, p_dropout=0.0, activation=None, causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        self.padding = self._causal_padding if causal else self._same_padding
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu": x = x * torch.sigmoid(1.702 * x)
        else: x = torch.relu(x)
        return self.conv_2(self.padding(self.drop(x)*x_mask)) * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1: return x
        padding = [[0,0],[0,0],[self.kernel_size-1,0]]
        return torch.nn.functional.pad(x, convert_pad_shape(padding))

    def _same_padding(self, x):
        if self.kernel_size == 1: return x
        padding = [[0,0],[0,0],[(self.kernel_size-1)//2,self.kernel_size//2]]
        return torch.nn.functional.pad(x, convert_pad_shape(padding))


class Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers,
                 kernel_size=1, p_dropout=0.0, window_size=10, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(
                hidden_channels, hidden_channels, n_heads,
                p_dropout=p_dropout, window_size=window_size))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(
                hidden_channels, hidden_channels,
                filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.drop(self.attn_layers[i](x, x, attn_mask))
            x = self.norm_layers_1[i](x + y)
            y = self.drop(self.ffn_layers[i](x, x_mask))
            x = self.norm_layers_2[i](x + y)
        return x * x_mask


class TextEncoder256(torch.nn.Module):
    def __init__(self, out_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, f0=True):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.emb_phone = torch.nn.Linear(256, hidden_channels)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)
        if f0 == True: self.emb_pitch = torch.nn.Embedding(256, hidden_channels)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone, pitch, lengths):
        if pitch == None: x = self.emb_phone(phone)
        else: x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = torch.transpose(self.lrelu(x*np.sqrt(self.hidden_channels)), 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        stats = self.proj(self.encoder(x*x_mask, x_mask)) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class WN(torch.nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate,
                 n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = torch.nn.Dropout(p_dropout)
        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")
        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size*dilation-dilation)/2)
            in_layer = torch.nn.Conv1d(
                hidden_channels, 2*hidden_channels,
                kernel_size, dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)
            if i < n_layers - 1: res_skip_channels = 2 * hidden_channels
            else: res_skip_channels = hidden_channels
            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        if g is not None: g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
            else: g_l = torch.zeros_like(x_in)
            acts = self.drop(fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor))
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x+res_skip_acts[:,:self.hidden_channels,:]) * x_mask
                output = output + res_skip_acts[:,self.hidden_channels:,:]
            else: output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


class ResidualCouplingLayer(torch.nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, p_dropout=0, gin_channels=0, mean_only=False):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = torch.nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                      p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = torch.nn.Conv1d(hidden_channels, self.half_channels*(2-mean_only),1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        stats = self.post(self.enc(self.pre(x0)*x_mask, x_mask, g=g)) * x_mask
        if not self.mean_only: m, logs = torch.split(stats, [self.half_channels]*2, 1)
        else: m, logs = stats, torch.zeros_like(stats)
        if not reverse: return torch.cat([x0,m+x1*torch.exp(logs)*x_mask], 1), torch.sum(logs, [1,2])
        else: return torch.cat([x0,(x1-m)*torch.exp(-logs)*x_mask], 1)

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class Flip(torch.nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if reverse: return x
        return x, torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)


class ResidualCouplingBlock(torch.nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size,
                 dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.flows = torch.nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(
                channels, hidden_channels, kernel_size, dilation_rate,
                n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        flows = reversed(self.flows) if reverse else self.flows
        for flow in flows: x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows): self.flows[i * 2].remove_weight_norm()


class PosteriorEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels,
                 kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.pre = torch.nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        stats = self.proj(self.enc(self.pre(x)*x_mask, x_mask, g=g)) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m+torch.randn_like(m)*torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = torch.nn.ModuleList([
            torch.nn.utils.weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[0],
                padding=get_padding(kernel_size, dilation[0]))),
            torch.nn.utils.weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[1],
                padding=get_padding(kernel_size, dilation[1]))),
            torch.nn.utils.weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[2],
                padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)
        self.convs2 = torch.nn.ModuleList([
            torch.nn.utils.weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=1,
                padding=get_padding(kernel_size, 1))),
            torch.nn.utils.weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=1,
                padding=get_padding(kernel_size, 1))),
            torch.nn.utils.weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=1,
                padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None: xt = xt * x_mask
            xt = torch.nn.functional.leaky_relu(c1(xt), LRELU_SLOPE)
            if x_mask is not None: xt = xt * x_mask
            x = c2(xt) + x
        return x if x_mask is None else x * x_mask

    def remove_weight_norm(self):
        for l in self.convs1: torch.nn.utils.remove_weight_norm(l)
        for l in self.convs2: torch.nn.utils.remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.utils.weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[0],
                padding=get_padding(kernel_size, dilation[0]))),
            torch.nn.utils.weight_norm(torch.nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[1],
                padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None: xt = xt * x_mask
            x = c(xt) + x
        return x if x_mask is None else x * x_mask

    def remove_weight_norm(self):
        for l in self.convs: torch.nn.utils.remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = torch.nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == "1" else ResBlock2
        self.ups = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(torch.nn.utils.weight_norm(torch.nn.ConvTranspose1d(
                upsample_initial_channel//(2**i),
                upsample_initial_channel//(2**(i+1)),
                k, u, padding=(k-u)//2)))
        self.resblocks = torch.nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None: x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = self.ups[i](torch.nn.functional.leaky_relu(x, LRELU_SLOPE))
            xs = None
            for j in range(self.num_kernels):
                if xs is None: xs = self.resblocks[i*self.num_kernels+j](x)
                else: xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        return torch.tanh(self.conv_post(torch.nn.functional.leaky_relu(x)))

    def remove_weight_norm(self):
        for l in self.ups: torch.nn.utils.remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()


class SineGen(torch.nn.Module):
    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        return torch.ones_like(f0) * (f0 > self.voiced_threshold)

    def forward(self, f0, upp):
        with torch.no_grad():
            f0 = f0[:,None].transpose(1, 2)
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:,:,0] = f0[:,:,0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:,:,idx+1] = f0_buf[:,:,0] * (idx+2)
            rad_values = (f0_buf/self.sampling_rate) % 1
            rand_ini = torch.rand(f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device)
            rand_ini[:,0] = 0
            rad_values[:,0,:] = rad_values[:,0,:] + rand_ini
            tmp_over_one = torch.cumsum(rad_values, 1)
            tmp_over_one *= upp
            tmp_over_one = torch.nn.functional.interpolate(
                tmp_over_one.transpose(2, 1),
                scale_factor=upp,
                mode="linear",
                align_corners=True).transpose(2, 1)
            rad_values = torch.nn.functional.interpolate(
                rad_values.transpose(2, 1),
                scale_factor=upp, mode="nearest").transpose(2, 1)
            tmp_over_one %= 1
            tmp_over_one_idx = (tmp_over_one[:,1:,:]-tmp_over_one[:,:-1,:]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:,1:,:] = tmp_over_one_idx * -1.0
            sine_waves = torch.sin(torch.cumsum(rad_values+cumsum_shift, dim=1)*2*np.pi)
            sine_waves = sine_waves * self.sine_amp
            uv = self._f02uv(f0)
            uv = torch.nn.functional.interpolate(
                uv.transpose(2, 1), scale_factor=upp, mode="nearest").transpose(2, 1)
            noise_amp = uv * self.noise_std + (1-uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
        is_half=True
    ):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp,
            add_noise_std, voiced_threshod)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp=None):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        if self.is_half == True: sine_wavs = sine_wavs.half()
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


class GeneratorNSF(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        sr,
        is_half=False
    ):
        super(GeneratorNSF, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sampling_rate=sr, harmonic_num=0, is_half=is_half)
        self.noise_convs = torch.nn.ModuleList()
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == "1" else ResBlock2
        self.ups = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(torch.nn.utils.weight_norm(torch.nn.ConvTranspose1d(
                upsample_initial_channel//(2**i),
                upsample_initial_channel//(2**(i+1)),
                k, u, padding=(k-u)//2)))
            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(torch.nn.Conv1d(
                    1, c_cur, kernel_size=stride_f0*2, stride=stride_f0, padding=stride_f0//2))
            else: self.noise_convs.append(torch.nn.Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = torch.nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        if gin_channels != 0: self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        self.upp = np.prod(upsample_rates)

    def forward(self, x, f0, g=None):
        har_source, noi_source, uv = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None: x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None: xs = self.resblocks[i * self.num_kernels + j](x)
                else: xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        return torch.tanh(self.conv_post(torch.nn.functional.leaky_relu(x)))

    def remove_weight_norm(self):
        for l in self.ups: torch.nn.utils.remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()


class SynthesizerTrnMs256NSFSid(torch.nn.Module):
    def __init__(
        self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
        n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes,
        resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
        upsample_kernel_sizes, spk_embed_dim, gin_channels, sr, **kwargs
    ):
        super().__init__()
        if type(sr) == str: sr = {"32k": 32000, "40k": 40000, "48k": 48000}[sr]
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.enc_p = TextEncoder256(
            inter_channels, hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size, p_dropout)
        self.dec = GeneratorNSF(
            inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
            upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
            gin_channels=gin_channels, sr=sr, is_half=kwargs["is_half"])
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = torch.nn.Embedding(self.spk_embed_dim, gin_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def forward(self, phone, phone_lengths, pitch, pitchf, y, y_lengths, ds):
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        pitchf = slice_segments2(pitchf, ids_slice, self.segment_size)
        o = self.dec(z_slice, pitchf, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, max_len=None):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec((z * x_mask)[:, :, :max_len], nsff0, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)


class SynthesizerTrnMs256NSFSidNono(torch.nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr=None,
        **kwargs
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.enc_p = TextEncoder256(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            f0=False,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = torch.nn.Embedding(self.spk_embed_dim, gin_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def forward(self, phone, phone_lengths, y, y_lengths, ds):
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, phone, phone_lengths, sid, max_len=None):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec((z * x_mask)[:, :, :max_len], g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)


class Converter():
    sr = 16000
    window = 160
    x_pad, x_query, x_center, x_max = (3, 10, 60, 65) if is_half else (1, 6, 38, 41)

    def __init__(self, sr, if_f0):
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max
        self.if_f0 = if_f0

    def __call__(self, model, net_g, sid, audio, times, f0_up_key, f0_method):
        index = None
        audio_pad = np.pad(audio, (self.window // 2, self.window //2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window): audio_sum += audio_pad[i:i-self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                temp = np.abs(audio_sum[t-self.t_query:t+self.t_query])
                opt_ts.append(t - self.t_query + np.where(temp==temp.min())[0][0])
        s = 0
        audio_opt = []
        t = None
        t1 = time.time()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid = torch.tensor(0, device=device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if self.if_f0 == 1:
            pitch, pitchf = self.get_f0(audio_pad, p_len, f0_up_key, f0_method)
            pitch = torch.tensor(pitch[:p_len], device=device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf[:p_len], device=device).unsqueeze(0).float()
        t2 = time.time()
        times[1] += t2 - t1
        for t in opt_ts:
            t = t // (self.window * self.window)
            if self.if_f0 == 1:
                audio_opt.append(self.vc(
                    model, net_g, sid, audio_pad[s:t+self.t_pad2+self.window],
                    pitch[:,s//self.window:(t+self.t_pad2)//self.window],
                    pitchf[:,s//self.window:(t+self.t_pad2)//self.window],
                    times)[self.t_pad_tgt:-self.t_pad_tgt])
            else:
                audio_opt.append(self.vc(
                    model, net_g, sid, audio_pad[s:t+self.t_pad2+self.window],
                    None, None, times)[self.t_pad_tgt:-self.t_pad_tgt])
            s = t
        if self.if_f0 == 1:
            audio_opt.append(self.vc(
                model, net_g, sid, audio_pad[t:],
                pitch[:,t//self.window:] if t is not None else pitch,
                pitchf[:,t//self.window:] if t is not None else pitchf,
                times)[self.t_pad_tgt:-self.t_pad_tgt])
        else:
            audio_opt.append(self.vc(
                model, net_g, sid, audio_pad[t:],
                None, None, times)[self.t_pad_tgt:-self.t_pad_tgt])
        audio_opt = np.concatenate(audio_opt)
        del pitch, pitchf, sid
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return audio_opt

    def get_f0(self, x, p_len, f0_up_key, f0_method):
        f0_min, f0_max = 50, 1100
        f0_mel_min = 1127 * np.log(1+f0_min/700)
        f0_mel_max = 1127 * np.log(1+f0_max/700)
        if f0_method == "pm":
            f0 = parselmouth.Sound(x, self.sr).to_pitch_ac(
                time_step=self.window/self.sr, voicing_threshold=0.6, pitch_floor=f0_min, pitch_ceiling=f0_max
                ).selected_array["frequency"]
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(f0, [[pad_size, p_len-len(f0)-pad_size]], mode="constant")
        elif f0_method == "harvest":
            f0, t = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
            f0 = signal.medfilt(f0, 3)
        f0 *= pow(2, f0_up_key / 12)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel>0] = (f0_mel[f0_mel>0]-f0_mel_min) * 254 / (f0_mel_max-f0_mel_min) + 1
        f0_mel[f0_mel<=1] = 1
        f0_mel[f0_mel>255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int_)
        return f0_coarse, f0bak


    def vc(self, model, net_g, sid, audio0, pitch, pitchf, times):
        feats = torch.from_numpy(audio0)
        feats = feats.half() if is_half else feats.float()
        if feats.dim() == 2: feats = feats.mean(-1)
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(device).fill_(False)
        inputs = {"source": feats.to(device), "padding_mask": padding_mask, "output_layer": 9}
        t0 = time.time()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0])
        feats = torch.nn.functional.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        t1 = time.time()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:,:p_len]
                pitchf = pitchf[:,:p_len]
        p_len = torch.tensor([p_len], device=device).long()
        with torch.no_grad():
            if pitch is not None and pitchf is not None:
                audio1 = (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0,0]*32768).to(torch.int16).data.cpu().numpy()
            else:
                audio1 = (net_g.infer(feats, p_len, sid)[0][0,0]*32768).to(torch.int16).data.cpu().numpy()
        del feats, p_len, padding_mask
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        t2 = time.time()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1
