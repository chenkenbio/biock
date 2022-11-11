
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import warnings

try:
    from performer_pytorch import SelfAttention as PerformerAttention
except ImportError:
    warnings.warn("Package performer_pytorch has not been installed, skipped")



class SelfAttentionEmbedding(nn.Module):
    """
    based on https://github.com/kaushalshetty/Structured-Self-Attention
    """
    def __init__(self, d_model: int, da: int, r: int, skip_max: bool=False, **kwargs):
        super(SelfAttentionEmbedding, self).__init__()
        self.da = da
        self.r = r
        self.att_first = nn.Linear(d_model, da)
        self.att_first.bias.data.fill_(0)
        self.att_second = nn.Linear(da, r)
        self.att_second.bias.data.fill_(0)
        self.skip_max = skip_max
        self.out_dim = d_model if skip_max else (2 * d_model)
    
    def forward(self, seq: Tensor, mask: Tensor=None, require_penalty=False) -> Tensor:
        """
        input
        ------
        seq : (B, S, E)
        mask : (B, S) or (B, S, E)
        
        Return
        ------ 
        embedding : (B, r, E)
        """
        bs, device  = seq.size(0), seq.device
        att = torch.tanh(self.att_first(seq))  #(B, S, E) -> (B, S, da)
        att = self.att_second(att) # (B, S, da) -> (B, S, r) 
        if mask is not None:
            if mask.ndim < att.ndim:
                mask = mask.unsqueeze(2)
            att.masked_fill_(mask, float("-inf"))

        att = F.softmax(att, dim=1).transpose(1, 2) # -> (B, r, S)

        seq = torch.matmul(att, seq) # (B, r, S) x (B, S, E) -> (B, r, E)
        if self.skip_max:
            seq = seq.mean(dim=1).view(bs, -1)
        else:
            seq = torch.cat((
                seq.mean(dim=1).view(bs, -1),
                seq.max(dim=1)[0].view(bs, -1)
            ), dim=1)
        if require_penalty:
            identity = torch.eye(self.r).to(device) # (r, r)
            identity = Variable(identity.unsqueeze(0).expand(bs, self.r, self.r)) # (B, r, r)
            penal = torch.linalg.norm(torch.matmul(att, att.transpose(1, 2)) - identity).mean() # different from the original implementation
            return seq, penal
        else:
            return seq


class PerformerEncoderLayer(nn.Module):
    def __init__(self, 
            d_model: int, nhead: int, dim_head: int, dim_feedforward: int, \
            dropout: float=0.1, \
            activation="relu", \
            layer_norm_eps=1e-5, \
            batch_first: bool=False,
            **kwargs):
        super(PerformerEncoderLayer, self).__init__()
        self.batch_first = batch_first
        self.self_attn = PerformerAttention(
            dim=d_model, 
            causal=False, 
            heads=nhead, 
            dim_head=dim_head, 
            dropout=dropout
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
    
    def forward(self, src: Tensor) -> Tensor:
        r"""
        Args:
            src: input sequence
        Shape: 
        src: :math: `(S, N, E)` if `batch_first=False` or `(N, S, E)` if `batch_first=True`
        """
        if not self.batch_first:
            src = src.transpose(0, 1)
        src2 = self.self_attn(src)
        src = src + src2
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + src2
        src = self.norm2(src)
        return src


def _get_clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PerformerEncoder(nn.Module):
    def __init__(self, \
            encoder_layer: PerformerEncoderLayer, \
            num_layers: int, norm=None):
        super(PerformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src
        """
        output = src
        for mod in self.layers:
            output = mod(output)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output



class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        x : (B, S, E) 
        """
        ## pos_table: (1, S, E)
        return x + self.pos_table[:, :x.size(1)].clone().detach()