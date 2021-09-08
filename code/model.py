import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers import GAT, InnerProductDecoder
import numpy as np
#自编码器
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.num_input = input_dim
        self.num_hidden = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


#图注意力自编码器
class GATModelAE(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 out_dim,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,):
        super(GATModelAE, self).__init__()
        self.GAT = GAT(g,
                     num_layers,
                     in_dim,
                     num_hidden,
                     out_dim,
                     heads,
                     activation,
                     feat_drop,
                     attn_drop,
                     negative_slope,
                     residual)
        self.g = g
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.num_hidden = num_hidden
        self.out_dim = out_dim
        self.heads = heads
        self.activation = activation
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual
        self.build()

    def build(self):
        self.encoder = self.GAT
        self.decoder = InnerProductDecoder(self.feat_drop, act=lambda x: x)

    def forward(self, input):
        self.embeddings = self.encoder(input)
        self.z_mean = self.embeddings
        self.reconstructions = self.decoder(self.embeddings)
        return self.z_mean, self.reconstructions


###################################################################
#各个网络特征注意力机制
def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze()

def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()

def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)
