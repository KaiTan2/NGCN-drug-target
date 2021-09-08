import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers import GAT, InnerProductDecoder
import numpy as np

class VGAE(nn.Module):
    def __init__(self, conv_para, deconv_para, fc1_num, fc2_num, latent_size, training):
        super(VGAE, self).__init__()
        # self.lstm = LSTMEncoder1(emb_size, hidden_size, layers_num, dropout)
        self.training = training
        #Encoder
        self.conv_encoder = nn.ModuleList(nn.Conv2d(in_channels=para[0], out_channels=para[1], kernel_size=para[2], stride=para[3], padding=para[4])
                                          for para in conv_para)
        self.fc1 = nn.Linear(fc1_num[0], fc1_num[1])
        # Latent space
        self.fc21 = nn.Linear(fc2_num[0], fc2_num[1])
        self.fc22 = nn.Linear(fc2_num[0], fc2_num[1])

        # Decoder
        self.latent_size = latent_size
        self.fc3 = nn.Linear(fc2_num[1], fc2_num[0])
        self.fc4 = nn.Linear(fc1_num[1], fc1_num[0])
        self.deconv_decoder = nn.ModuleList(nn.ConvTranspose2d(in_channels=para[0], out_channels=para[1], kernel_size=para[2], stride=para[3], padding=para[4])
                                          for para in deconv_para)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        conv_out = x
        for conv_layer in self.conv_encoder:
            conv_out = self.relu(conv_layer(conv_out))
        conv_out = conv_out.view(conv_out.size(0), -1)
        h1 = self.relu(self.fc1(conv_out))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        out = out.view(out.size(0), self.latent_size[0], self.latent_size[1], self.latent_size[2])
        for deconv_layer in self.deconv_decoder[:-1]:
            out = self.relu(deconv_layer(out))

        out = self.sigmoid(self.deconv_decoder[-1](out))
        return out


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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
