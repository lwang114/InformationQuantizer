import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from utils.utils import cuda


class InfoQuantizer(nn.Module):
  def __init__(self, 
               in_channels, 
               channels, 
               n_embeddings, 
               z_dim, 
               init_embedding=None, 
               use_conv=False, 
               conv_width=5):
    super(InfoQuantizer, self).__init__()
    self.use_conv = use_conv
    self.conv = nn.Conv1d(in_channels, channels, conv_width, 1, int(conv_width // 2))
    if self.use_conv:
      print('Use convolutional input layer')
      self.encoder = nn.Sequential(
          nn.LayerNorm(channels),
          nn.ReLU(True),
          nn.Linear(channels, channels, bias=False),
          nn.LayerNorm(channels),
          nn.ReLU(True),
          nn.Linear(channels, channels, bias=False),
          nn.LayerNorm(channels),
          nn.ReLU(True),
          nn.Linear(channels, channels, bias=False),
          nn.LayerNorm(channels),
          nn.ReLU(True),
          nn.Linear(channels, z_dim),
      )
    else:
      self.encoder = nn.Sequential(
          nn.Linear(in_channels, channels, bias=False),
          nn.LayerNorm(channels),
          nn.ReLU(True),
          nn.Linear(channels, channels, bias=False),
          nn.LayerNorm(channels),
          nn.ReLU(True),
          nn.Linear(channels, channels, bias=False),
          nn.LayerNorm(channels),
          nn.ReLU(True),
          nn.Linear(channels, channels, bias=False),
          nn.LayerNorm(channels),
          nn.ReLU(True),
          nn.Linear(channels, z_dim),
      )
    self.codebook = IQEmbeddingEMA(n_embeddings, z_dim, init_embedding=init_embedding)
    self.ds_ratio = False

  def encode(self, x, masks=None):
    if self.use_conv:
      x = self.conv(x.permute(0, 2, 1))
      x = x.permute(0, 2, 1)
    z = self.encoder(x) 
    p = F.log_softmax(z, dim=-1)
    q, indices = self.codebook.encode(p, masks=masks)    
    return z, q, indices

  def forward(self, x, masks=None):
    if self.use_conv:
      x = self.conv(x.permute(0, 2, 1))
      x = x.permute(0, 2, 1)
    z = self.encoder(x)
    p = F.log_softmax(z, dim=-1)
    q, loss = self.codebook(p, masks=masks)
    return z, q, loss


class MultiHeadInfoQuantizer(nn.Module):
  def __init__(self, 
               in_channels, 
               channels, 
               n_embeddings, 
               z_dims, 
               decay=0.999,
               use_rnn=False):
    super(MultiHeadInfoQuantizer, self).__init__()
    self.in_channels = in_channels
    self.use_rnn = use_rnn
    if use_rnn:
      self.encoder = nn.LSTM(input_size=in_channels,
                             hidden_size=channels,
                             num_layers=1,
                             batch_first=True,
                             bidirectional=False)
    else:
      self.encoder = nn.Sequential(
          nn.Linear(in_channels, channels, bias=False),
          nn.LayerNorm(channels),
          nn.ReLU(True),
          #nn.Linear(channels, channels, bias=False),
          #nn.LayerNorm(channels),
          #nn.ReLU(True),
          #nn.Linear(channels, channels, bias=False),
          #nn.LayerNorm(channels),
          #nn.ReLU(True),
          #nn.Linear(channels, channels, bias=False),
          #nn.LayerNorm(channels),
          #nn.ReLU(True),
          nn.Linear(channels, sum(z_dims)),
      )
    self.z_dims = z_dims
    
    init_embedding = []
    for z_dim in z_dims:
        alpha = [100] * z_dim
        init_embedding.append(torch.Tensor(np.random.dirichlet(alpha, size=(n_embeddings,))))
    init_embedding = torch.cat(init_embedding, dim=-1)
    init_embedding = torch.FloatTensor(init_embedding)
    self.codebook = IQEmbeddingEMA(n_embeddings, sum(z_dims), 
                                   init_embedding=init_embedding, 
                                   decay=decay)
    # XXX self.codebook = IQEmbeddingEMA(n_embeddings, z_dims[0], 
    #                               decay=decay)
    self.ds_ratio = False

  def encode(self, x, masks=None):
    device = x.device
    batch_size = x.size(0)
    if self.use_rnn:
      h0 = torch.zeros((1, batch_size, self.in_channels), device=device)
      c0 = torch.zeros((1, batch_size, self.in_channels), device=device)
      z, _ = self.encoder(x, (h0, c0))    
    else:
      z = self.encoder(x) 
    p = []
    start_idx = 0
    
    for z_dim in self.z_dims: # [self.z_dims[0]]: XXX
        p.append(F.log_softmax(z[:, :, start_idx:start_idx+z_dim], dim=-1))
        start_idx += z_dim
    p = torch.cat(p, dim=-1)
    q, indices = self.codebook.encode(p, masks=masks)    
    return z, q, indices

  def forward(self, x, masks=None):
    device = x.device
    batch_size = x.size(0)
    if self.use_rnn:
      h0 = torch.zeros((1, batch_size, self.in_channels), device=device)
      c0 = torch.zeros((1, batch_size, self.in_channels), device=device)
      z, _ = self.encoder(x, (h0, c0))    
    else:
      z = self.encoder(x) 
    p = []
    start_idx = 0
    for z_dim in self.z_dims: # [self.z_dims[0]]: XXX
        p.append(F.log_softmax(z[:, :, start_idx:start_idx+z_dim], dim=-1))
        start_idx += z_dim
    p = torch.cat(p, dim=-1)
    q, loss = self.codebook(p, masks=masks)
    return z, q, loss

class IQEmbeddingEMA(nn.Module):
  def __init__(self, n_embeddings, 
               embedding_dim, 
               commitment_cost=0.25, 
               decay=0.999, 
               epsilon=1e-5,
               div_type="kl",
               init_embedding=None):
    super(IQEmbeddingEMA, self).__init__()
    self.commitment_cost = commitment_cost
    self.decay = decay
    self.epsilon = epsilon
    self.div_type = div_type
    if not self.div_type in ['kl', 'js']:
      raise ValueError(f"Divergence type {self.div_type} not defined")

    if init_embedding is None:
      alpha = [100]*n_embeddings # 10 ** np.linspace(-1.4, 1.4, n_embeddings)
      embedding = torch.stack(
                    [torch.Tensor(np.random.dirichlet([alpha[k]]*embedding_dim))
                    for k in range(n_embeddings)]
                  ) # Sample a codebook of pmfs
    else:
      embedding = init_embedding
    self.register_buffer("embedding", embedding)
    self.register_buffer("ema_count", torch.ones(n_embeddings))
    self.register_buffer("ema_weight", self.embedding.clone())

  def encode(self, x, masks=None):
    M, D = self.embedding.size()
    x_flat = x.detach().reshape(-1, D)
    mask_flat = masks.reshape(-1, 1)

    if self.div_type == "kl":
      divergences = masked_kl_div(self.embedding.unsqueeze(0),
                                  x_flat.unsqueeze(-2),
                                  mask=mask_flat,
                                  reduction=None)
    elif self.div_type == 'js':
      divergences = masked_js_div(self.embedding.unsqueeze(0),
                                  x_flat.unsqueeze(-2),
                                  mask=mask_flat,
                                  reduction=None)

    indices = torch.argmin(divergences.float(), -1)
    quantized = F.embedding(indices, self.embedding)
    quantized = quantized.view_as(x)
    return quantized, indices.view(x.size(0), x.size(1))

  def forward(self, x, masks=None):
    M, D = self.embedding.size()
    x_flat = x.detach().reshape(-1, D)
    mask_flat = None
    if masks is not None:
      mask_flat = masks.reshape(-1, 1)

    if self.div_type == "kl":
      divergences = masked_kl_div(self.embedding.unsqueeze(0),
                                  x_flat.unsqueeze(-2),
                                  mask=mask_flat,
                                  reduction=None)
    elif self.div_type == "js":
      divergences = masked_js_div(self.embedding.unsqueeze(0),
                                  x_flat.unsqueeze(-2),
                                  mask=mask_flat,
                                  reduction=None)

    indices = torch.argmin(divergences.float(), -1)
    encodings = F.one_hot(indices, M).float()
    if mask_flat is not None:
      encodings = encodings * mask_flat
    quantized = F.embedding(indices, self.embedding)
    quantized = quantized.view_as(x)
    
    if self.training:
      self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
      dw = torch.matmul(encodings.t(), torch.exp(x_flat))

      self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
      self.embedding = (self.ema_weight + self.epsilon / M) / (self.ema_count.unsqueeze(-1) + self.epsilon)

    if self.div_type == "kl":
      e_latent_loss = masked_kl_div(quantized.detach(), x, mask=masks)
    elif self.div_type == "js":
      divergences = masked_js_div(quantized.detach(), x, mask=masks)

    loss = self.commitment_cost * e_latent_loss

    quantized = x + (quantized - x).detach()
   
    avg_probs = torch.mean(encodings, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))) 
    return quantized, loss

class MLP(nn.Module):
  def __init__(self,
               embedding_dim,
               n_layers=1,
               n_class=65,
               input_size=80,
               max_seq_len=100,
               context_width=5,
               position_dependent=False):
    super(MLP, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.position_dependent = position_dependent
    self.ds_ratio = 1
    in_channels = input_size
    channels = embedding_dim

    self.mlp = nn.Sequential(
        nn.Linear(in_channels, channels, bias=False),
        nn.LayerNorm(channels),
        nn.ReLU(True),
        nn.Linear(channels, channels, bias=False),
        nn.LayerNorm(channels),
        nn.ReLU(True),
        nn.Linear(channels, channels, bias=False),
        nn.LayerNorm(channels),
        nn.ReLU(True),
        nn.Linear(channels, channels, bias=False),
        nn.LayerNorm(channels),
        nn.ReLU(True)
    )

    '''
    self.mlp = nn.Sequential(
                 # nn.Linear(embedding_dim, embedding_dim),
                 nn.Linear(input_size, embedding_dim),
                 nn.ReLU(),
                 nn.Linear(embedding_dim, embedding_dim),
                 nn.ReLU(),
               )
    '''
    if position_dependent:
      self.decode = nn.Linear(embedding_dim * round(max_seq_len // self.ds_ratio),
                              self.n_class,
                              bias=False)
    else:
      self.decode = nn.Linear(embedding_dim,
                              self.n_class,
                              bias=False)

    
  def forward(self, x,
              masks=None,
              return_feat=False):
    B = x.size(0)
    embed = self.mlp(x)
    if self.position_dependent:
      out = self.decode(embed.view(B, -1))
    else:
      out = self.decode(embed).sum(-2)

    if return_feat:
      return out, embed
    else:
      return out
 
class GumbelMLP(nn.Module):
  def __init__(self,
               embedding_dim,
               n_layers=1,
               n_class=65,
               n_gumbel_units=40,
               input_size=80,
               max_len=100,
               position_dependent=False):
    super(GumbelMLP, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.ds_ratio = 1
    in_channels = input_size
    channels = embedding_dim

    '''
    self.mlp = nn.Sequential(
        nn.Linear(in_channels, channels, bias=False),
        nn.LayerNorm(channels),
        nn.ReLU(True),
        nn.Linear(channels, channels, bias=False),
        nn.LayerNorm(channels),
        nn.ReLU(True),
        nn.Linear(channels, channels, bias=False),
        nn.LayerNorm(channels),
        nn.ReLU(True),
        nn.Linear(channels, channels, bias=False),
        nn.LayerNorm(channels),
        nn.ReLU(True)
    )
    '''
     
    self.mlp = nn.Sequential(
                 nn.Linear(input_size, embedding_dim),
                 nn.ReLU(),
                 # nn.Linear(embedding_dim, embedding_dim),
                 # nn.ReLU(),
                 nn.Linear(embedding_dim, embedding_dim),
                 nn.ReLU(),
               )
    
    self.bottleneck = nn.Linear(embedding_dim, n_gumbel_units)
    self.decoders = nn.ModuleList([nn.Linear(n_gumbel_units, 
                                             self.n_class) 
                                  for _ in range(round(max_len // self.ds_ratio))])
    self.position_dependent = position_dependent
    self.Nd = len(self.decoders)

  def forward(self, x, 
              num_sample=1,
              masks=None,
              temp=1.,
              return_feat=False):
    B = x.size(0)
    logits, encoding, embed = self.encode(x, 
                                          masks=masks,
                                          n=num_sample,
                                          temp=temp)
    out = self.decode(encoding,
                      masks=masks,
                      n=num_sample)
    if return_feat:
      return logits, out, encoding, embed
    else:
      return logits, out

  def encode(self, x, masks=None, n=1, temp=1):
    device = x.device
    embed = self.mlp(x)
    logits = self.bottleneck(embed)  
    if masks is not None:
      logits = logits * masks.unsqueeze(-1)
    encoding = self.reparametrize_n(logits,
                                    masks=masks,
                                    n=n, 
                                    temp=temp)
    return logits, encoding, embed

  def decode(self, encoding, masks=None, n=1):
    device = encoding.device
    if n > 1:
      if self.position_dependent:
        out = [self.decoders[i](encoding[:, :, i]) for i in range(self.Nd)]
      else:
        out = [self.decoders[0](encoding[:, :, i]) for i in range(self.Nd)]
      
      # (n samples, batch size, max n segments, n word classes)
      out = torch.stack(out, dim=2)
      # (batch size, max n segments, n word classes)
      out = out.mean(0)
    else:
      if self.position_dependent:
        out = [self.decoders[i](encoding[:, i]) for i in range(self.Nd)]
        out = torch.stack(out, dim=1)
      else:
        out = self.decoders[0](encoding)
      # (batch size, max n segments, n word classes)
    # out = F.log_softmax(out, dim=-1) # XXX

    if masks is not None:
      out = out * masks.unsqueeze(-1)
    return out

  def reparametrize_n(self, x, masks=None, n=1, temp=1.):
    # reference :
    # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
    # param x: FloatTensor of size (batch size, num. frames, num. classes) 
    # param n: number of samples
    # return encoding: FloatTensor of size (n, batch size, num. frames, num. classes)
    def expand(v):
        if v.ndim < 1:
            return torch.Tensor([v]).expand(n, 1)
        else:
            return v.expand(n, *v.size())

    if n != 1 :
        x = expand(x)
    encoding = F.gumbel_softmax(x, tau=temp)
    if masks is not None:
      encoding = encoding * masks.unsqueeze(-1)
    return encoding         


class BLSTM(nn.Module):
  def __init__(self, 
               embedding_dim, 
               n_layers=1, 
               n_class=65,
               input_size=80, 
               ds_ratio=1,
               bidirectional=True,
               decoder=None):
    super(BLSTM, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.ds_ratio = ds_ratio
    self.bidirectional = bidirectional
    self.rnn = nn.LSTM(input_size=input_size,
                       hidden_size=embedding_dim,
                       num_layers=n_layers,
                       batch_first=True,
                       bidirectional=bidirectional)
    if decoder is None:
      self.decode = nn.Linear(2 * embedding_dim if bidirectional
                              else embedding_dim, self.n_class)
    else:
      self.decode = decoder

  def forward(self, x, 
              return_feat=False):
    device = x.device
    ds_ratio = self.ds_ratio
    
    B = x.size(0)
    T = x.size(1)
    if self.bidirectional:
      h0 = torch.zeros((2 * self.n_layers, B, self.K), device=device)
      c0 = torch.zeros((2 * self.n_layers, B, self.K), device=device)
    else:
      h0 = torch.zeros((self.n_layers, B, self.K), device=device)
      c0 = torch.zeros((self.n_layers, B, self.K), device=device)
       
    embed, _ = self.rnn(x, (h0, c0))
    logit = self.decode(embed)

    if return_feat:
        L = ds_ratio * (T // ds_ratio)
        embedding = embed[:, :L].view(B, int(L // ds_ratio), ds_ratio, -1)
        embedding = embedding.sum(-2)
        return logit, embedding
    return logit     


class GumbelBLSTM(nn.Module):
  def __init__(self, 
               embedding_dim, 
               n_layers=1, 
               n_class=65, 
               n_gumbel_units=49,
               input_size=80, 
               ds_ratio=1,
               bidirectional=True,
               max_len=100,
               position_dependent=False):
    super(GumbelBLSTM, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.ds_ratio = ds_ratio
    self.bidirectional = bidirectional
    self.rnn = nn.LSTM(input_size=input_size,
                       hidden_size=embedding_dim,
                       num_layers=n_layers,
                       batch_first=True,
                       bidirectional=bidirectional)
    self.bottleneck = nn.Sequential(nn.Linear(2*embedding_dim+input_size if bidirectional 
                                              else embedding_dim+input_size, 
                                              embedding_dim),
                                    nn.ReLU(),
                                    nn.Linear(embedding_dim,
                                              n_gumbel_units))
    self.decoders = nn.ModuleList([nn.Linear(n_gumbel_units, 
                                             self.n_class)
                                  for _ in range(round(max_len // self.ds_ratio))])
    self.position_dependent = position_dependent
    self.Nd = round(max_len // self.ds_ratio)

  def forward(self, x, 
              num_sample=1, 
              masks=None, 
              temp=1., 
              return_feat=False):
    ds_ratio = self.ds_ratio
    logits, encoding, embed = self.encode(x, 
                                          masks=masks,
                                          n=num_sample,
                                          temp=temp)
    out = self.decode(encoding,
                      masks=masks,
                      n=num_sample)

    if return_feat:
      return logits, out, encoding, embed
    return logits, out
  
  def encode(self, x, masks, n, temp):
    device = x.device
    EPS = 1e-10
    B = x.size(0)
    N = x.size(1) # Max. number of segments
    T = x.size(2) # Max. segment length
    if self.bidirectional:
      h0 = torch.zeros((2 * self.n_layers, B * N, self.K))
      c0 = torch.zeros((2 * self.n_layers, B * N, self.K))
    else:
      h0 = torch.zeros((self.n_layers, B * N, self.K))
      c0 = torch.zeros((self.n_layers, B * N, self.K))

    if torch.cuda.is_available():
      h0 = h0.cuda()
      c0 = c0.cuda()
       
    embed_size = (B, N, T, self.K)
    embed, _ = self.rnn(x.view(B*N, T, -1), (h0, c0))
    embed = embed.view(B, N, T, -1)
    embed = torch.cat([embed, x], dim=-1) # Highway connection
    if masks is not None:
      embed = embed * masks.unsqueeze(-1)
    # (B, N, K)
    embed = embed.sum(-2) / (masks.sum(-1, keepdim=True) + EPS)
    
    logits = self.bottleneck(embed)
    if masks is not None:
      masks_1d = torch.where(masks.sum(-1) > 0,
                             torch.tensor(1., device=device),
                             torch.tensor(0., device=device))
      logits = logits * masks_1d.unsqueeze(-1)
    encoding = self.reparametrize_n(logits, n, temp)
    return logits, encoding, embed

  def decode(self, encoding, masks, n):
    device = encoding.device
    if n > 1:
      if self.position_dependent:
        out = [self.decoders[i](encoding[:, :, i]) for i in range(self.Nd)]
      else:
        out = [self.decoders[0](encoding[:, :, i]) for i in range(self.Nd)]
      out = torch.stack(out, dim=2).mean(0)
    else:
      if self.position_dependent:
        out = [self.decoders[i](encoding[:, i]) for i in range(self.Nd)]
      else:
        out = [self.decoders[0](encoding[:, i]) for i in range(self.Nd)]
      # (B, N, n word class)
      out = torch.stack(out, dim=1)

    if masks is not None:
      # (B, N)
      masks_1d = torch.where(masks.sum(-1) > 0,
                             torch.tensor(1., device=device),
                             torch.tensor(0., device=device))

    return out

  def reparametrize_n(self, x, n=1, temp=1.):
      # reference :
      # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
      # param x: FloatTensor of size (batch size, num. frames, num. classes) 
      # param n: number of samples
      # return encoding: FloatTensor of size (n, batch size, num. frames, num. classes)
      def expand(v):
          if v.ndim < 1:
              return torch.Tensor([v]).expand(n, 1)
          else:
              return v.expand(n, *v.size())

      if n != 1:
          x = expand(x)
      encoding = F.gumbel_softmax(x, tau=temp)

      return encoding

  def weight_init(self):
      pass
