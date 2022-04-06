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

class ClassAttender(nn.Module):
  def __init__(self, 
               input_dim,
               hidden_dim,
               n_class):
    super(ClassAttender, self).__init__()
    self.attention = nn.Linear(input_dim, n_class, bias=False)

  def forward(self, x, mask):
    """
    Args :
        x : FloatTensor of size (batch size, seq length, input size)
        mask : FloatTensor of size (batch size, seq length)
    """
    attn_weights = self.attention(x).permute(0, 2, 1)
    attn_weights = attn_weights * mask.unsqueeze(-2)
    attn_weights = torch.where(attn_weights != 0,
                               attn_weights,
                               torch.tensor(-1e10, device=x.device).float())
    
    # (batch size, n class, seq length)
    attn_weights = F.softmax(attn_weights, dim=-1)
    # (batch size, n class, input size)
    attn_applied = torch.bmm(attn_weights, x)

    return attn_applied, attn_weights


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


class GaussianBLSTM(nn.Module):
    def __init__(self,
                 embedding_dim,
                 n_layers=1,
                 n_class=65,
                 input_size=80,
                 ds_ratio=1,
                 bidirectional=True):
        super(GaussianBLSTM, self).__init__()
        self.K = 2 * embedding_dim if bidirectional\
                 else embedding_dim
        self.n_layers = n_layers
        self.n_class = n_class
        self.ds_ratio = ds_ratio
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=embedding_dim,
                           num_layers=n_layers,
                           batch_first=True,
                           bidirectional=bidirectional)
        self.encode = nn.Linear(2 * embedding_dim if bidirectional
                                else embedding_dim,
                                4 * embedding_dim if bidirectional
                                else 2 * embedding_dim)
        self.decode = nn.Linear(2 * embedding_dim if bidirectional
                                else embedding_dim, self.n_class)
        
    def forward(self, x, 
                num_sample=1, 
                masks=None,
                temp=1.,
                return_feat=False):
        ds_ratio = self.ds_ratio
        device = x.device
        if x.dim() < 3: 
          x = x.unsqueeze(0)
        elif x.dim() > 3:
          x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        
        B = x.size(0)
        T = x.size(1)
        if self.bidirectional:
          h0 = torch.zeros((2 * self.n_layers, B, int(self.K // 2)), device=x.device)
          c0 = torch.zeros((2 * self.n_layers, B, int(self.K // 2)), device=x.device)
        else:
          h0 = torch.zeros((self.n_layers, B, self.K), device=x.device)
          c0 = torch.zeros((self.n_layers, B, self.K), device=x.device)          
        embedding, _ = self.rnn(x, (h0, c0))
        statistics = self.encode(embedding) 
        mu = statistics[:, :, :self.K]
        std = F.softplus(statistics[:, :, self.K:]-5, beta=1)
        encoding = self.reparametrize_n(mu, std, num_sample)
        logit = self.decode(encoding)

        if num_sample == 1: pass
        elif num_sample > 1: logit = torch.log(F.softmax(logit, dim=2).mean(0))

        if return_feat:
          return (mu, std), logit, embedding
        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))
        return mu + eps * std
 

class GumbelMarkovModelCell(nn.Module):
  def __init__(self, input_size, num_states):
    """
    Discrete deep Markov model with Gumbel softmax samples. 
    """
    super(GumbelMarkovModelCell, self).__init__()
    self.num_states = num_states
    self.weight_ih = nn.Parameter(
        torch.FloatTensor(input_size, num_states))
    self.weight_hh = nn.Parameter(
        torch.FloatTensor(num_states, num_states))
    self.bias = nn.Parameter(torch.FloatTensor(num_states))

    self.fc = nn.Linear(input_size, num_states) 
    self.trans = nn.Parameter(torch.FloatTensor(num_states, num_states))
    
    self.reset_parameters()

  def reset_parameters(self):
    init.orthogonal_(self.weight_ih.data)
    init.eye_(self.weight_hh)
    init.constant_(self.bias.data, val=0)
    init.eye_(self.trans)
    
  def forward(self, input_, z_0, temp=1.): # TODO Generalize to k-steps
    """
    :param input_: FloatTensor of size (batch, input size), input features
    :param z_0: FloatTensor of size (batch, num. states), sample at the current time step
    :return z_1: FloatTensor of size (batch, num. states), sample for the next time step 
    :return logit_z1_given_z0: FloatTensor of size (batch, num. states)
    """
    batch_size = input_.size(0)
    bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
    logit_z_1 = self.fc(input_)
    wh_b = torch.addmm(bias_batch, z_0, self.weight_hh)
    wi = torch.mm(input_, self.weight_ih)
    g = wh_b + wi

    logit_prior_z1_given_z0 = torch.mm(z_0, self.trans)
    logit_z1_given_z0 = torch.sigmoid(g)*logit_prior_z1_given_z0 +\
                        (1 - torch.sigmoid(g))*logit_z_1
    z_1 = F.gumbel_softmax(logit_z1_given_z0, tau=temp)
    
    return z_1, logit_z1_given_z0


class GumbelMarkovBLSTM(nn.Module):
  def __init__(self, embedding_dim=100, n_layers=1, n_class=65, input_size=80):
    super(GumbelMarkovBLSTM, self).__init__()
    self.K = embedding_dim
    self.bottleneck_dim = 49
    self.n_layers = n_layers
    self.n_class = n_class
    self.rnn = nn.LSTM(input_size=input_size, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
    self.bottleneck = GumbelMarkovModelCell(embedding_dim*2, self.bottleneck_dim)
    self.decode = nn.Linear(self.bottleneck_dim, self.n_class) 

  @staticmethod
  def _forward_bottleneck(cell, input_, length, z_0, n=1, temp=1.):
    device = input_.device
    def expand(v):
      if isinstance(v, Number):
        return torch.Tensor([v]).expand(n, 1)
      else:
        return v.expand(n, *v.size())

    B = z_0.size(0)
    num_states = z_0.size(-1)
    in_size = input_.size()[1:]
    if n != 1:
        z_0 = expand(z_0).contiguous().view(B*n, num_states)
        input_ = expand(input_).contiguous().view(B*n, *in_size) 
        length = expand(length).flatten()
    
    input_ = input_.permute(1, 0, 2)
    max_time = input_.size(0)
    output = []
    logits = []
    for time in range(max_time):
      z_1, logit = cell(input_=input_[time], z_0=z_0, temp=temp)
      mask = (time < length).float().unsqueeze(1).expand_as(z_1).to(device)
      z_1 = z_1*mask + z_0*(1 - mask)
      output.append(z_1)
      logits.append(logit)
      z_0 = z_1
    output = torch.stack(output, 1)
    logits = torch.stack(logits, 1)
    
    if n != 1:
      output = output.view(n, B, max_time, num_states)
      logits = logits.view(n, B, max_time, num_states)
      
    return output, logits

  def forward(self, x, num_sample=1, masks=None, temp=1., return_feat=None):
    if x.dim() < 3:
      x = x.unsqueeze(0)
    elif x.dim() > 3:
      x = x.squeeze(1)
    x = x.permute(0, 2, 1)

    B = x.size(0)   
    T = x.size(1)
    h0 = torch.zeros((2 * self.n_layers, B, self.K)).to(x.device)
    c0 = torch.zeros((2 * self.n_layers, B, self.K)).to(x.device)
    z0 = torch.zeros((B, 49)).to(x.device)
    embed, _ = self.rnn(x, (h0, c0))
    
    if not masks is None:
      length = masks.sum(-1)
    else:
      length = T * torch.ones(B, dtype=torch.int)
    encoding, in_logit = GumbelMarkovBLSTM._forward_bottleneck(
        cell=self.bottleneck, input_=x, length=length, z_0=z0, n=num_sample, temp=temp)
    logit = self.decode(encoding)

    if num_sample != 1:
        logit = torch.log(F.softmax(logit, dim=2).mean(0))
    
    if return_feat:
        if return_feat == 'bottleneck':
            return in_logit, logit, encoding
        elif return_feat == 'rnn':
            return in_logit, logit, embed
    else:
        return in_logit, logit
    
  def weight_init(self):
      pass
        
class VQCPCEncoder(nn.Module):
    def __init__(self, in_channels, channels, n_embeddings, z_dim, c_dim):
        super(VQCPCEncoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, channels, 4, 2, 1, bias=False)
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
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, z_dim),
        )
        self.codebook = VQEmbeddingEMA(n_embeddings, z_dim)
        self.rnn = nn.LSTM(z_dim, c_dim, batch_first=True)

    def encode(self, mel):
        z = self.conv(mel)
        z = self.encoder(z.transpose(1, 2))
        z, indices = self.codebook.encode(z)
        c, _ = self.rnn(z)
        return z, c, indices

    def forward(self, mels):
        z = self.conv(mels)
        z = self.encoder(z.transpose(1, 2))
        z, loss = self.codebook(z)
        c, _ = self.rnn(z)
        return z, c, loss

class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)
        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
            
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss

class VQMLP(torch.nn.Module):
  def __init__(self, 
               embedding_dim,
               n_layers=1,
               n_class=65,
               n_embeddings=40,
               input_size=80):
    super(VQMLP, self).__init__()
    self.K = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.conv = nn.Conv2d(1, embedding_dim,
                          kernel_size=(input_size, 5),
                          stride=(1, 1),
                          padding=(0, 2))
    self.mlp = nn.Sequential(
                 nn.LayerNorm(embedding_dim),
                 nn.Dropout(0.2),
                 nn.ReLU(),
                 nn.Linear(embedding_dim, embedding_dim),
                 nn.LayerNorm(embedding_dim),

                 nn.Dropout(0.2),
                 nn.ReLU(),
                 nn.Linear(embedding_dim, embedding_dim),
                 nn.LayerNorm(embedding_dim),
                 nn.Dropout(0.2),
                 nn.ReLU()
               )
    self.bottleneck = VQEmbeddingEMA(n_embeddings, embedding_dim)
    self.decode = nn.Linear(n_embeddings, self.n_class)
    self.ds_ratio = 1

  def forward(self, x,
              num_sample=1,
              masks=None,
              temp=1.,
              return_feat=False):
    B = x.size(0)
    D = x.size(1)
    T = x.size(2)

    x = self.conv(x.unsqueeze(1)).squeeze(2)  
    x = x.permute(0, 2, 1)
    embed = self.mlp(x)
    x_flat = embed.view(-1, D)
    quantized, loss = self.bottleneck(embed)
    logits = torch.addmm(torch.sum(self.bottleneck.embedding ** 2, dim=1) +
                         torch.sum(x_flat ** 2, dim=1, keepdim=True),
                         x_flat, self.bottleneck.embedding.t(),
                         alpha=2.0, beta=-1.0).view(B, T, -1)

    logits = logits / ((D ** 0.5) * temp)
    encoding = F.softmax(logits, dim=-1)
    if masks is not None:
      quantized = quantized * masks.unsqueeze(2)
      logits = logits * masks.unsqueeze(2)
    out = self.decode(encoding)
    out = torch.cat((out, quantized), dim=2)
    if return_feat:
      return logits, out, encoding, embed 
    else:
      return logits, out
    
  def quantize_loss(self, embed, quantized, masks=None):
    if masks is not None:
      embed = masks.unsqueeze(2) * embed
      masks = masks.unsqueeze(2) * quantized.detach() 
    return self.bottleneck.commitment_cost * F.mse_loss(embed, masks)

class TDSBlock(torch.nn.Module):
    def __init__(self, in_channels, num_features, kernel_size, dropout):
        super(TDSBlock, self).__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        fc_size = in_channels * num_features
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(fc_size, fc_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(fc_size, fc_size),
            torch.nn.Dropout(dropout),
        )
        self.instance_norms = torch.nn.ModuleList(
            [
                torch.nn.InstanceNorm1d(fc_size, affine=True),
                torch.nn.InstanceNorm1d(fc_size, affine=True),
            ]
        )

    def forward(self, inputs, return_feat=False):
        # inputs shape: [B, C * H, W]
        B, CH, W = inputs.shape
        C, H = self.in_channels, self.num_features
        outputs = self.conv(inputs.view(B, C, H, W)).view(B, CH, W) + inputs
        outputs = self.instance_norms[0](outputs)

        outputs = self.fc(outputs.transpose(1, 2)).transpose(1, 2) + outputs
        outputs = self.instance_norms[1](outputs)

        # outputs shape: [B, C * H, W]
        return outputs


class GumbelTDS(torch.nn.Module):
    def __init__(self, 
                 tds_groups=[
                   { "channels" : 4, "num_blocks" : 5 },
                   { "channels" : 8, "num_blocks" : 5 },
                   { "channels" : 16, "num_blocks" : 5 }],
                 kernel_size=5, 
                 dropout=0.2,
                 n_class=258,
                 n_gumbel_units=49,
                 input_size=80):
        super(GumbelTDS, self).__init__()
        modules = []
        in_channels = input_size
        for tds_group in tds_groups:
            # add downsample layer:
            out_channels = input_size * tds_group["channels"]
            modules.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        stride=tds_group.get("stride", 2),
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.InstanceNorm1d(out_channels, affine=True),
                ]
            )
            for _ in range(tds_group["num_blocks"]):
                modules.append(
                    TDSBlock(tds_group["channels"], input_size, kernel_size, dropout)
                )
            in_channels = out_channels
        self.tds = torch.nn.Sequential(*modules)
        self.bottleneck = torch.nn.Linear(in_channels, n_gumbel_units)
        self.linear = torch.nn.Linear(n_gumbel_units, n_class)
        self.ds_ratio = 2 ** len(tds_groups)

    def forward(self, inputs,
                masks=None,
                num_sample=1,
                temp=1.,
                return_feat=False):
        # inputs shape: [B, H, W]
        embeddings = self.tds(inputs)
        embeddings = embeddings.permute(0, 2, 1)
        in_logits = self.bottleneck(embeddings)

        if masks is not None:
          in_logits = in_logits * masks.unsqueeze(2)
        encodings = self.reparametrize_n(in_logits,
                                         n=num_sample,
                                         temp=temp)

        # outputs shape: [B, W, output_size]
        out_logits = self.linear(encodings)
        if num_sample > 1:
          out_logits = out_logits.mean(0) 

        if return_feat:
          if masks is not None:
            embeddings = (embeddings * masks.unsqueeze(2)).sum(1)
            embeddings = embeddings / masks.sum(-1, keepdim=True)
          else:
            embeddings = embeddings.sum(-2)
          return in_logits, out_logits, encodings, embeddings 
        return in_logits, out_logits 
    
    def reparametrize_n(self, x, n=1, temp=1.):
      def expand(v):
        if v.ndim < 1:
          return torch.Tensor([v]).expand(n, 1)
        else:
          return v.expand(n, *v.size())

      if n != 1:
          x = expand(x)
      encoding = F.gumbel_softmax(x, tau=temp)

      return encoding

def masked_kl_div(input, target, mask,
                  log_input=False,
                  reduction="mean"):
  EPS = 1e-10
  # (B, *, D)
  if log_input:
    KL = torch.exp(target) * (target - input)
  else:
    KL = torch.exp(target) * (target - torch.log(input))

  if not reduction:
    return KL.sum(-1)

  if mask is not None:
    loss = (KL.sum(-1) * mask).mean(0).sum()
  else:
    loss = KL.mean(0).sum()

  return loss  

def masked_js_div(input, target, mask,
                  log_input=False,
                  reduction="mean"):
  if log_input:
    m = (torch.exp(input) + torch.exp(target)) / 2.
  else:
    m = (input + torch.exp(target)) / 2.
  loss = masked_kl_div(m, target, mask, reduction=reduction)\
         + masked_kl_div(m, input, mask, reduction=reduction)
  return loss

def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
