import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# from .functions import batch_flating

class MMT(nn.Module):
    def __init__(self, input_dim, rank, modalities, beta, nlayers, droprate=0.1) -> None:
        super(MMT, self).__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.modalities = modalities
        self.n_modals = len(modalities)
        self.beta = beta
        self.nlayers = nlayers

        self.layers = nn.ModuleList([MMTLayer(input_dim, rank, self.n_modals, beta, droprate) 
                                    for _ in range(nlayers)])
        
    
    def forward(self, x):
        out = list(x.values())

        for j in range(self.nlayers):
            out = self.layers[j](out)
                
        out = {
            self.modalities[j]: out[j]
            for j in range(self.n_modals)
        }
        
        return out
        
        
class MMTLayer(nn.Module):
    def __init__(self, input_dim, rank, n_modals, beta, droprate=0.1) -> None:
        super(MMTLayer, self).__init__()
        self.input_dim = input_dim
        self.n_modals = n_modals

        self.attention = MMAttention(input_dim, rank, n_modals, beta)
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        aware = self.attention(x)
        aware = [self.dropout(aware[j]) for j in range(self.n_modals)]

        return aware



class MMAttention(nn.Module):
    def __init__(self, input_dim, rank, n_modals, beta) -> None:
        super(MMAttention, self).__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.n_modals = n_modals
        self.beta = beta

        self.trans_q1 = self.get_trans()
        self.trans_q2 = self.get_trans()
        self.trans_k1 = self.get_trans()
        self.trans_k2 = self.get_trans()
        self.lin_att = nn.ModuleList([Linear(rank * rank, input_dim[j]) for j in range(n_modals)])
        
        
    def get_trans(self):
        return nn.ModuleList([
            Linear(self.input_dim[j], self.rank) 
                for j in range(self.n_modals)
        ])
    
    def forward(self, x):
        """
        Input: List[torch.Tensor[batch_size, length, embed_dim]]
        """
        G_qk = []
        M_qk = []
        att = []
        for j in range(self.n_modals):
            G_q = self.trans_q1[j](x[j]).unsqueeze(-1) * self.trans_q2[j](x[j]).unsqueeze(-2) # mode-1 khatri-rao product
            G_k = self.trans_k1[j](x[j]).unsqueeze(-1) * self.trans_k2[j](x[j]).unsqueeze(-2)
            G_qk.append(G_q * G_k)
            M_qk.append(G_qk[j].mean(dim=1))
        
            
        for j in range(self.n_modals):
            att.append(G_qk[j])
            for l in range(self.n_modals):
                if j == l: continue
                att[j] = torch.einsum('ijkl,ilo->ijko' ,att[j], M_qk[l]) # Tensor contraction
            B, T, R1, R2 = att[j].size()
            att[j] = att[j].view(B, T, R1 * R2)
            att[j] = self.lin_att[j](att[j])
            att[j] = att[j] * x[j] + self.beta * x[j]

        return att

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.lin = nn.Linear(in_features, out_features, bias)
    
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias:
            nn.init.constant_(self.lin.bias, 0.)
    
    def forward(self, x):
        return self.lin(x)
    

def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    
    Args:
        tensor (torch.Tensor): Tensor to generate padding on.   
        padding_idx (int): Position numbers start at padding_idx + 1
        left_pad (bool): Whether to pad from the left or from the right.

    Returns:
        torch.Tensor: Padded output
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(
        make_positions, buf_name).type_as(tensor))
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos,
                     out=getattr(make_positions, buf_name))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[
        :tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - \
            mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx=0, left_pad=0) -> None:
        super(SinusoidalEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()
        self.register_buffer('_float_tensor', torch.FloatTensor(1))
    
    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb
    
    def forward(self, input):
        """Apply PositionalEncodings to Input.
        
        Input is expected to be of size [bsz x seqlen].

        Args:
            input (torch.Tensor): Layer input

        Returns:
            torch.Tensor: Layer output
        """
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.reshape(-1)).reshape((bsz, seq_len, -1)).detach()
    
if __name__ == "__main__":
   module = MMAttention([5, 6, 6], 4, 3, 0.5)

   A = torch.rand((2, 10, 5))
   B = torch.rand((2, 10, 6))
   C = torch.rand((2, 10, 6))

   print(module([A, B, C]))


