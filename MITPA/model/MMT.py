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
    


    
if __name__ == "__main__":
   module = MMAttention([5, 6, 6], 4, 3, 0.5)

   A = torch.rand((2, 10, 5))
   B = torch.rand((2, 10, 6))
   C = torch.rand((2, 10, 6))

   print(module([A, B, C]))


