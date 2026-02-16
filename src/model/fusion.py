import torch
import torch.nn as nn

class Fusion(nn.Module):
    def __init__(self, emb_dim, bsg_types, trbg_types):
        super(Fusion, self).__init__()
        self.emb_dim = emb_dim
        self.bsg_types = bsg_types
        self.trbg_types = trbg_types
        self.proj_layers = nn.ModuleList([nn.Linear(emb_dim*2, 1),
                                  nn.Linear(emb_dim*2, 1)])
        
    def reset_parameters(self):
        for layer in self.proj_layers:
            nn.init.xavier_uniform_(layer.weight)
    def forward(self, emb_dict):
        key = emb_dict['buy'] # [N, D]
        for i, behavior_types in enumerate([self.bsg_types, self.trbg_types]):
            key = key.unsqueeze(1).repeat(1, len(behavior_types), 1) # [N, |B|, D]
            query_emb = torch.stack([emb_dict[behavior] for behavior in behavior_types], dim=1) # [N, |B|, D]
            concat_emb = torch.concat([key, query_emb], dim=2) # [N, |B|, 2D]
            attention = self.proj_layers[i](concat_emb).softmax(dim=1) # [N, |B|, 1]
            key = (attention * query_emb).sum(dim=1) # [N, D]
        
        return key
