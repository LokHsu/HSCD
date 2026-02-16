import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_conv import GraphConvLayer
from .fusion import Fusion

def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.transpose(0,1)) - 1) / sigma)    ### real_kernel

def hsic(Kx, Ky, m):
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - \
        2 * torch.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2

def bpr_loss(p_score, n_score):
    loss = -torch.log(1e-10 + torch.sigmoid(p_score - n_score))
    return loss.mean()


class GenLoss(nn.Module):
    def __init__(self, n_items, edge_dict_raw):
        super(GenLoss, self).__init__()
        self.n_items = n_items
        self.edge_dict_raw = edge_dict_raw
    
    def sample_positive(self, edges, ratio=0.05):
        u, i = edges
        k = max(1, int(u.size(0) * ratio))
        idx = torch.randperm(u.size(0), device=u.device)[:k]
        return u[idx], i[idx]
    
    def sample_negative(self, u):
        j = torch.randint(
            1, self.n_items + 1,
            (u.size(0),),
            device=u.device
        )
        return j

    def forward(self, pre_beh, user_embs, item_embs, alpha = 0.1):
        edge_u, edge_i = self.edge_dict_raw[pre_beh]

        u, i = self.sample_positive((edge_u, edge_i))
        pos_scores = (user_embs[u] * item_embs[i]).sum(dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-10)

        j = self.sample_negative(u)
        neg_scores = (user_embs[u] * item_embs[j]).sum(dim=1)
        neg_loss = -alpha * torch.log(1 - torch.sigmoid(neg_scores) + 1e-10)

        total_loss = (pos_loss + neg_loss).sum()
        return total_loss

"""EmbLoss, regularization on embeddings"""
class EmbLoss(nn.Module):
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.pow(
                input=torch.norm(embedding, p=self.norm), exponent=self.norm
            )
        emb_loss /= embeddings[-1].shape[0]
        emb_loss /= self.norm
        return emb_loss


class HSCD(nn.Module):
    def __init__(self, data, emb_dim, beta, sigma, alpha):
        super(HSCD, self).__init__()
        self.edge_dict = data['edge_dict']
        
        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.temperature = 1
        self.dropout = nn.Dropout(0.1)
      
        self.bsg_types = data['bsg_types']
        self.tcb_types = data['tcb_types'] # target-complemented behaviors
        self.tib_types = data['tib_types'] # target-intersected behaviors
        self.trbg_types = self.tcb_types + self.tib_types
        self.total_behaviors = ['ubg'] + self.bsg_types + self.trbg_types

        self.fusion = Fusion(emb_dim, self.bsg_types, self.trbg_types)
        self.ce_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()
        self.gen_loss = GenLoss(self.n_items, data['edge_dict_raw'])
        
        self.weights = [1.0, alpha, beta, 0.0001]
        self.sigma = sigma

        self.user_embedding = nn.Embedding(self.n_users+1, emb_dim, padding_idx=0) # index 0 is padding
        self.item_embedding = nn.Embedding(self.n_items+1, emb_dim, padding_idx=0) # index 0 is padding
        
        self.convs = nn.ModuleDict()
        for behavior_type in self.total_behaviors:
            if behavior_type in self.tcb_types:
                self.convs[behavior_type] = nn.ModuleList([GraphConvLayer(emb_dim, emb_dim, 'gcn') for _ in range(3)])
            else:
                self.convs[behavior_type] = nn.ModuleList([GraphConvLayer(emb_dim, emb_dim, 'gcn') for _ in range(1)])
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.fusion.reset_parameters()

    def propagate(self, x, edge_index, behavior_type):
        result = [x]
        for i, conv in enumerate(self.convs[behavior_type]):
            x = conv(x, edge_index)
            x = F.normalize(x, dim=1)
            result.append(x)
        result = torch.stack(result, dim=1)
        x = result.sum(dim=1)
        return x

    
    def forward(self):
        edge_dict = self.edge_dict
        emb_dict = dict()

        ## Unified behavior graph aggregation ##
        init_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        emb_dict['init'] = init_emb
        ubg_emb = self.propagate(init_emb, edge_dict['ubg'], 'ubg')
        emb_dict['ubg'] = ubg_emb
        
        ## Behavior-speicfic graph aggregation ##
        for behavior_type in self.bsg_types:
            previous_emb = ubg_emb
            bsg_emb = self.propagate(self.dropout(previous_emb), edge_dict[behavior_type], behavior_type)
            emb_dict[behavior_type] = bsg_emb
        
        ## Target-intersected behavior graph aggregation ##
        for behavior_type in self.tib_types:
            if 'buy' in behavior_type:
                previous_behavior = behavior_type.split('_')[0] # view or cart or collect
                previous_emb = emb_dict[previous_behavior]
                tib_emb = self.propagate(self.dropout(previous_emb), edge_dict[behavior_type], behavior_type)
                emb_dict[behavior_type] = tib_emb

        for behavior_type in self.tcb_types:
            if 'buy' in behavior_type:
                previous_behavior = behavior_type.split('_')[0] # view or cart or collect
                previous_emb = emb_dict[previous_behavior]
                target_emb = emb_dict['buy']
                tcb_emb = self.propagate(self.dropout(previous_emb), edge_dict[behavior_type], behavior_type, target_emb)
                emb_dict[behavior_type] = tcb_emb
                
        final_emb = self.fusion(emb_dict)

        emb_dict['final'] = final_emb

        return emb_dict
    
    def hsic_graph(self, users_emb1, items_emb1, users_emb2, items_emb2):
        ### user part ###
        input_x = users_emb1
        input_y = users_emb2
        input_x = F.normalize(input_x, p=2, dim=1)
        input_y = F.normalize(input_y, p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        loss_user = hsic(Kx, Ky, users_emb1.shape[0])
        ### item part ###
        input_i = items_emb1
        input_j = items_emb2
        input_i = F.normalize(input_i, p=2, dim=1)
        input_j = F.normalize(input_j, p=2, dim=1)
        Ki = kernel_matrix(input_i, self.sigma)
        Kj = kernel_matrix(input_j, self.sigma)
        loss_item = hsic(Ki, Kj, users_emb1.shape[0])
        loss = loss_user + loss_item
        return loss
    

    def loss(self, users, pos_idx, neg_idx):
        emb_dict = self.forward()
        user_emb, item_emb = torch.split(emb_dict['final'], [self.n_users+1, self.n_items+1], dim=0)
        cl_loss = torch.tensor(0.0).cuda()
        hsic_loss = torch.tensor(0.0).cuda()
        gen_loss = torch.tensor(0.0).cuda()
        for behavior in self.total_behaviors+["init"]:
            user, item = torch.split(emb_dict[behavior], 
                                    [self.n_users+1, self.n_items+1], dim=0)

            emb_dict[behavior] = {'user': user, 'item': item}

        for behavior in self.bsg_types:
            hsic_loss += self.hsic_graph(emb_dict[behavior]["user"][users], emb_dict[behavior]["item"][pos_idx], emb_dict["ubg"]["user"][users], emb_dict["ubg"]["item"][pos_idx])
        
        for behavior in self.trbg_types:
            hsic_loss += self.hsic_graph(emb_dict[behavior]["user"][users], emb_dict[behavior]["item"][pos_idx], emb_dict[behavior.split('_')[0]]["user"][users], emb_dict[behavior.split('_')[0]]["item"][pos_idx])
            hsic_loss += self.hsic_graph(emb_dict[behavior]["user"][users], emb_dict[behavior]["item"][pos_idx], emb_dict["ubg"]["user"][users], emb_dict["ubg"]["item"][pos_idx])

        for behavior in self.bsg_types+["ubg"]+self.tib_types:
            cl_loss += self.cl_loss(user_emb[users], emb_dict[behavior]["user"][users])
            cl_loss += self.cl_loss(item_emb[pos_idx], emb_dict[behavior]["item"][pos_idx])
      
        for pre_beh, next_beh in zip(self.tcb_types[:-1], self.tcb_types[1:]):
            gen_loss += self.gen_loss(pre_beh, emb_dict[next_beh]["user"], emb_dict[next_beh]["item"])

        reg_loss = self.reg_loss(user_emb, item_emb)
        logits = torch.matmul(user_emb[users], item_emb.transpose(0, 1))
        ce_loss = self.ce_loss(logits, pos_idx)

        main_loss = ce_loss + 0.1*reg_loss
        aux_losses = {
            'cl': cl_loss,
            'dis': hsic_loss,
            'sharp': gen_loss
        }
        return main_loss, aux_losses
    
    def cl_loss(self, x1, x2):
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).mean()


    def predict(self, users):
        final_embeddings = self.forward()['final']
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.n_users + 1, self.n_items + 1])

        user_emb = final_user_emb[users.long()]
        scores = torch.matmul(user_emb, final_item_emb.transpose(0, 1))
        return scores