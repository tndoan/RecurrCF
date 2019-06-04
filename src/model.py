import torch
import torch.nn as nn
import torch.nn.functional as F

class RCF(nn.Module):
    def __init__(self, config):
        super(RCF, self).__init__()

        self.device = config['device']

        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.emb_size = config['emb_size']

        self.hidden_size = config['emb_size'] # hidden size must be equal to embedding size

        self.u_emb = nn.Embedding(self.num_users, self.emb_size)
        self.i_emb = nn.Embedding(self.num_items, self.emb_size)
        self.u_bias = nn.Parameter(torch.randn(self.num_users))
        self.i_bias = nn.Parameter(torch.randn(self.num_items))

        self.gru = nn.GRU(self.emb_size, self.hidden_size)
        #self.linear = nn.Linear(self.emb_size, 6) # use 6 since the rating is 0 - 5

    def forward(self, u_idx, i_idx, n_i_idx, rating):
        # u_idx  : id of user
        # i_idx  : vector of item id. suppose we have N items. 0-th item is the 1st purchase item of user, etc
        # n_i_idx: matrix N x M. negative sampling. M negative items for each item
        # rating : rating of user to 
        
        epsilon = 1e-5
        m = n_i_idx.size()[1] # number of negative sampling
        #print('# of negative:', m)

        # matrix factorization
        u_e = self.u_emb(u_idx)
        i_e = self.i_emb(i_idx)
        u_b = self.u_bias[u_idx]
        i_b = self.i_bias[i_idx]
        
        n_i_e = self.i_emb(n_i_idx)
        n_i_b = self.i_bias[n_i_idx]

        neg_pred = torch.matmul(n_i_e, u_e.transpose(1, 2)) 
        neg_pred = neg_pred.squeeze() + u_b + n_i_b
        #print('neg_pred:', neg_pred)
        neg_rating_loss = torch.log(epsilon + 1.0 - torch.sigmoid(neg_pred)).sum() / m
        #print('abc:', torch.sigmoid(neg_pred))
        #print('xyz:', torch.log(epsilon + 1.0 - torch.sigmoid(neg_pred)))

        rating_pred = torch.matmul(i_e, u_e.squeeze().t()) + u_b + i_b
        pos_rating_loss = torch.log(epsilon + torch.sigmoid(rating_pred)).sum()

        #print('neg_rating_loss:', neg_rating_loss.item())
        #print('pos_rating_loss:', pos_rating_loss.item())
        #exit()

        rating_loss = neg_rating_loss + pos_rating_loss

        # sequential data
        i_e_s = i_e.squeeze()
        uu = u_idx.squeeze().cpu().numpy()
        start = uu[0]
        indexes = []
        for i in range(1, len(uu)):
            if uu[i] != start:
                indexes.append(i)
                start = uu[i]
        indexes.append(len(uu))
        #print(indexes)
        pos_seq = 0; neg_seq = 0
        for i in range(len(indexes)):
            start = 0
            if i != 0:
                start = i - 1
            sub_i_e = i_e_s[start:indexes[i]]
            hiddens, _ = self.gru(sub_i_e.unsqueeze(1), None)
            hiddens = hiddens + u_e[start:indexes[i]]
            hiddens = hiddens.squeeze()
            #print('hiddens:', hiddens)

            pos_seq += torch.log(epsilon + torch.sigmoid(torch.diag(torch.matmul(hiddens, sub_i_e.t())))).sum()
            #print('n_i_e:', n_i_e)
            
            sub_n_i_e = n_i_e[start:indexes[i]]
            n_matmul = torch.matmul(sub_n_i_e, hiddens.t())
            num_state = n_i_idx[start:indexes[i]].size()[0]
            n_matmul = -n_matmul[range(num_state), :, range(num_state)]
            neg_seq += torch.log(epsilon + torch.sigmoid(n_matmul)).sum() / m

        seq_loss = pos_seq + neg_seq

        #return -rating_loss - seq_loss
        return rating_loss, seq_loss
