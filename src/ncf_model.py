import torch
import torch.nn as nn

class NCF_Model(nn.Module):
    def __init__(self, config):
        super(NCF_Model, self).__init__()

        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        self.emb_item_mlp = nn.Embedding(self.num_items, self.latent_dim_mlp)
        self.emb_item_mf  = nn.Embedding(self.num_items, self.latent_dim_mf)
        self.emb_user_mlp = nn.Embedding(self.num_users, self.latent_dim_mlp)
        self.emb_user_mf  = nn.Embedding(self.num_users, self.latent_dim_mf)

        layers = config['layers']
        self.fc_layers = nn.ModuleList()
        # first layers
        self.fc_layers.append(nn.Linear(2 * self.latent_dim_mlp, layers[0])) 
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.affine_output = nn.Linear(layers[-1] + self.latent_dim_mf, 1)
        self.logistic = nn.Sigmoid()

    def forward(self, uIds, iIds):
        u_mlp = self.emb_user_mlp(uIds)
        u_mf  = self.emb_user_mf(uIds)
        i_mlp = self.emb_item_mlp(iIds)
        i_mf  = self.emb_item_mf(iIds)

        gmf = torch.mul(u_mf, i_mf)
        
        mlp_input = torch.cat((u_mlp, i_mlp), 1)
        for i in range(len(layers) + 1):
            mlp_input = self.fc_layers[i](mlp_input)
            mlp_input = nn.ReLU()(mlp_input)

        neumf = torch.cat((gmf, mlp_input), 1)
        return self.logistic(self.affine_output(neumf))
