import torch, pickle
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from model import RCF
from data_loader import get_loader

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #data = pickle.load(open('../movieLens_data/leaveOneOut.pkl', "rb"))
    data = pickle.load(open('../movieLens_data/genPredData.pkl', "rb"))

    num_items = data['num_items']
    train_data = data['train']

    # data loader
    data_loader, numUsers, numItems = get_loader(train_data, num_items, 2, 2, 3)
    print(numUsers)
    print(numItems)
    
    # Build model
    config = {'device': device,
            'num_users': numUsers,
            'num_items': numItems,
            'emb_size': 50
            }
    m = RCF(config).to(device)
    optimizer = optim.Adam(m.parameters(), lr=1e-3, weight_decay=0.01)

    for i in range(10): # number of epochs
        for (uId, pos_data, neg_data) in data_loader:
            #print('uId:', uId)
            #print('pos_data:', pos_data)
            #print('neg_data:', neg_data)
            uId = uId.to(device)
            pos = pos_data.to(device)
            neg = neg_data.to(device)

            m.zero_grad()
            #loss = m.forward(uId, pos, neg, None)
            r_loss, s_loss = m.forward(uId, pos, neg, None)
            #print('rating loss:', r_loss.item())
            #print('seq loss:', s_loss.item())
            loss = - r_loss - s_loss
            print('loss:', loss.item())
            #print('loss:', loss)
            loss.backward()
            optimizer.step()
        print('finish epoch:', i)

#def test_code():
#    device = 'cpu'
#    config = {'device': device,
#            'num_users': 3,
#            'num_items': 5,
#            'emb_size': 10
#            }
#    m = RCF(config).to(device)
#    optimizer = optim.Adam(m.parameters(), lr=1e-3)
#    uId = torch.tensor([[0], [0], [1], [1], [1]], dtype=torch.long)
#    pos = torch.tensor([[1], [2], [0], [1], [2]], dtype=torch.long)
#    neg = torch.tensor([[0, 4], [0, 3], [3, 4], [3, 4], [3, 4]], dtype=torch.long)
#    m.forward(uId, pos, neg, None)

if __name__ == '__main__':
    train()
    #test_code()
