import torch, pickle
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from model import RCF
from data_loader import get_loader
import gc

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
        count = 0.0
        total_loss = 0.0
        for (uId, pos_data, neg_data) in data_loader:
            count += 1.0
            #print('count:', count)
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
            if count % 100 == 0:
                print('loss:', loss.item())
            #print('loss:', loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / 20.0
            gc.collect()
        print('finish epoch:', i)
        print("Loss:", total_loss / count)

def test_code():
    device = 'cpu'
    config = {'device': device,
            'num_users': 3,
            'num_items': 5,
            'emb_size': 10
            }
    m = RCF(config).to(device)
    optimizer = optim.Adam(m.parameters(), lr=1e-3)
    #uId = torch.tensor([[0], [0], [1], [1], [1]], dtype=torch.long)
    uId = torch.tensor([[0], [1], [1]], dtype=torch.long)
    pos = torch.tensor([[1], [1], [2]], dtype=torch.long)
    neg = torch.tensor([[0, 4], [2, 4], [3, 1]], dtype=torch.long)
    r_loss, s_loss = m.forward(uId, pos, neg, None)
    loss = - r_loss - s_loss
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    train()
    #test_code()
