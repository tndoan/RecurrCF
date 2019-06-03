import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np

class MovieLensDataset(data.Dataset):
    def __init__(self, train_data, num_items, num_neg_sampling):
        #self.rating_fname = rating_fname
        self.num_items = num_items

        movie_freq = [0] * num_items # max number of movie.
        #raw_data = dict()
        ## read file
        #f = open(rating_fname, 'r')
        #lines = f.readlines()
        #f.close()
        #for line in lines:
        #    comp = line.strip('\n').split('::')
        #    userId = int(comp[0]) - 1
        #    movieId = int(comp[1]) - 1
        #    rating = int(comp[2])
        #    timestamp = int(comp[3])

        #    uData = raw_data.get(userId)
        #    if uData == None:
        #        uData = list()
        #        raw_data[userId] = uData
        #    uData.append((movieId, rating, timestamp))
        #    movie_freq[movieId] += 1
        num_datapoint = 0.0
        for uId in train_data.keys():
            dataOfEachUser = train_data[uId]
            for iId, rating, timestamp in dataOfEachUser:
                movie_freq[iId] += 1
                num_datapoint += 1.0
        
        movie_freq = [x / num_datapoint for x in movie_freq] # normalize
        data = dict()
        #for (uId, uData) in raw_data.items():
        for (uId, uData) in train_data.items():
            uData.sort(key=lambda x: x[2])
            clean_uData = [x[0] for x in uData]
            sub_movie_freq = movie_freq.copy()
            for i in clean_uData:
                sub_movie_freq[i] = 0.0
            total_sub = sum(sub_movie_freq)
            sub_movie_freq = [i / total_sub for i in sub_movie_freq]
            # negative sampling
            neg_items = np.random.choice(len(movie_freq), (len(clean_uData), num_neg_sampling), p=sub_movie_freq)
            data[uId] = (clean_uData, neg_items)

        self.data = data

    def __getitem__(self, index):
        pos_data, neg_data = self.data[index]
        pos_data = torch.transpose(torch.LongTensor([pos_data]), 0, 1) 
        neg_data = torch.LongTensor(neg_data)
        return torch.ones((len(pos_data), 1), dtype=torch.long) * index, pos_data, neg_data

    def __len__(self):
        return len(self.data.keys())

    def numUsers(self):
        return len(self.data.keys())

    def numItems(self):
        return self.num_items

def collate_fn(data):
    uId, pos_data, neg_data = zip(*data)
    uId = torch.cat(uId)

    pos_data = torch.cat(pos_data)
    neg_data = torch.cat(neg_data)
    return uId, pos_data, neg_data

def get_loader(train_data, num_items, num_neg, batch_size, num_workers):
    dataset = MovieLensDataset(train_data, num_items, num_neg)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader, dataset.numUsers(), dataset.numItems()
