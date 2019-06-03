import pickle

def makeGenPredData(rating_file, percentage):
    """
    percentage must be between 0 and 1
    first, we sort all data by created time. percentage is the ratio of data
    points is used for training. (1 - percentage) is portion of data for testing
    10% of training is used for validation set.
    """
    raw_data = readRatingFile(rating_file)
    raw_data.sort(key=lambda tup: tup[2]) # sort by created time

    num_test = int(len(raw_data) * (1 - percentage))
    num_validation = int(0.1 * (len(raw_data) - num_test))
    num_train = len(raw_data) - num_test - num_validation

    raw_train = list(); raw_test = list(); raw_validation = list()
    for i in range(len(raw_data)):
        if i < num_train:
            raw_train.append(raw_data[i])
        elif num_train <= i < num_train + num_validation:
            raw_validation.append(raw_data[i])
        else:
            raw_test.append(raw_data[i])

    count_iIds = 0; count_uIds = 0; iMap = dict(); uMap = dict()
    for raw_uId, raw_mId, rating, timestamp in raw_train:
        c_uId = uMap.get(raw_uId)
        if c_uId == None:
            c_uId = count_uIds
            uMap[raw_uId] = c_uId
            count_uIds += 1
        c_mId = iMap.get(raw_mId)
        if c_mId == None:
            c_mId = count_iIds
            iMap[raw_mId] = c_mId
            count_iIds += 1

    train = dict(); test = dict(); validation = dict()
    for raw_uId, raw_mId, rating, timestamp in raw_train:
        c_uId = uMap.get(raw_uId)
        c_mId = iMap.get(raw_mId)
        uData = train.get(c_uId)
        if uData == None:
            uData = list()
            train[c_uId] = uData
        uData.append((c_mId, rating, timestamp))
    data = dict(); data['num_users'] = count_uIds; data['num_items'] = count_iIds
    
    count_invalid_test = 0
    for raw_uId, raw_mId, rating, timestamp in raw_test:
        c_uId = uMap.get(raw_uId)
        c_mId = iMap.get(raw_mId)
        if c_uId == None or c_mId == None:
            count_invalid_test += 1
            continue
        uData = test.get(c_uId) 
        if uData == None:
            uData = list()
            test[c_uId] = uData
        uData.append((c_mId, rating, timestamp))

    count_invalid_validation = 0
    for raw_uId, raw_mId, rating, timestamp in raw_validation:
        c_uId = uMap.get(raw_uId)
        c_mId = iMap.get(raw_mId)
        if c_uId == None or c_mId == None:
            count_invalid_validation += 1
            continue
        uData = validation.get(c_uId) 
        if uData == None:
            uData = list()
            validation[c_uId] = uData
        uData.append((c_mId, rating, timestamp))

    print('# invalid test:', count_invalid_test)
    print('# invalid validation:', count_invalid_validation)
    
    data['train'] = train; data['validation'] = validation; data['test'] = test
    data['raw_uId_to_cont'] = uMap
    data['raw_mId_to_cont'] = iMap
    return data

def makeLeaveOneOutData(rating_file):
    raw_data = readRatingFile(rating_file)
    uMap = dict() # key is user id and value is list of movie he watches
    for (uId, mId, rating, timestamp) in raw_data:
        uData = uMap.get(uId)
        if uData == None:
            uData = list()
            uMap[uId] = uData
        uData.append((mId, rating, timestamp))

    # map real movie id to continuous id
    cont_iIds = 0; iMap = dict(); cont_uIds = 0
    raw_train = dict(); raw_test = dict(); raw_validation = dict()
    for uId in uMap.keys():
        uData = uMap.get(uId)
        uData.sort(key=lambda tup: tup[2]) # sort by created time
        t_mId, t_rating, t_timestamp = uData.pop() # get the last movie for testing
        v_mId, v_rating, v_timestamp = uData.pop() # get the 2nd last movie for validation
        raw_train[cont_uIds] = uData
        raw_test[cont_uIds] = (t_mId, t_rating, t_timestamp)
        raw_validation[cont_uIds] = (v_mId, v_rating, v_timestamp)

        for mId, rating, timestamp in uData:
            c_mId = iMap.get(mId)
            if c_mId == None:
                c_mId = cont_iIds
                iMap[mId] = c_mId
                cont_iIds += 1
        cont_uIds += 1
   
    countInvalidTest = 0; countInvalidValidation = 0
    train = dict(); test = dict(); validation = dict()
    for uId in range(cont_uIds):
        uData = raw_train[uId]
        
        # data for training
        trainDataForEachUser = list()
        for mId, rating, timestamp in uData:
            c_mId = iMap.get(mId)
            trainDataForEachUser.append((c_mId, rating, timestamp))
        train[uId] = trainDataForEachUser

        # check for validation
        v_data = raw_validation[uId]
        c_mId = iMap.get(v_data[0])
        if c_mId == None:
            countInvalidValidation += 1
        else:
            validation[uId] = (c_mId, v_data[1], v_data[2])

        # check for testing
        t_data = raw_test[uId]
        c_mId = iMap.get(t_data[0])
        if c_mId == None:
            countInvalidTest += 1
        else:
            test[uId] = (c_mId, t_data[1], t_data[2])

    print("# invalid test:", countInvalidTest)
    print("# invalid validation:", countInvalidValidation)
    
    data = dict()
    data['train'] = train; data['validation'] = validation; data['test'] = test
    data['num_users'] = cont_uIds; data['num_items'] = cont_iIds
    data['raw_uId_to_cont'] = iMap
    return data

def readRatingFile(rating_file):
    data = list()
    with open(rating_file) as f:
        lines = f.readlines()

    for line in lines:
        comp = line.strip('\n').split('::')
        userId = int(comp[0]) - 1
        movieId = int(comp[1]) - 1
        rating = int(comp[2])
        timestamp = int(comp[3])
        data.append((userId, movieId, rating, timestamp))
    return data

if __name__ == '__main__':
    data = readRatingFile('../movieLens_data/ratings.dat')
    leaveOneOutData = makeLeaveOneOutData('../movieLens_data/ratings.dat')
    print('--')
    genPredData = makeGenPredData('../movieLens_data/ratings.dat', 0.8)
    pickle.dump(leaveOneOutData, open( "../movieLens_data/leaveOneOut.pkl", "wb"))
    pickle.dump(genPredData, open( "../movieLens_data/genPredData.pkl", "wb"))
