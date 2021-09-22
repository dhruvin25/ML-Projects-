import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == "__main__":
    # read the data
    df = pd.read_csv('/home/dhruvin/python/code/Corona Symptom/data.csv')
    
    
    # train test split 
    train, test = data_split(df, 0.2)
    
    x_train = train[['fever','Bodypain','age','Runnynose','diffBreath']].to_numpy()
    x_test = test[['fever','Bodypain','age','Runnynose','diffBreath']].to_numpy()

    y_train = train[['infectionProb']].to_numpy().reshape(560,)
    y_test = test[['infectionProb']].to_numpy().reshape(140,)

    # model implementation
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # open a file, np.where you want to store the data
    file = open('model.pkl','wb')

    # dump information to that file
    pickle.dump(model, file)
    file.close()
