    

import pandas as pd
from pyparsing import Path
import numpy as np

def read_synthetic_data(N,noise,deg):

    path_dir = Path(__file__).parent / "SyntheticPortfolioData"
    Train_dfx= pd.read_csv(
        path_dir / 
        "TraindataX_N_{}_noise_{}_deg_{}.csv"
        .format(N,noise,deg),header=None
    )
    Train_dfy= pd.read_csv(
        path_dir /
        "Traindatay_N_{}_noise_{}_deg_{}.csv"
        .format(N,noise,deg),header=None
    )
    x_train =  Train_dfx.T.values.astype(np.float32)
    y_train = Train_dfy.T.values.astype(np.float32)*-1

    Validation_dfx= pd.read_csv(
        path_dir / 
        "ValidationdataX_N_{}_noise_{}_deg_{}.csv"
        .format(N,noise,deg),header=None
    )
    Validation_dfy= pd.read_csv(
        path_dir /
        "Validationdatay_N_{}_noise_{}_deg_{}.csv"
        .format(N,noise,deg),header=None
    )
    x_valid =  Validation_dfx.T.values.astype(np.float32)
    y_valid = Validation_dfy.T.values.astype(np.float32)*-1

    Test_dfx= pd.read_csv(
        path_dir /
        "TestdataX_N_{}_noise_{}_deg_{}.csv"
        .format(N,noise,deg),header=None
    )
    Test_dfy= pd.read_csv(
        path_dir /
        "Testdatay_N_{}_noise_{}_deg_{}.csv"
        .format(N,noise,deg),header=None
    )
    x_test =  Test_dfx.T.values.astype(np.float32)
    y_test = Test_dfy.T.values.astype(np.float32)*-1
    data =  np.load(
        path_dir /
        "GammaSigma_N_{}_noise_{}_deg_{}.npz".format(N,noise,deg)
    )
    cov = data['sigma']
    gamma = data['gamma']
    return x_train, y_train, x_valid, y_valid, x_test, y_test, cov, gamma