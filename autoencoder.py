"""
General AutoEncoder implementation.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib as mpl


class UnsupervisedDataSet(Dataset):
    """
    A class to deal right with our dataset.
    """
    def __init__(self, X_train, labels=None, transforms=None):
        self.X = X_train
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X.iloc[i, :]
        data = np.asarray(data)

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return data, self.y[i]
        else:
            return data


class AutoEncoder(nn.Module):
    """
    Our autoencoder implementation for several uses:
    1. Dimension Reduction- Reduce our data from 17 features to 2 features
    2. Join features- take a number of features and show them as a one vector representation.
    For every mission we have a little different structure.
    """
    def __init__(self, num_features, reduce_num_features):
        super(AutoEncoder, self).__init__()

        if num_features == 2:
            self.encoder = nn.Sequential(
                nn.Linear(2, 2),
                nn.Tanh(),
                nn.Linear(2, reduce_num_features),
            )
            self.decoder = nn.Sequential(
                nn.Linear(reduce_num_features, 2),
                nn.Tanh(),
                nn.Linear(2, 2)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(num_features, int(num_features/2)),
                nn.Tanh(),
                nn.Linear(int(num_features/2), reduce_num_features),
            )
            self.decoder = nn.Sequential(
                nn.Linear(reduce_num_features, int(num_features/2)),
                nn.Tanh(),
                nn.Linear(int(num_features/2), num_features)
            )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def data_preparation(data, features):
    """
    Function to prepare data frame that contains only the features we want to bring to a one
    vector representation.
    :param data: Our data
    :param features: The features we want to join together
    :return: A data frame with the specific features.
    """
    new_data_dict = {}
    for i in range(len(features)):
        x = data[features[i]]
        new_data_dict.update({features[i]: x.tolist()})
    new_data = pd.DataFrame(new_data_dict)
    return new_data


def train(model, train_loader, loss_func, optimizer):
    """
    Train function of the autoencoder.
    :param model: The autoencoder model
    :param train_loader: The train set
    :param loss_func: The loss function- MSE
    :param optimizer: The optimizer- Adam
    :return: The loss of the train process to every epoch
    """
    train_loss = 0
    for step, x in enumerate(train_loader):
        encoded, decoded = model(x.float())
        loss = loss_func(decoded, x.float())
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= int(len(train_loader.dataset))
    print('Train set: Average loss: ', train_loss)
    return train_loss


def validation(model, val_loader, loss_func):
    """
    Validation function of the autoencoder.
    :param model: The autoencoder model
    :param val_loader: The validation set
    :param loss_func: The loss function- MSE
    :return: The loss of the validation process to every epoch
    """
    val_loss = 0
    for x in val_loader:
        encoded, decoded = model(x.float())
        val_loss += loss_func(decoded, x.float()).item()
    val_loss /= int(len(val_loader.dataset))
    print('Validation set: Average loss: ', val_loss)
    return val_loss


def train_and_test(model, train_loader, val_loader, loss_func, optimizer, epochs):
    """
    A function to perform both train and test processes.
    :param model: The autoencoder model
    :param train_loader: The train set
    :param val_loader: The validation set
    :param loss_func: The loss function- MSE
    :param optimizer: The optimizer- Adam
    :param epochs: Number of epochs
    :return: Loss of train and validation processes
    """
    loss_train = []
    loss_val = []
    for epoch in range(epochs):
        loss = train(model, train_loader, loss_func, optimizer)
        loss_train.append(loss)
        loss = validation(model, val_loader, loss_func)
        loss_val.append(loss)
    return loss_train, loss_val


def final_encode(model, loader):
    """
    We built an autoencoder and trained and tested it, but in the end we want only the encoded data and
    not the decoded one. Therefore, this function will return us the encoded data, meaning the reduced
    data representation.
    :param model: The autoencoder model
    :param loader: The train or test set
    :return: The encoded data
    """
    for step, x in enumerate(loader):
        encoded, _ = model(x.float())
        if step > 0:
            final_encode = torch.cat((final_encode, encoded))
        else:
            final_encode = encoded
    return final_encode


def do_autoencoder(data, features, f1, f2, plot=False):
    """
    A final function to run all autoencoder model in order to get the final reduced data.
    :param data: Our data
    :param features: The features we want to reduce to a one or 2 vector representation
    :param f1: The dimension of the given data
    :param f2: The dimension of the encoded data
    :param plot: bool variable- if True, plot the loss per epoch plots for train and validation set
                and if False don't plot it.
    :return: The final encoded data
    """
    new_data = data_preparation(data, features)
    scaler = MinMaxScaler()
    new_data_scaled = scaler.fit_transform(new_data)
    new_data_scaled = pd.DataFrame(new_data_scaled)

    X_train, X_test = train_test_split(new_data_scaled, test_size=0.2, random_state=1)

    train_data = UnsupervisedDataSet(X_train)
    test_data = UnsupervisedDataSet(X_test)
    data = UnsupervisedDataSet(new_data_scaled)

    if f1 != 17:
        # Hyper Parameters
        epochs = 4
        batch_size = 128
        lr = 0.008
    else:
        # Hyper Parameters
        epochs = 7
        batch_size = 256
        lr = 0.008

    # data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # apply autoencoer and prepare the encoded data as a new data frame
    autoencoder = AutoEncoder(f1, f2)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    loss_train, loss_val = train_and_test(autoencoder, train_loader, test_loader, loss_func, optimizer, epochs)
    encode = final_encode(autoencoder, data_loader)
    encode = encode.tolist()
    encode = np.asarray(encode)
    encoded_data = pd.DataFrame(encode)
    names = []
    for i in range(f2):
        names.append('factor_{}.'.format(i))
    encoded_data.columns = names
    print(encoded_data.head())

    # if plot == True, plot graphs of loss per epoch for train and validation
    if plot is True:
        epochs = list(range(1, 4))
        plt.figure(1)
        mpl.rcParams['xtick.labelsize'] = 14
        mpl.rcParams['ytick.labelsize'] = 14
        mpl.rcParams['axes.titlesize'] = 20
        mpl.rcParams['axes.labelsize'] = 16
        plt.title('Loss Per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(epochs, loss_train, '-ok', color='red', label='train')
        plt.plot(epochs, loss_val, '-ok', color='blue', label='val')
        plt.legend()
        plt.show()

    return encoded_data
