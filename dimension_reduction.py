"""
Code for applying dimensionality reduction techniques- PCA, Kernel PCA, t-SNE and two kinds of AutoEncoders. Dimension
Reduction can be applied into 2 or 3 dimensions (in the main function choose dim=2 or dim=3). We also visualize these
five dimension reduction by plotting them next to each other.
"""

from models import *
from plotting_utils import *
from sklearn.preprocessing import MinMaxScaler
from autoencoder import do_autoencoder


def read_data():
    """
    Read the data.
    :return: data after cleaning
    """
    data = pd.read_csv("CC GENERAL.csv")
    data.loc[(data['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].median()
    data.loc[(data['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].median()
    data = data.drop(['CUST_ID'], 1)
    return data


def normalize(data):
    """
    Data normalization by min-max scaler.
    :param data: Our data set
    :return: The normalized data values, the normalized data and the names of the columns.
    """
    scaler = MinMaxScaler()
    names = data.columns
    my_data_scaled = scaler.fit_transform(data)
    my_data_scaled = pd.DataFrame(my_data_scaled, columns=names)
    X = data.values
    X_scaled = scaler.fit_transform(X)
    return X_scaled, my_data_scaled, names


def dimension_reduction(X_scaled, names, data, data_scaled, dim):
    """
    Function to apply all dimensionality reduction methods that were used: PCA, kernel PCA, t-SNE
    and two types of auroencoders.
    :param X_scaled: The normalized data values
    :param names: The columns of the original data (without categorical feature)
    :param data: Our data set
    :param data_scaled: The normalized data set
    :param dim: The dimension of the reduced data
    :return: Data after dimension reduction from all types and plots for visualization
    """
    # pca
    print("PCA:")
    points_pca = perform_pca(X_scaled, dim)

    # kernel pca
    print("Kernel PCA")
    nystroem_kwargs = {'kernel': 'cosine', 'n_components': 17}
    pca_kwargs = {'n_components': dim}
    points_nystroem_pca = nystroem_pca(X_scaled, nystroem_kwargs, pca_kwargs)

    if dim == 2:
        # tsne
        points_tsne = perform_tsne(X_scaled, 2)

    # autoencoder from 17 -> dim
    encoded_data = do_autoencoder(data_scaled, names, len(names), dim)
    scaler = MinMaxScaler()
    encoded_data_scaled = scaler.fit_transform(encoded_data)
    encoded_data_scaled = pd.DataFrame(encoded_data_scaled)
    points_encoded = encoded_data_scaled.values

    # autoencoder 14 -> dim
    features = ['BALANCE_FREQUENCY', 'PRC_FULL_PAYMENT', 'TENURE', 'BALANCE']
    encoded_data = do_autoencoder(data_scaled, features, len(features), 1)
    data2 = data.copy()
    data2 = data2.drop(features, 1)
    for i in range(len(encoded_data.columns)):
        data2['factor_{}.'.format(i)] = encoded_data['factor_{}.'.format(i)]
    names2 = data2.columns
    data_scaled = scaler.fit_transform(data2)
    data_scaled = pd.DataFrame(data_scaled, columns=names2)
    points_2_dim = do_autoencoder(data_scaled, names2, len(names2), dim)
    points_2_dim = points_2_dim.values

    return points_pca, points_nystroem_pca, points_tsne, points_encoded, points_2_dim


def main(dim):
    """
    Main function to calculate and visualize reduced data
    :param dim: The dimension of the reduced data
    :return: data after dimension reduction and visualization
    """
    # read and normalize the dataa
    data = read_data()
    X_scaled, data_scaled, names = normalize(data)
    # apply dimension reduction methods
    points_pca, points_nystroem_pca, points_tsne, points_encoded, points_2_dim = \
        dimension_reduction(X_scaled, names, data, data_scaled, dim)
    # plot for visualization
    if dim == 2:
        fig, axes = plot_2d(points=(points_pca, points_nystroem_pca, points_tsne, points_encoded, points_2_dim))
        fig.tight_layout()
        plt.show()
    else:
        fig, axes = plot_3d(points=(points_pca, points_nystroem_pca, points_encoded, points_2_dim))
        plt.show()
    return points_pca, points_tsne, points_2_dim, X_scaled


