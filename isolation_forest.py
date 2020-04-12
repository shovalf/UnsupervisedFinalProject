"""
Isolation Forest Algorithm implementation on our data in order to identify outliers. We also apply 2D and 3D PCA in
order to visualize them.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl

# for plotting
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 14


def read_data():
    """
    Read the data and clean it
    """
    data = pd.read_csv("CC GENERAL.csv")
    data.loc[(data['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].median()
    data.loc[(data['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].median()
    data = data.drop(['CUST_ID'], 1)
    return data


def normalization(data):
    """
    Normalize the data by min-max normalization
    """
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)
    return data


def isolation_forest(data):
    """
    Apply isolation forest algorithm
    """
    clf = IsolationForest(n_estimators=100, max_samples=3, contamination=float(.02),
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
    clf.fit(data)
    pred = clf.predict(data)
    return pred


def outliers(data, pred):
    """
    Function to know which specific samples are outliers
    """
    data2 = data
    data2['anomaly'] = pred
    outliers = data.loc[data2['anomaly'] == -1]
    outlier_index = list(outliers.index)
    return data2, outlier_index


def perform_3d_pca(data, outlier_index):
    """
    Apply 3D PCA and visualize to see inliers and outliers
    """
    pca = PCA(n_components=3)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data)
    X_reduce = pca.fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlabel("x_composite_3")
    # Plot the compressed data points
    ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers",c="lightblue")
    # Plot x's for the ground truth outliers
    ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1], X_reduce[outlier_index,2],
               lw=2, s=4, marker="x", c="turquoise", label="outliers")
    ax.set_title("Data Visualization after 3D PCA, Outliers in Red")
    ax.set_xlabel("PC0")
    ax.set_ylabel("PC1")
    ax.set_zlabel("PC2")
    ax.legend()
    plt.show()


def perform_2d_pca(data, outlier_index):
    """
    Apply 2D PCA and visualize to see inliers and outliers
    """
    pca = PCA(2)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data)
    X_reduce = pca.fit_transform(X)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    # Plot the compressed data points
    ax.scatter(X_reduce[:, 0], X_reduce[:, 1], s=4, lw=1, label="inliers",c="turquoise")
    # Plot x's for the ground truth outliers
    ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1],
               lw=2, s=4, marker="x", c="red", label="outliers")
    ax.set_title("Data Visualization after 2D PCA, Outliers in Red")
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    plt.show()


def main():
    data = read_data()
    data = normalization(data)
    predictions = isolation_forest(data)
    data2, outlier_index = outliers(data, predictions)
    # perform_3d_pca(data2, outlier_index)
    perform_2d_pca(data2, outlier_index)


# main()
