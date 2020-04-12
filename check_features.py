"""
Code to check which features have the most influence on clustering performance and which features don't, so
we can merge them into a one feature representation.
"""

from models import *
from plotting_utils import *
from sklearn.preprocessing import MinMaxScaler


def main():
    # read the data
    data = pd.read_csv("CC GENERAL.csv")
    data.loc[(data['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].median()
    data.loc[(data['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].median()
    data = data.drop(['CUST_ID'], 1)

    names = data.columns.tolist()

    for i in range(len(names)):
        # the full data set
        data = pd.read_csv("CC GENERAL.csv")
        data.loc[(data['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].median()
        data.loc[(data['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].median()
        data = data.drop(['CUST_ID'], 1)

        # every iteration remove different feature
        data = data.drop([names[i]], 1)

        # check correlation
        plot_correlation_matrix(data)

        # normalization
        X = data.values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # perform 2D PCA
        points_pca_2 = perform_pca(X_scaled, 2)
        plot_2d_pca_tsne(points_pca_2, 'PCA')

        # elbow method to check number of clusters
        perform_elbow_method(X_scaled)

        # silhouette scores to check number of clusters
        calculate_and_plot_silhouette_scores(X_scaled, points_pca_2, "K-means")

        # perform clustering and plot it. For now it is K-means, one can change it to 'gmm' and 'hc'
        kmeans_predict = perform_clustering_methods('kmeans', points_pca_2, 3)
        plot_clustering(points_pca_2, kmeans_predict, 'PCA', 'gmm', 2)
