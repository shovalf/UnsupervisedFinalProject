"""
Code to apply all different clustering methods (k-means, GMM and Hierarchical Clustering, with all three different
dimensionality reduction (PCA, autoencoder and t-SNE). Moreover, check clustering performance with average silhouette
score, davies bouldin score and calinski harabasz score.
"""

from models import *
from plotting_utils import *
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score


def main(points_pca, points_tsne, points_2_dim, X_scaled):
    # read the data
    data = pd.read_csv("CC GENERAL.csv")
    data.loc[(data['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].median()
    data.loc[(data['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].median()
    data = data.drop(['CUST_ID'], 1)

    # for plotting
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 14

    # elbow method
    perform_elbow_method(X_scaled)

    # our data after dimensionality reduction with autoencoder, PCA and t-SNE, respectively
    points = [points_2_dim, points_pca, points_tsne]
    # three different clustering methods
    clustering = ["K-means", "GMM", "Hierarchical Clustering"]
    methods = ['AutoEncoder', 'PCA', 'T-SNE']

    for i in range(len(points)):
        # silhouette scores for K-means, GMM and Hierarchical Clustering
        print("\n")
        print(methods[i])
        print("Average silhouette scores for K-means:")
        calculate_and_plot_silhouette_scores(X_scaled, points[i], clustering[0])
        print("\n")
        print("Average silhouette scores for GMM:")
        calculate_and_plot_silhouette_scores(X_scaled, points[i], clustering[1])
        print("\n")
        print("Average silhouette scores for Hierarchical Clustering:")
        calculate_and_plot_silhouette_scores(X_scaled, points[i], clustering[2])

        # apply Kmeans, GMM and Hierarchical Clustering
        kmeans_predict = perform_clustering_methods('kmeans', points[i], 3)
        gmm_predict = perform_clustering_methods('gmm', points[i], 3)
        hc_predict = perform_clustering_methods('hc', points[i], 3)

        predictions = [kmeans_predict, gmm_predict, hc_predict]

        # visualize the clustering for every clustering method
        fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 4), constrained_layout=True)
        for j in range(3):
            axes[j].scatter(points[i][:, 0], points[i][:, 1], c=predictions[j], cmap='tab10', alpha=0.8, s=8)
            axes[j].set_title("{} on {} reduced data".format(clustering[j], methods[i]))
            if methods[i] == 'PCA':
                axes[j].set_xlabel("First Principal Component")
                axes[0].set_ylabel("Second Principal Component")

        fig.suptitle('Clustering Methods on PCA Reduced Data', fontsize=20)
        plt.show()

        # calculate davies bouldin score and calinski harabasz score
        for k in range(len(predictions)):
            a = davies_bouldin_score(points[i], predictions[k])
            b = calinski_harabasz_score(points[i], predictions[k])
            print("\n")
            print("For {} Algorithm, Davies Douldin Score is: {}".format(clustering[k], a))
            print("For {} Algorithm, Calinski Harabasz Score is: {}".format(clustering[k], b))
