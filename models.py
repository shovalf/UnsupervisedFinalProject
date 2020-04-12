"""
All models that we used in this project. For example, kernel PCA, applying clustering methods and more.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer
from sklearn.kernel_approximation import Nystroem
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib as mpl


def perform_pca(x_scaled, dim):
    """
    Perform PCA on our data in order to reduce dimensions and return the reduced data
    :param x_scaled: Scaled values of our data
    :param dim: Number of dimensions of the reduced dimensions data. if dim==2, we perform 2d pca
            if dim==3, we perform 3d pca.
    :return:
    """
    pca = PCA(n_components=dim)
    pca.fit(x_scaled)
    # Print relevant information
    x_reduced = pca.transform(x_scaled)
    print("Dimensionality reduction: {} -> {}".format(x_scaled.shape[1], x_reduced.shape[1]))
    print("Variance explained by each component:", (pca.explained_variance_ratio_ * 1000).astype(int) / 1000)
    print("Total variance explained by those {} components:".format(x_reduced.shape[1]),
          format(pca.explained_variance_ratio_.sum(), ".4f"))
    return x_reduced


def nystroem_pca(X, nystroem_kwargs, pca_kwargs, verbose=1):
    """
    Perform Nystroem kernel approximation and then PCA decomposition. The Nystroem method constructs an
    approximate feature map, for an arbitrary kernel using a subset of the data as basis. We then
    apply this feature map to X, and perform PCA decomposition in the new feature space.
    :param X: Values of our data
    :param nystroem_kwargs: (dict) Keyword arguments that are passed to the constructor of an implementation
            of the Nystroem method - `sklearn.kernel_approximation.Nystroem`.
    :param pca_kwargs: dict) Keyword arguments that are passed to `sklearn.decomposition.PCA`.
    :param verbose: (int) If verbose=1, print information relevant to the PCA decomposition. Set verbose=0
            for no output.
    :return: `numpy.ndarray`:  The PCA decomposition of our data set in the new feature space, that was
            approximated via the nystroem method.
    """

    # Get the nystroem approximation
    nystroem = Nystroem(**nystroem_kwargs)
    nystroem.fit(X)
    X_nystroem = nystroem.transform(X)

    # Get the PCA decomposition
    pca_nystroem = PCA(**pca_kwargs)
    pca_nystroem.fit(X_nystroem)
    X_nystroem_pca = pca_nystroem.transform(X_nystroem)

    # Print Relevant PCA information
    if verbose:
        print("Dimensionality reduction: {} -> {}"
              .format(X_nystroem.shape[1], X_nystroem_pca.shape[1]))
        print("Variance explained by each component:",
              (pca_nystroem.explained_variance_ratio_ * 1000).astype(int) / 1000)
        print("Total variance explained by those {} components:".format(X_nystroem_pca.shape[1]),
              format(pca_nystroem.explained_variance_ratio_.sum(), ".4f"))

    return X_nystroem_pca


def perform_tsne(x_scaled, dim):
    """
    Perform t-SNE dimension reduction technique
    :param x_scaled: The values of the data after normalization
    :param dim: The dimension of the reduced data
    :return: data after dimensionality reduction
    """
    x_embedded = TSNE(n_components=dim, perplexity=50).fit_transform(x_scaled)
    return x_embedded


def perform_elbow_method(x_scaled):
    """
    Perform the elbow method to help us decide the number of clusters
    :param x_scaled: Values of our data after normalization
    :return: A plot of the elbow method
    """
    # Instantiate the clustering model and visualizer
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 18
    mpl.rcParams['axes.labelsize'] = 14
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1, 12))
    # Fit the data to the visualizer
    visualizer.fit(x_scaled)
    visualizer.set_title("The Elbow Method")
    visualizer.show()


def calculate_and_plot_silhouette_scores(X, points, method):
    """
    Calculate and plot silhouette scores for a given clustering method method and range of number of clusters.
    :param X: The data values
    :param points: The data values after dimensionality reduction
    :param method: Clustering method- K-means / GMM / Hierarchical Clustering
    :return: Silhouette score and plots
    """
    range_n_clusters = [2, 3, 4, 5, 6]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot. The silhouette coefficient can range from -1, 1
        ax1.set_xlim([-0.1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        if method == 'K-means':
            # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(points)
        elif method == "GMM":
            # gmm clustering
            clusterer = GaussianMixture(n_components=n_clusters, n_init=3, covariance_type='diag', tol=1e-3, verbose=0, random_state=1)
            clusterer.fit(points)
            cluster_labels = clusterer.predict(points)
        else:
            # Hierarchical Clustering
            linkages = ['ward', 'average', 'complete', 'single']
            clusterer = AgglomerativeClustering(linkage=linkages[2], n_clusters=n_clusters, affinity='euclidean')
            cluster_labels = clusterer.fit_predict(points)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(points, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(points, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for {} clusters".format(n_clusters))
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(points[:, 0], points[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        if method == 'K-means':
            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("First Principal Component")
        ax2.set_ylabel("Second Principal Component")

        plt.suptitle(("Silhouette analysis for {} clustering on sample data "
                      "with n_clusters = {}".format(method, n_clusters)),
                     fontsize=14, fontweight='bold')

    plt.show()


def perform_clustering_methods(method, points, n):
    """
    Perform 3 clustering methods for unsupervised learning- Kmeans, GMM, DBSCAN and hierarchical clustering
    :param method: The clustering method
    :param points: The reduced data after performing 2d/3d pca or tsne.
    :param n: Number of clusters
    :return: The predicted label
    """
    if method == 'kmeans':
        # k-means clustering method
        predictions = KMeans(n_clusters=n, init='k-means++', random_state=101).fit_predict(points)
    elif method == 'gmm':
        # GMM clustering method, means initialized with K-means centers
        kmeans = KMeans(n_clusters=n).fit(points)
        centers = kmeans.cluster_centers_
        # means_init = np.array([(1, -0.22, 0.9),
        #                        (0.1561, 0.095, -0.565),
        #                        (-0.98, 0.12, 0.123),
        #                        (0.453, -.874, -0.298)])
        gmm = GaussianMixture(n_components=n, n_init=3, means_init=centers,
                              covariance_type='diag', tol=1e-3, verbose=0, random_state=1)
        gmm.fit(points)
        predictions = gmm.predict(points)
    elif method == 'dbscan':
        # DBSCAN clustering method
        dbscan = DBSCAN(eps=0.0001, min_samples=100)
        predictions = dbscan.fit_predict(points)
    elif method == 'hc':
        # hierarchical clustering initialized with linkage='complete' and affinity='euclidean'
        linkages = ['ward', 'average', 'complete', 'single']
        hc = AgglomerativeClustering(linkage=linkages[2], n_clusters=n, affinity='euclidean')
        predictions = hc.fit_predict(points)
    # return the predicted labels for every sample
    return predictions


