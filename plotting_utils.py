"""
Plotting Utils, meaning functions for plotting for example the data after dimensionality reduction, clustered data,
missing values and more.
"""

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl


def plot_missing_values(data, save_fig=False):
    """
    Some features has missing values. This function plots all of them and displays the percentage of missing values.
    :param data: (pd.DataFrame) Our data set.
    :param save_fig: (bool):Specify whether the output figure will be saved.
    :return: plot
    """
    # Get percentage of missing values
    missing_values_columns = data.isnull().mean().sort_values(ascending=False) * 100
    missing_values = pd.DataFrame({'available': 100 - missing_values_columns, 'missing': missing_values_columns})

    # --- Plotting --- #

    figure, ax = plt.subplots(figsize=(9, 5))

    y_height = list(reversed(range(len(missing_values))))
    ax.barh(y_height, missing_values['available'], color='C0')
    ax.barh(y_height, missing_values['missing'], left=missing_values['available'], color='C3')
    ax.set_yticks(y_height)
    ax.set_yticklabels(missing_values.index.values)
    ax.set_ylabel("feature")
    ax.set_xlabel("percentage")
    ax.set_title("Features with missing values")
    ax.legend(['available', 'missing'])

    # Add percent text
    for p in ax.patches:
        if p.get_x() > 0:
            width = p.get_width()

            # Arguments: xPos, yPos, text, alignment
            ax.text(50,
                    p.get_y() + p.get_height() / 2,
                    '{:.2f}%'.format(width) + " missing",
                    ha="center", va="center")

    figure.tight_layout()
    plt.show()

    if save_fig:
        figure.savefig("Figures/missing_values.eps", format='eps')


def plot_sparse_features_table(data):
    """
    Display a table of features that have dominant values. Features with dominant values are defined as features who
    have a single value over more than 99% of the samples. We will call these features sparse features.
    :param data: (pd.DataFrame) Our data set
    :return: pandas.DataFrame: The index of this data frame contains the names of the sparse features that
            we have found. This data frame has one column, "Dominant value feature coverage (%)". It holds
            the percentage at which the dominant value occurs in the data set.
    """

    print("Check if there are features that contain the same value over too many instances")

    # --- Extract relevant information from our data --- #

    def modified_value_counts(series):
        return series.value_counts().values[0]

    dominant_values = data.apply(func=modified_value_counts, axis=0)
    dominant_values = dominant_values.sort_values(ascending=False)

    # Check which columns have a repeating value over more than 99% of the data set

    # if I do 99 it's an error because the biggets is 85% and nothing should be removed

    population_cut = 0 * len(data)  # exclusion threshold
    dominant_values = dominant_values[dominant_values > population_cut]
    dominant_values = (dominant_values / len(data) * 100).to_frame()  # convert to percentage
    dominant_values.rename(columns={0: 'Dominant value feature coverage (%)'}, inplace=True)

    # --- Plotting --- #

    fig, ax = plt.subplots(figsize=(7, 7))

    # Hide axes, show table only
    ax.axis('off')

    # Round to three decimal places
    cell_text = dominant_values.values
    cell_text = np.floor(cell_text * 1000) / 1000

    # Create the table
    ax.table(cellText=cell_text,
             rowLabels=dominant_values.index.values,
             colLabels=dominant_values.columns.values,
             cellLoc='center',
             loc='center')

    fig.tight_layout()
    plt.show()
    print("For the moment, before dropping outliers, we get that nothing should be removed.")

    return dominant_values


def plot_correlation_matrix(data):
    """
    Plot correlation matrix between numeric features- in our case all features are numeric.
    :param data: Our data
    :return: Correlation matrix plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Correlation Matrix Between Numeric Features", y=1.03)
    sns.heatmap(data.corr(), annot=True, cmap='twilight', xticklabels=data.columns, yticklabels=data.columns)
    plt.xticks(rotation=40)
    plt.show()


def plot_numeric_features_distribution(data, num_dtypes_columns):
    """
    Plot the distribution of numeric features. For all numeric features, for each value, we plot the density
    of that value.
    :param data: (DataFrame) Our data set
    :param num_dtypes_columns: (list) List of the names of the numeric features.
    :return: Plots of distributions
    """

    def get_bins(series):
        """Get the number of desired bins for this series."""

        bins = max(series) - min(series)
        bins = int(bins)

        if bins > 40:
            bins = 40
        if bins < 10:
            bins = 10

        return bins

    fig, ax = plt.subplots(5, 4, figsize=(16, 10))

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            k = i * ax.shape[1] + j  # track the current feature to plot

            # Make sure we do not exceed the number of features we have
            if k < len(num_dtypes_columns):
                curr_col_name = num_dtypes_columns[k]
                curr_col = data[curr_col_name]
                print(curr_col_name, "| bins =", get_bins(curr_col))  # print bin size

                sns.distplot(curr_col, bins=get_bins(curr_col), norm_hist=True, kde=False, ax=ax[i, j])
            else:
                ax[i, j].axis('off')

        ax[i, 0].set_ylabel("Density")

    fig.tight_layout()
    plt.show()


def plot_2d(points):
    """
    Plot 2D visualization of dimension reduction of every method- PCA, Kernel PCA t-SNE and 2 types of autoencoder.
    :param points: the data after dimension reduction (2D)
    :return: Plots
    """
    n = len(points)  # the number of desired plots
    fig, axes = plt.subplots(2, 3, figsize=(4 * n, 8))
    mpl.rcParams['axes.titlesize'] = 18
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    # Scatter each matrix of points
    for i in range(len(points)):
        if i == 0:
            # 2d pca
            axes[0, i].scatter(points[i][:, 0], points[i][:, 1])
            axes[0, i].set_xlabel("First Principal Component")
            axes[0, i].set_ylabel("Second Principal Component")
            axes[0, i].set_title("PCA - Projection into the first 2 PCs of the data")
        elif i == 1:
            # 2d kernel pca
            axes[0, i].scatter(points[i][:, 0], points[i][:, 1])
            axes[0, i].set_xlabel("First Principal Component")
            axes[0, i].set_ylabel("Second Principal Component")
            axes[0, i].set_title("Cosine Kernel")
        elif i == 2:
            # 2d t-sne
            axes[0, i].scatter(points[i][:, 0], points[i][:, 1])
            axes[0, i].set_title("T-SNE 2d Projection")
        elif i == 3:
            # 2d autoencoder 17->2
            axes[1, 0].scatter(points[i][:, 0], points[i][:, 1])
            axes[1, 0].set_title("Autoencoder 2d Dimension Reduction, 17 -> 2")
        else:
            # 2d autoencoder 14->2
            axes[1, 1].scatter(points[i][:, 0], points[i][:, 1])
            axes[1, 1].set_title("Autoencoder 2d Dimension Reduction, 14 -> 2")

    axes[1, 2].axis('off')
    return fig, axes


def plot_3d(points):
    """
    Plot 3D visualization of dimension reduction of every method- PCA, Kernel PCA and 2 types of autoencoder.
    :param points: the data after dimension reduction (2D)
    :return: Plots
    """
    n = len(points)  # the number of desired plots
    fig, axes = plt.scatter(2, 2, figsize=(5 * n, 4))

    # Scatter each matrix of points
    for i in range(len(points)):
        if i == 0:
            # 3d pca
            axes[0, i].scatter(points[i][:, 0], points[i][:, 1], points[i][:, 2])
            axes[0, i].set_title("PCA - Projection into the first 3 PCs of the data")
        elif i == 1:
            # 3d kernel pca
            axes[0, i].scatter(points[i][:, 0], points[i][:, 1], points[i][:, 2])
            axes[0, i].set_title("Cosine Kernel")
        elif i == 2:
            # 3d autoencoder, 17->3
            axes[1, 0].scatter(points[i][:, 0], points[i][:, 1], points[i][:, 2])
            axes[1, 0].set_title("Autoencoder 3d Dimension Reduction, 17 -> 3")
        else:
            # 3d autoencoder 14->3
            axes[1, 1].scatter(points[i][:, 0], points[i][:, 1], points[i][:, 2])
            axes[1, 1].set_title("Autoencoder 3d Dimension Reduction, 14 -> 2")

    return fig, axes


def plot_2d_pca_tsne(points, method):
    """
    Plot visualization of dimension reduction of 2D PCA and t-SNE.
    :param points: the data after dimension reduction (2D)
    :param method: the dimension reduction method that was used- PCA or t-SNE
    :return: Plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1])
    if method == 'PCA':
        ax.set_title("PCA - Projection into the first 2 PCs of the data")
        ax.set_xlabel("First Principal Component")
        ax.set_ylabel("Second Principal Component")
    elif method == 'ae':
        ax.set_title("Autoencoder 2d Dimension Reduction")
    else:
        ax.set_title("T-SNE 2d Projection")
    plt.show()


def plot_3d_pca(points):
    """
    Plot visualization of dimension reduction of 3D PCA.
    :param points: the data after dimension reduction (3D)
    :return:
    """
    result = pd.DataFrame(points, columns=['PCA%i' % i for i in range(3)])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], cmap="Set2_r", s=8)
    ax.set_xlabel("PC0")
    ax.set_ylabel("PC1")
    ax.set_zlabel("PC2")
    ax.set_title("PCA - Projection into the first 3 PCs of the numeric data")
    plt.show()


def plot_clustering(points, predictions, method, clustering, dim=2):
    """
    Plot the clustered data, each label has its own color.
    :param points: The reduced data after performing 2d/3d pca or tsne.
    :param predictions: The predicted labels from the clustering method
    :param method: 2d/3d PCA or TSNE
    :param clustering: clustering method- Kmeans, GMM, DBSCAN and HC.
    :param dim: The dimension of the reduced data
    :return: Representation plot
    """
    fig = plt.figure()
    if dim == 2:
        # if the dimension of the reduced data is 2
        ax = fig.add_subplot(111)
        ax.scatter(points[:, 0], points[:, 1], c=predictions, cmap='tab10', alpha=0.8, s=8)
        ax.set_title("{} on {} reduced numeric data".format(clustering, method))
    elif dim == 3:
        # if the dimension of the reduced data is 3
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=predictions, cmap='tab10', alpha=0.8, s=8)
        ax.set_title("{} on {} reduced numeric data".format(clustering, method))
    plt.show()


def apply_dendrogram(x_scaled):
    """
    Plot dendrogram to decide number of clusters by Hierarchical Clustering method.
    :param x_scaled: values of our data after normalization
    :return: Dendrogram
    """
    dendrogram = sch.dendrogram(sch.linkage(x_scaled, method = 'ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean Distances')
    plt.show()


def do_boxplot(data):
    """
    Plot a box plot for outliers visualization
    :param data: Our data set
    :return: A box plot
    """
    X = data.values
    names = data.columns
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X) # normalize
    scaled_df = pd.DataFrame(X_scaled, columns=names)
    plt.figure(figsize=(10, 10))
    plt.title("Box Plot")
    sns.boxplot(data=scaled_df)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.show()
