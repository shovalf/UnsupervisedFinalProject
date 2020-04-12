from sklearn.metrics import mutual_info_score
from plotting_utils import *
from sklearn.preprocessing import MinMaxScaler
from autoencoder import do_autoencoder
from models import *


def normalization(data, scaler=MinMaxScaler()):
    """
    Normalize the data by min-max normalization
    :param data: Our data set
    :param scaler: min-max scaler
    :return: normalized data values
    """
    X = data.values
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def change_data_one_feature(data, feature):
    """
    Remove the feature that is being used for labeling, sort the data by this feature value and return
    the new data.
    :param data: Our data set
    :param feature: The feature used for labeling
    :return: The new data after changing s explained above
    """
    factor = data[feature]
    data['factor'] = factor
    data = data.sort_values(by=['factor'])
    data.to_csv('with_factor.csv',index=False)
    test = pd.read_csv("with_factor.csv")
    test = test.drop(['factor'], axis=1)
    test = test.drop(feature, 1)
    return test


def change_data_autoencoder(data):
    """
    Remove the features that were used to create a new feature for labeling, sort the data by this new feature
    and remove it too.
    :param data: Our data set
    :return: The new data
    """
    names = data.columns
    encoded_data = do_autoencoder(data, names, len(names), 1)
    factor = encoded_data['factor_0.']
    data['factor'] = factor
    data = data.sort_values(by=['factor'])
    data.to_csv('with_factor.csv', index=False)
    test = pd.read_csv("with_factor.csv")
    test = test.drop(['factor'], axis=1)
    return test


def make_labels(test):
    """
    Given the new data (after we removed what is needed), apply a label for every sample- the data samples
    are sorted by the feature value and we want to separate it into 3 clusters with equal number of points,
    so for the first third give label 0, second third label 1 and lase third label 2.
    :param test: The new dataset after necessary features were removed.
    :return: The dataset with the labels
    """
    test['LABELS'] = np.arange(8950)
    num = int(8950/3)
    for i in range(8950):
        if i <= num:
            test.at[i, 'LABELS'] = 0
        elif num < i < 2*num:
            test.at[i, 'LABELS'] = 1
        else:
            test.at[i, 'LABELS'] = 2
    labels = test['LABELS']
    test = test.drop(['LABELS'], 1)
    return test, labels


def main(method, mission):
    """
    The main function. Apply labels to the datset and calculate mutual information index in order to compare
    between different mecthod of labeling and between the clustered data and the labeled data.
    :param method: method for dimensionality reduction after applying labels for the dataset (pca or autoencoder)
    :param mission: if mission == 1: the data is labeled by an existing feature, else: the data is labeled by
            all 17 features that were merged into one by an autoencoder.
    :return: Mutual information index as explained above
    """
    # read the data
    data = pd.read_csv(r"C:\Users\shova\Desktop\CC GENERAL.csv")
    data.loc[(data['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].median()
    data.loc[(data['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].median()
    data = data.drop(['CUST_ID'], 1)

    # change the data
    if mission == 1:
        new_data = change_data_one_feature(data, "PURCHASES_FREQUENCY")
    elif mission == 2:
        new_data = change_data_one_feature(data, "PURCHASES")
    else:
        new_data = change_data_autoencoder(data)
    new_data, labels = make_labels(new_data)

    if method == "ae":
        # autoencoder 14 -> dim
        scaler = MinMaxScaler()
        features = ['BALANCE_FREQUENCY', 'PRC_FULL_PAYMENT', 'TENURE', 'BALANCE']
        encoded_data = do_autoencoder(new_data, features, len(features), 1)
        data2 = new_data.copy()
        data2 = data2.drop(features, 1)
        for i in range(len(encoded_data.columns)):
            data2['factor_{}.'.format(i)] = encoded_data['factor_{}.'.format(i)]
        names2 = data2.columns
        data_scaled = scaler.fit_transform(data2)
        data_scaled = pd.DataFrame(data_scaled, columns=names2)
        points_2_dim = do_autoencoder(data_scaled, names2, len(names2), 2)
        points_2_dim = points_2_dim.values
    else:
        X_scaled = normalization(new_data)
        points_2_dim = perform_pca(X_scaled, 2)

    plot_2d_pca_tsne(points_2_dim, method)
    if method == 'ae':
        plot_clustering(points_2_dim, labels, 'AutoEncoder', 'With Labels', 2)
    else:
        plot_clustering(points_2_dim, labels, 'PCA', 'With Labels', 2)

    kmeans_predict = perform_clustering_methods('kmeans', points_2_dim, 3)
    if method == 'ae':
        plot_clustering(points_2_dim, kmeans_predict, 'AutoEncoder', 'K-means', 2)
    else:
        plot_clustering(points_2_dim, kmeans_predict, 'PCA', 'K-means', 2)

    if mission == 1:
        print("with PURCHASES_FREQUENCY")
    elif mission == 2:
        print("with PURCHASES")
    else:
        print("with 17 features merged as one")

    # check mutual information between k-means clustering and our labels
    mi_kmeans = mutual_info_score(labels, kmeans_predict)
    print("Mutual Information between K-means predictions and our labels is: ", mi_kmeans)

    # check mutual information between random labels and k-means clustering
    random_label = np.random.randint(0, 2, 8950)
    mi_random = mutual_info_score(random_label, kmeans_predict)
    print("Mutual Information between K-means predictions and random labels is: ", mi_random)


# here one can choose the dimensionality reduction method (PCA or autoencoder) and the requested labeling method,
# as explained above
# main('PCA', 1)
