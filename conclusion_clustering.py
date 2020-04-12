"""
Code to get to conclusions from clustering for costumers segmentation. We choose features that represent best our
data set and apply clustering on this "reduced" data. We then make a pair-plot to represent connections between each
feature anc luster. This will help us divide the people into three different groups with different characteristics.
"""

from models import *
from plotting_utils import *


def main():
    # read the data
    data = pd.read_csv("CC GENERAL.csv")
    data.loc[(data['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].median()
    data.loc[(data['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].median()
    data = data.drop(['CUST_ID'], 1)

    # choose the features that represent best our data set
    best_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]
    # dataframe with best columns
    data_final = data[best_cols]

    # normalization
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data_final)
    data_final = pd.DataFrame(X, columns=best_cols)

    # apply KMeans clustering
    alg = KMeans(n_clusters=3)
    label = alg.fit_predict(data_final)

    # create a 'cluster' column
    data_final['cluster'] = label
    best_cols.append('cluster')

    # make a Seaborn pairplot
    sns.pairplot(data_final, hue='cluster')
    plt.show()
