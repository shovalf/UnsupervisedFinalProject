"""
Code to perform initial analysis- We check for missing values and fill them with the median value, we plot correlation
matrix, the features' distribution and a box plot to visualize outliers.
"""

from plotting_utils import *
from models import *


def plot_my_correlation_matrix(data):
    """
    Plot correlation matrix between numeric features- in our case all features are numeric.
    :param data: Our data
    :return: Correlation matrix plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    mpl.rcParams['axes.titlesize'] = 22
    ax.set_title("Correlation Matrix Between Numeric Features", y=1.03)
    sns.heatmap(data.corr(), annot=True, cmap='twilight', xticklabels=data.columns, yticklabels=data.columns, center=0)
    fig.tight_layout()
    plt.xticks(rotation=40, ha='right')
    plt.show()


def main():
    """
    Main function to plot correlation matrix and features distribution.
    """
    data = pd.read_csv("CC GENERAL.csv")
    data.loc[(data['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].median()
    data.loc[(data['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].median()
    data = data.drop(['CUST_ID'], 1)
    plot_my_correlation_matrix(data)
    columns = data.columns
    plot_numeric_features_distribution(data, columns)
    do_boxplot(data)


# main()
