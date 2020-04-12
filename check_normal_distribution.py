"""
Code to check if our data set has a normal distribution with two different tests: Jurque-Bera Test and Anderson Test.
"""

from plotting_utils import *
from scipy.stats import jarque_bera, anderson
from sklearn.preprocessing import MinMaxScaler


def main():
    # read the data
    data = pd.read_csv("CC GENERAL.csv")
    data.loc[(data['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].median()
    data.loc[(data['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].median()
    data = data.drop(['CUST_ID'], 1)

    names = data.columns.tolist()

    # normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=names)

    # apply Jurque-Bera Test
    print("Jurque-Bera Test:")
    for i in range(len(names)):
        X = data_scaled[names[i]]
        jb_value, p_value = jarque_bera(X)
        print("{} for {} feature, test value is {} and p-value is {}".format(i+1, names[i], jb_value, p_value))

    print("\n")

    # apply Anderson Test
    print("Anderson Test:")
    for i in range(len(names)):
        X = data_scaled[names[i]]
        a = anderson(X, dist='norm')
        print("for {} feature, test value is {}".format(names[i], a))
    print("\n")

