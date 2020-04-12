from plotting_utils import *


def print_relevant_information(data):
    print("Dataset shape:", data.shape)
    print("Data Information:")
    print(data.info())
    print("Check if there are NULL variables:")
    print(data.isna().sum().sort_values(ascending=False))
    print("There are 313 null in MINIMUM_PAYMENTS and 1 null in CREDIT_LIMIT.")


def deal_with_missing_values(data):
    print("Impute missing values with the median value and check again for missing values:")
    # impute with median
    data.loc[(data['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].median()
    data.loc[(data['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].median()
    print(data.isna().sum())
    print("There are no missing values now")
    return data


def main():
    data = pd.read_csv("CC GENERAL.csv")
    print_relevant_information(data)
    plot_missing_values(data)
    data = deal_with_missing_values(data)
    # drop ID column
    data = data.drop('CUST_ID', 1)
    print("Final dataset shape:", data.shape)


#main()
