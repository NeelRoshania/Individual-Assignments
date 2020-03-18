# Neel Roshania - A1 Regression-Based Analysis (Individual)
# The aim of this script is to produce a model that can predict Appentice Chef's Revenue based on data presented

# Ingested: March 2020
# Owner: Apprentics Chef 

# Library dependencies
import pandas as pd
import numpy as np

# Modeling
import statsmodels.api as sm
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LinearRegression # linear regression (scikit-learn)
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression

# Key functions
    
# outlier and threshold mapping  
def threshold_outlier_flagging(dataframe, feature, threshold_val, threshold_type, bound):
#     automate outlier and flagging process
    if threshold_type == 'outlier':
        if bound == 'high':
            _out_label = "out_{}_hi".format(feature)
            dataframe[_out_label] = 0
            for index, val in dataframe.iterrows(): 
                if dataframe.loc[index, feature] > threshold_val:
                    dataframe.loc[index, _out_label] = 1
        elif bound == "low":
            _out_label = "out_{}_lo".format(feature)
            dataframe[_out_label] = 0
            for index, val in dataframe.iterrows(): 
                if dataframe.loc[index, feature] < threshold_val:
                    dataframe.loc[index, _out_label] = 1
        else:
            print("Outlier flagging failed")
    elif threshold_type == 'threshold':
        if bound == 'high':
            _out_label = "flag_{}_hi".format(feature)
            dataframe[_out_label] = 0
            for index, val in dataframe.iterrows(): 
                if dataframe.loc[index, feature] > threshold_val:
                    dataframe.loc[index, _out_label] = 1
        elif bound == "low":
            _out_label = "flag_{}_lo".format(feature)
            dataframe[_out_label] = 0
            for index, val in dataframe.iterrows(): 
                if dataframe.loc[index, feature] < threshold_val:
                    dataframe.loc[index, _out_label] = 1
        else:
            print("Threshold flagging failed")
            
# Variance inflation factor to measure collinearity among features
def vim_multicollinearity(df):
    feature_name = []
    vif_value = []
    rsq_value = []

    for i in range(0, len(df.columns)):
        # prepare features
        X = df.loc[:, df.columns != df.columns[i]]
        y = df.loc[:, df.columns == df.columns[i]]

        # Regress on each feature
        lr = LinearRegression().fit(X, y)
        
        # Determine rsq
        rsq = lr.score(X, y)
        if rsq != 1:
            vif = round(1 / (1 - rsq), 2)
        else:
            vif = float("inf")

        feature_name.append(df.columns[i])
        rsq_value.append(rsq)
        vif_value.append(vif)

    return pd.DataFrame({
            "r_squared": rsq_value,
            "vif": vif_value },
        index = feature_name
    ).sort_values(by="vif")

# import data
file_data = "data/Apprentice_Chef_Dataset.xlsx"
file_definitions = "data/Apprentice_Chef_Data_Dictionary.xlsx"

original_df = pd.read_excel(file_data)
df_data = original_df.copy()
df_definitions = pd.read_excel(file_definitions)

df = df_data.copy()
df_def = df_definitions.copy()


### Dictionary-data conflicts & remedying
# convert to float
df.loc[:, "FOLLOWED_RECOMMENDATIONS_PCT"] = df.loc[:, "FOLLOWED_RECOMMENDATIONS_PCT"].astype(float)

# rename columns for ease of interpretation
renamed_columns = {
    "CONTACTS_W_CUSTOMER_SERVICE":  "CUSTOMER_SERVICE_TICKETS_CNT",
    "AVG_TIME_PER_SITE_VISIT":      "SITE_VISIT_TIME_PER_VISIT_AVG",
    "CANCELLATIONS_BEFORE_NOON":    "MEALS_CANCEL_BEFORE_NOON",
    "CANCELLATIONS_AFTER_NOON":     "MEALS_CANCEL_AFTER_NOON",
    "TASTES_AND_PREFERENCES":       "SPECIFIED_TASTE_AND_PREFERENCES",
    "PC_LOGINS":                    "PC_LOGINS_CNT",
    "MOBILE_LOGINS":                "MOBILE_LOGINS_CNT",
    "WEEKLY_PLAN":                  "WEEKS_SUBSCRIBED_TO_WEEKLY_PLAN_CNT",
    "EARLY_DELIVERIES":             "ORDERS_DELIVERED_BEFORE_DELIVERY_CNT",
    "LATE_DELIVERIES":              "ORDERS_DELIVERED_AFTER_DELIVERY_CNT",
    "PACKAGE_LOCKER":               "PACKAGE_ROOM_AT_CUSTOMER",
    "REFRIGERATED_LOCKER":          "FRIDGE_LOCKER_IN_PACKAGE_ROOM",
    "FOLLOWED_RECOMMENDATIONS_PCT": "PRCNT_FOLLOWED_MEAL_RECOM_WEB_MOBILE",
    "AVG_PREP_VID_TIME":            "SECONDS_WATCHING_PREP_VID_AVG",
    "MASTER_CLASSES_ATTENDED":      "MASTER_CLASSES_ATTENDED_CNT",
    "MEDIAN_MEAL_RATING":           "MEAL_RATING_MEDIAN",
    "AVG_CLICKS_PER_VISIT":         "SITE_CLICKS_PER_VISIT_AVG",
    "TOTAL_PHOTOS_VIEWED":          "PHOTOS_VIEWED_COUNT"
}

df.rename(columns=renamed_columns, inplace=True)

# Define field types
continuous_features = pd.DataFrame(columns=[
    'REVENUE', 'SITE_VISIT_TIME_PER_VISIT_AVG', 'PRCNT_FOLLOWED_MEAL_RECOM_WEB_MOBILE', 
    'SECONDS_WATCHING_PREP_VID_AVG'
                    ])

string_features = pd.DataFrame(columns=[
    'NAME', 'EMAIL', 'FIRST_NAME','FAMILY_NAME'
                    ])

boolean_features = pd.DataFrame(columns=[
    'CROSS_SELL_SUCCESS', 'SPECIFIED_TASTE_AND_PREFERENCES',
    'PACKAGE_ROOM_AT_CUSTOMER', 'FRIDGE_LOCKER_IN_PACKAGE_ROOM', 
                   ])

count_features = pd.DataFrame(columns=[
    'TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH', 'CUSTOMER_SERVICE_TICKETS_CNT',
    'PRODUCT_CATEGORIES_VIEWED', 'MEALS_CANCEL_BEFORE_NOON', 'MEALS_CANCEL_AFTER_NOON',
    "PC_LOGINS_CNT", "MOBILE_LOGINS_CNT", "WEEKS_SUBSCRIBED_TO_WEEKLY_PLAN_CNT",
    "ORDERS_DELIVERED_BEFORE_DELIVERY_CNT", "ORDERS_DELIVERED_AFTER_DELIVERY_CNT",
    "MASTER_CLASSES_ATTENDED_CNT", 'SITE_CLICKS_PER_VISIT_AVG', 'PHOTOS_VIEWED_COUNT'
                    ])

discrete_features = pd.DataFrame(columns=[
    'MOBILE_NUMBER', 'LARGEST_ORDER_SIZE', 'MEAL_RATING_MEDIAN'
                    ])

professional_domain = [
    'mmm',
    'amex',
    'apple',
    'boeing',
    'caterpillar',
    'chevron',
    'cisco',
    'cocacola',
    'disney',
    'dupont',
    'exxon',
    'ge',
    'goldmansacs',
    'homedepot',
    'ibm',
    'intel',
    'jnj',
    'jpmorgan',
    'mcdonalds',
    'merck',
    'microsoft',
    'nike',
    'pfizer',
    'pg',
    'travelers',
    'unitedtech',
    'unitedhealth',
    'verizon',
    'visa',
    'walmart'
]

personal_domain = [
    'gmail',
    'yahoo',
    'protonmail',
]

junk_domain = [
    'me',
    'aol',
    'hotmail',
    'live',
    'msn',
    'passport',
]

# This classification was later found to produce a bug. Quick fix done to change get_unknown_domain key
unknown_domain = [
    float('nan')
]

company_domain_type = { 
    'com': 'for_profit',
    'org': 'non-profit'
}

get_professional_domain = dict(zip(professional_domain, ["professional_domain" for i in range(len(professional_domain))]))
get_personal_domain = dict(zip(personal_domain, ["personal_domain" for i in range(len(personal_domain))]))
get_junk_domain = dict(zip(junk_domain, ["junk_domain" for i in range(len(junk_domain))]))
get_unknown_domain = dict(zip(unknown_domain, ["for_proft_domain" for i in range(len(unknown_domain))]))

# Missing value inspection
# flag missing values
df["missing_FAMILY_NAME"] = 0
missing_FAMILY_NAME_index = df[df.FAMILY_NAME.isnull() == True].index.to_list()
df.loc[missing_FAMILY_NAME_index, "missing_FAMILY_NAME"] = 1

# Copy of the dataset to prevent overwritting
df.to_csv("df_missing_out.csv")
df_fe = df.copy()

# perform fe calculations
df_fe["fe_MEAL_CHOICE_SPECIFICITY"] = df_fe["UNIQUE_MEALS_PURCH"]/df_fe["TOTAL_MEALS_ORDERED"]
df_fe["fe_SITE_CLICK_RATE"] = df_fe["SITE_CLICKS_PER_VISIT_AVG"]/df_fe["SITE_VISIT_TIME_PER_VISIT_AVG"]

# rename file
df_fe.rename(
    columns = {
    "CROSS_SELL_SUCCESS": "HWT_SUBSCRIBER"},
    inplace = True
)

# A large increase in revenue between 0 and 15 represents the cross-sell period
df_fe["fe_SUBSCRIPTIONS_SYNERGY"] = 0
for index, val in df_fe.iterrows(): 
    if df_fe.loc[index, "WEEKS_SUBSCRIBED_TO_WEEKLY_PLAN_CNT"] < 15:
        df_fe.loc[index, "fe_SUBSCRIPTIONS_SYNERGY"] = 1

# threshold flagging
threshold_outlier_flagging(df_fe, "WEEKS_SUBSCRIBED_TO_WEEKLY_PLAN_CNT", 15, "threshold", "low")

# Extract company from email address
df_fe["fe_CUSTOMER_COMPANY"] = df_fe.loc[:, string_features.columns].EMAIL.str.extract('@(?P<company>\w+)\.com')

# replace email domains with domain types
df_fe["fe_CUSTOMER_DOMAIN"] = df_fe["fe_CUSTOMER_COMPANY"]
df_fe["fe_CUSTOMER_DOMAIN"] = df_fe["fe_CUSTOMER_COMPANY"].replace(
    {
        **get_professional_domain,
        **get_personal_domain, 
        **get_junk_domain,
        **get_unknown_domain
    }
)

df_fe["fe_CUSTOMER_COMPANY"] = df_fe["fe_CUSTOMER_COMPANY"].replace({float('nan'): 'ge'})

# explicitely define engineered features
features_eng = pd.DataFrame(columns=[
    'fe_MEAL_CHOICE_SPECIFICITY', 'fe_SITE_CLICK_RATE', 'fe_SUBSCRIPTIONS_SYNERGY',
    'fe_CUSTOMER_COMPANY', 'fe_CUSTOMER_DOMAIN',
                    ])

# Copy of the dataset to prevent overwritting
df_fe.to_csv("df_feature_engineering.csv")
df_out = df_fe.copy()
