# timeit

# Student Name : Neel Roshania
# Cohort       : 5

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

# use this space for all of your package imports

import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import random as rand
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm

################################################################################
# Class and method declarations
################################################################################

class ModelResults():
    
    # Class to append results of modeling scenarios into a convenient wrapper
    #    - Assumes pandas as a preloaded resource
    def __init__(self, columns):
        self.columns = columns
        self.df = pd.DataFrame(columns = self.columns)
    
    # save results into class
    def save(self, model_dict_outcome):
        self.df = pd.concat([self.df, pd.DataFrame(model_dict_outcome)])
    
    # display results
    def display(self):
        return self.df
    
    # export results
    def export_csv(self, location):
        self.location = location
        self.df.to_csv(self.location)

class NormalityTest():
     
    # Determine whether the supplied function could be sampled from a Gaussian distribution
    #   - Normality tests have been validated by 100 randomly distributed normal floats
    #   - pandas is a pre-assumed dependency
    #      - NormalityTest(np.random.randn(100), 0.05, 0).get_result()
    #   - Source: https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
    
    from scipy import stats
    from numpy.random import seed, randn, seed
 
    def __init__(self, arr, alpha, dataframe, predictor_str):
        self.arr = arr
        self.alpha = alpha
        self.dataframe = dataframe
        self.predictor_str = predictor_str
        return None
    
    def check_normal_dist(self):
        return {
            'is_normally_distributed': {
                'shapiroWilk': self.shapiroWilk(self.alpha),
                'd_agostinoK2': self.d_agostinoK2(self.alpha),
                'anderson_darling': self.anderson_darling(self.alpha)},
        }
        
    def shapiroWilk(self, alpha):
        stat, p = self.stats.shapiro(self.arr)
        if p > alpha:
            return 'Gaussian (fail to reject H0)'
        else:
            return 'Not Gaussian (reject H0)'
    
    def d_agostinoK2(self, alpha):
        stat, p = self.stats.normaltest(self.arr)
        if p > alpha:
            return 'Gaussian (fail to reject H0)'
        else:
            return 'Not Gaussian (reject H0)'
        
    def anderson_darling(self, alpha):
        
        outcome = {
            "statistic": None,
            "percentile": [],
            "sl_cv": [],
            "result": []
        }
        # normality test
        result = self.stats.anderson(self.arr)
        statistic = result.statistic
        p = 0

        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < result.critical_values[i]:
                outcome['sl_cv'].append([sl, cv])
                outcome['result'].append(["Normal (Fail to reject H0)"])
            else:
                outcome['sl_cv'].append([sl, cv])
                outcome['result'].append(["Not Normal (Reject H0)"])
                
        outcome['statistic'] = statistic
        return outcome
    
    def check_population_mean(self, pop_mean):
        
        # NH: population mean is zero
        
        outcome = {}
        data = self.arr
        tset, pval = self.stats.ttest_1samp(data, pop_mean) # t-test
        if pval < 0.05:    # alpha value is 0.05 or 5%
            outcome['status'] = "Reject Null hypothesis"
        else:
            outcome['status'] = "Accept Null hypothesis"
        outcome['sample_mean'] = np.mean(data)
        return outcome
    
    def get_correlation_matrix(self):
        
        # arr and dataframe will be joined to produce correlation matrix
        return self.dataframe.join(pd.DataFrame({'arr': self.arr})).corr()
    
    
    def check_homoscedasticity(self, model):
        
        # generate a qqplot to visually inspect quantiles that break normality
        
        # We can also use two statistical tests: Breusch-Pagan and Goldfeld-Quandt. In both of them, the null hypothesis assumes homoscedasticity.
        #    - A p-value below a certain level (like 0.05) indicates we should reject the null in favor of heteroscedasticity.
        #    - Source: https://towardsdatascience.com/verifying-the-assumptions-of-linear-regression-in-python-and-r-f4cd2907d4c0
 
        import pylab 
        import scipy.stats as stats

        measurements = model.resid  
        stats.probplot(measurements, dist="norm", plot=pylab)
        pylab.show()
    
    
    def get_vif(self):
        # Variance inflation factor to measure collinearity among features

        feature_name = []
        vif_value = []
        rsq_value = []
        self.df_vif = self.dataframe.copy()
        if self.predictor_str in self.df_vif.columns:
            self.df_vif = self.df_vif.drop([self.predictor_str], axis=1)
        
        for i in range(0, len(self.df_vif.columns)):
            # prepare features
            X = self.df_vif.loc[:, self.df_vif.columns != self.df_vif.columns[i]]
            y = self.df_vif.loc[:, self.df_vif.columns == self.df_vif.columns[i]]

            # Regress feature on every other feature
            lr = LinearRegression().fit(X, y)

            # Determine rsq
            rsq = lr.score(X, y)
            if rsq != 1:
                vif = round(1 / (1 - rsq), 2)
            else:
                vif = float("inf")

            feature_name.append(self.df_vif.columns[i])
            rsq_value.append(rsq)
            vif_value.append(vif)

        return pd.DataFrame({
                "r_squared": rsq_value,
                "vif": vif_value },
            index = feature_name
        ).sort_values(by="vif")

def skl_train_test_pred_results(model, fit_X, fit_y, test_X, test_y, pred_X):
#     quick method to output model, train and test scores
    
    # Fit and score
    mod_fit = model.fit(fit_X, fit_y)
    mod_train_score = model.score(fit_X, fit_y)
    mod_test_score = model.score(test_X, test_y)
    y_pred = mod_fit.predict(pred_X)
    
    return mod_fit, mod_train_score, mod_test_score, y_pred

def skl_mod_confusion_matrix_roc_auc(y_true, y_pred, y_score):
    # Thus in binary classification, 
#         - C00, True  Negatives 
#         - C01, False Positive
#         - C10, False Negative  
#         - C11  True  Positive

    cm = confusion_matrix(y_true, y_pred)
    cm_score = {
        'TN': cm[0, 0],
        'FP': cm[0, 1],
        'FN': cm[1, 0],
        'TP': cm[1, 1]
    }
    
    return cm_score, roc_auc_score(y_true, y_score)

def vim_multicollinearity(df):
    # Variance inflation factor to measure collinearity among features
    feature_name = []
    vif_value = []
    rsq_value = []

    for i in range(0, len(df.columns)):
        # prepare features
        X = df.loc[:, df.columns != df.columns[i]]
        y = df.loc[:, df.columns == df.columns[i]]

        # Regress feature on every other feature
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

def statsmodel_ols_lgst(library, model_type, X_train, X_test, y_train, y_test, class_likelihood):
    # Method to automate process of retrieving performance parameters
    
    model_spec = {
        'ols': sm.OLS(y_train, X_train).fit(),
        'lgst': sm.Logit(y_train, X_train).fit(method='ncg')
    }
    
    model_spec_sklearn = {
        'ols': LinearRegression().fit(X_train, y_train),
        'lgst': LogisticRegression().fit(X_train, y_train)
    }
    
    if library == 'statsmodels':
        model = model_spec[model_type]

        # apply class heuristic
        #     - Any class less than 0.5: 0
        #     - ANy class greater than 0.5: 1
        y_pred = model.predict(X_test)
        y_pred = np.array(y_pred)
        y_pred[y_pred < class_likelihood] = 0
        y_pred[y_pred >= class_likelihood] = 1
        y_pred = y_pred.astype(int)
        
        # Get significant features
        mod_sig_features = list(model.pvalues[model.pvalues <= 0.05].index.values)
        
    elif library == 'sklearn':
        model = model_spec_sklearn[model_type]
        
        # apply class heuristic
        y_pred = model.predict(X_test)
        y_pred = np.array(y_pred)
        y_pred[y_pred < class_likelihood] = 0
        y_pred[y_pred >= class_likelihood] = 1
        y_pred = y_pred.astype(int)
        
        # Get significant features
        mod_sig_features = []
        
    else:
        raise Exception("KindError: Library argument is incorrect.")

    # Get scores: output interpreted as probability of predicting class 1
    mod_cm_score, mod_auc_score = skl_mod_confusion_matrix_roc_auc(y_test, y_pred, y_pred)
    mod_precision = mod_cm_score['TP']/(mod_cm_score['TP']+mod_cm_score['FP'])
    
    return model, mod_cm_score, mod_auc_score, mod_precision, mod_sig_features

# outlier and threshold mapping  
def threshold_outlier_flagging(dataframe, feature, threshold_val, threshold_type, bound):
#     automate outlier and flagging process
    if threshold_type == 'outlier':
        if bound == 'high':
            _out_label = "out_{}_hi".format(feature)
            dataframe.loc[:, _out_label] = 0
            for index, val in dataframe.iterrows(): 
                if dataframe.loc[index, feature] > threshold_val:
                    dataframe.loc[index, _out_label] = 1
        elif bound == "low":
            _out_label = "out_{}_lo".format(feature)
            dataframe.loc[:, _out_label] = 0
            for index, val in dataframe.iterrows(): 
                if dataframe.loc[index, feature] < threshold_val:
                    dataframe.loc[index, _out_label] = 1
        else:
            print("Outlier flagging failed")
            
    elif threshold_type == 'threshold':
        if bound == 'high':
            _out_label = "flag_{}_hi".format(feature)
            dataframe.loc[:, _out_label] = 0
            for index, val in dataframe.iterrows(): 
                if dataframe.loc[index, feature] > threshold_val:
                    dataframe.loc[index, _out_label] = 1
        elif bound == "low":
            _out_label = "flag_{}_lo".format(feature)
            dataframe.loc[:, _out_label] = 0
            for index, val in dataframe.iterrows(): 
                if dataframe.loc[index, feature] < threshold_val:
                    dataframe.loc[index, _out_label] = 1
        else:
            print("Threshold flagging failed")
            
    elif threshold_type == 'trend':
        _out_label = "trend_{}".format(feature)
        dataframe.loc[:, _out_label] = 0
        for index, val in dataframe.iterrows(): 
            if dataframe.loc[index, feature] == threshold_val:
                dataframe.loc[index, _out_label] = 1

# plot a ncol x 4 row matrix of distributions
def plot_box_facets(_df, df, fig_rc, viz_type, y, cat_x):
    g_count = 0
    
    if viz_type == 'box':
        if y not in cat_x:
            a = plt.subplot(fig_rc[0], fig_rc[1], g_count+1)

            # plot a facet
            box_plot = sns.boxplot(x=cat_x, y=y, data=df)
            g_count += 1
            
            # despine figure
            sns.despine()
            return box_plot
    else:
        raise Exception("Failed to generated Box Plot")

################################################################################
# Load Data
################################################################################

# use this space to load the original dataset
# MAKE SURE TO SAVE THE ORIGINAL FILE AS original_df
# Example: original_df = pd.read_excel('Apprentice Chef Dataset.xlsx')

df_original = pd.read_csv("data/df_feature_engineering.csv")
df_optimization = df_original.copy()
df_optimization = df_optimization.drop("Unnamed: 0", axis=1)

# feature name correction
df_optimization.rename(columns={'MOBILE_NUMBER': 'MOBILE_REGISTRATION'}, inplace=True)

# preparing response variable data
predictor = 'HWT_SUBSCRIBER'
target = df_optimization.loc[:, predictor]
all_features = df_optimization.loc[:, df_optimization.columns.isin([predictor]) == False]


################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# use this space for all of the feature engineering that is required for your
# final model

# if your final model requires dataset standardization, do this here as well

# drop redundent features
df_optimization.drop(
            ["flag_WEEKS_SUBSCRIBED_TO_WEEKLY_PLAN_CNT_lo",
            "NAME",
            "EMAIL",
            "FIRST_NAME",
            "FAMILY_NAME"], 
            axis = 1, inplace = True)

# feature engineer weakly plan subscriber types and others
# Ratio of total meals ordered and weeks subscribed to weekly plan are not all perfectly divisible by 3 or 5
#     - Assumption: Weekly plan subscription is a stratified discount plan as opposed to a weekly delivery plan
# If the total number of meals order is greater 
#     - Than a multiple of 3 weeks subscribed to the weekly plan -> Basic subscriber
#     - Than a multiple of 5 5 of weeks subscribed to the weekly -> Premium subscriber
#     - Otherwise inactive weekly

# instantiate features
df_optimization["fe_WP_NOT_SUB"] = 0
df_optimization["fe_WP_INACTIVE_SUB"] = 0 
df_optimization["fe_WP_BASIC_SUB"] = 0 
df_optimization["fe_WP_PREMIUM_SUB"] = 0

# Generate features
for index, val in df_optimization.iterrows(): 
    
    # stratify by type of weekly plan subscriber
    if df_optimization.loc[index, "WEEKS_SUBSCRIBED_TO_WEEKLY_PLAN_CNT"]*3 > df_optimization.loc[index, "TOTAL_MEALS_ORDERED"]:
        df_optimization.loc[index, "fe_WP_BASIC_SUB"] = 1
    elif df_optimization.loc[index, "WEEKS_SUBSCRIBED_TO_WEEKLY_PLAN_CNT"]*5 > df_optimization.loc[index, "TOTAL_MEALS_ORDERED"]:
        df_optimization.loc[index, "fe_WP_PREMIUM_SUB"] = 1
    else:
        df_optimization.loc[index, "fe_WP_INACTIVE_SUB"] = 1
    
    # feature out customers that are not subscribers of a weekly meal plan
    if df_optimization.loc[index, "WEEKS_SUBSCRIBED_TO_WEEKLY_PLAN_CNT"] == 0:
        df_optimization.loc[index, "fe_WP_NOT_SUB"] = 1

# intialize feature
df_optimization.rename(columns={"fe_WP_NOT_SUB": "fe_ONLY_HWT_CUSTOMER"}, inplace=True)


################################################################################
# Trend, outlier and flagging
################################################################################

# Seperate customers based on type
df_dual_WP_HWT = df_optimization[df_optimization["fe_ONLY_HWT_CUSTOMER"] == 0]
df_only_HWT = df_optimization[df_optimization["fe_ONLY_HWT_CUSTOMER"] == 1]

# copy dataframes
df_flagged = df_optimization.copy()

# export data for further analysis
df_dual_WP_HWT.to_csv("existing_customers.csv")
df_dual_WP_HWT.to_csv("new_HWT_customers.csv")

# autoflagging all customers
threshold_outlier_flagging(df_flagged, "REVENUE", 4200, "outlier", "high")
threshold_outlier_flagging(df_flagged, "TOTAL_MEALS_ORDERED", 175, "outlier", "high")
threshold_outlier_flagging(df_flagged, "UNIQUE_MEALS_PURCH", 12.5, "outlier", "high")
threshold_outlier_flagging(df_flagged, "CUSTOMER_SERVICE_TICKETS_CNT", 12, "outlier", "high")
threshold_outlier_flagging(df_flagged, "MEALS_CANCEL_BEFORE_NOON", 6, "outlier", "high")
threshold_outlier_flagging(df_flagged, "WEEKS_SUBSCRIBED_TO_WEEKLY_PLAN_CNT", 33, "outlier", "high")
threshold_outlier_flagging(df_flagged, "ORDERS_DELIVERED_BEFORE_DELIVERY_CNT", 7, "outlier", "high")
threshold_outlier_flagging(df_flagged, "ORDERS_DELIVERED_AFTER_DELIVERY_CNT", 8, "outlier", "high")
threshold_outlier_flagging(df_flagged, "FRIDGE_LOCKER_IN_PACKAGE_ROOM", 1, "trend", "hight")
threshold_outlier_flagging(df_flagged, "PRCNT_FOLLOWED_MEAL_RECOM_WEB_MOBILE", 30, "threshold", "high")
threshold_outlier_flagging(df_flagged, "SECONDS_WATCHING_PREP_VID_AVG", 280, "outlier", "high")
threshold_outlier_flagging(df_flagged, "LARGEST_ORDER_SIZE", 8, "outlier", "high")
threshold_outlier_flagging(df_flagged, "MASTER_CLASSES_ATTENDED_CNT", 2, "outlier", "high")
threshold_outlier_flagging(df_flagged, "MEAL_RATING_MEDIAN", 4, "outlier", "high")
threshold_outlier_flagging(df_flagged, "SITE_CLICKS_PER_VISIT_AVG", 8, "outlier", "low")
threshold_outlier_flagging(df_flagged, "PHOTOS_VIEWED_COUNT", 400, "outlier", "high")
threshold_outlier_flagging(df_flagged, "fe_MEAL_CHOICE_SPECIFICITY", 0.28, "outlier", "high")
threshold_outlier_flagging(df_flagged, "fe_SITE_CLICK_RATE", 0.4, "outlier", "high")

# Define feature sets
continuous_features = [
    'REVENUE', 
    'SITE_VISIT_TIME_PER_VISIT_AVG', 
    'PRCNT_FOLLOWED_MEAL_RECOM_WEB_MOBILE', 
    'SECONDS_WATCHING_PREP_VID_AVG'
                    ]

# # MOBILE_LOGINS_CNT dropped, collinearity
count_features = [
    'TOTAL_MEALS_ORDERED', 
    'UNIQUE_MEALS_PURCH', 
    'CUSTOMER_SERVICE_TICKETS_CNT',
    'PRODUCT_CATEGORIES_VIEWED', 
    'MEALS_CANCEL_BEFORE_NOON', 
    'MEALS_CANCEL_AFTER_NOON',
    "PC_LOGINS_CNT", 
    "WEEKS_SUBSCRIBED_TO_WEEKLY_PLAN_CNT",
    "ORDERS_DELIVERED_BEFORE_DELIVERY_CNT", 
    "ORDERS_DELIVERED_AFTER_DELIVERY_CNT",
    "MASTER_CLASSES_ATTENDED_CNT", 
    'SITE_CLICKS_PER_VISIT_AVG', 
    'PHOTOS_VIEWED_COUNT'
                    ]


boolean_features = [
    'CROSS_SELL_SUCCESS', 
    'SPECIFIED_TASTE_AND_PREFERENCES',
    'PACKAGE_ROOM_AT_CUSTOMER', 
    'FRIDGE_LOCKER_IN_PACKAGE_ROOM', 
                   ]

discrete_features = [
    'MOBILE_REGISTRATION', 
    'LARGEST_ORDER_SIZE', 
    'MEAL_RATING_MEDIAN'
                    ]

# make copy of dataframe
df_transformed_cont_scaled_count_disc = df_flagged.copy()

# Tranform continuous by natural log
for i in df_transformed_cont_scaled_count_disc.loc[:, continuous_features]:
    df_transformed_cont_scaled_count_disc.loc[:, i] = np.log(df_transformed_cont_scaled_count_disc.loc[:, i]).replace({-float('inf'): 0.01})

# prepare dataframe for scaling
df_to_scale = all_features.loc[:, count_features + discrete_features]

# Scale count and discrete
scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(df_to_scale)
df_to_scale = pd.DataFrame(scaler.transform(df_to_scale), columns=df_to_scale.columns)

# merge scaled df with other fields
df_transformed_cont_scaled_count_disc.loc[:, df_to_scale.columns] = df_to_scale

################################################################################
# Train/Test Split
################################################################################

# set predetermined optimal features
opt_features = ['MOBILE_REGISTRATION',
                'MEALS_CANCEL_BEFORE_NOON',
                'SPECIFIED_TASTE_AND_PREFERENCES',
                'MOBILE_LOGINS_CNT',
                'PRCNT_FOLLOWED_MEAL_RECOM_WEB_MOBILE',
                'fe_WP_PREMIUM_SUB',
                'flag_PRCNT_FOLLOWED_MEAL_RECOM_WEB_MOBILE_hi',
                'fe_CUSTOMER_COMPANY_protonmail',
                'fe_CUSTOMER_COMPANY_travelers',
                'fe_CUSTOMER_DOMAIN_professional_domain',
                predictor]

# slice dataframe by significant features of last linear regression model
df_transformed_cont_scaled_count_disc.to_csv("df_pre_final_model_selection.csv")
df_final_optimization = df_transformed_cont_scaled_count_disc.copy()
df_final_optimization = df_final_optimization.loc[:, df_final_optimization.columns.isin(opt_features)]

# Split training and testing data for cross validation
X_train, X_test, y_train, y_test = train_test_split(
            df_final_optimization.loc[:, df_final_optimization.columns.isin([predictor]) == False],
            df_final_optimization.loc[:, predictor],
            test_size = 0.25,
            random_state = 222)

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

train_score = []
test_score = []

# use this space to instantiate, fit, and predict on your final model
lr_model_optimal, lr_optimal_cm_score, lr_optimal_auc_score, lr_mod_optimal_precision, sig_features = statsmodel_ols_lgst('sklearn', 'ols', X_train, X_test, y_train, y_test, 0.7)
train_score = lr_model_optimal.score(X_train, y_train)
test_score = lr_model_optimal.score(X_test, y_test)

################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)

test_score = lr_optimal_auc_score
print(test_score) 


