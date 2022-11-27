########################################
###### (Business Problem)
########################################

""" FLO wants to know which customer groups could be possibly bring more value compared to other groups within the given time period.
 When they know about how many customer could be more valuable, they can manage their storage activities, CRM campaigns, special offers
 and many others... We will be detecting the most potential clients for the different time of periods and report back to FLO.
"""

########################################
# Dataset Story
########################################

"""
The dataset consists of information obtained from the past shopping behaviors of customers who made their last purchases as OmniChannel (both online and offline shopper) in 2020 - 2021.
"""

# master_id: Unique client number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : The channel where the last purchase was made
# first_order_date : The date of the customer's first purchase
# last_order_date : The date of the last purchase made by the customer
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : The date of the last purchase made by the customer on the offline platform
# order_num_total_ever_online : The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer offline
# customer_value_total_ever_offline : The total price paid by the customer for offline purchases
# customer_value_total_ever_online : The total price paid by the customer for their online shopping
# interested_in_categories_12 : List of categories the customer has purchased from in the last 12 months

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
import researchpy as rp
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
from sklearn.preprocessing import MinMaxScaler
#pd.set_option('display.float_format', lambda x: '%.4f' % x)

from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_period_transactions
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv('/Users/serhandulger/Documents/Miuul_DS_Path/CRM Analitiği/FLO_RFM_Analizi/flo_data_20k.csv')

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### NA SUM #####################")
    print(dataframe.isnull().sum().sum())
    print("##################### Describe #####################")
    print(dataframe.describe())
    print("##################### Nunique #####################")
    print(dataframe.nunique())

check_df(df)

def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_ = (df[na_columns_].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

missing_values_analysis(df)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df["Total_Order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["Total_Payment"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.describe().T

extension = ["Total_Order","Total_Payment"]
num_cols = [col for col in df.columns if df[col].dtype != 'O' and col not in extension ]
num_cols

features = ["customer_value_total_ever_online","customer_value_total_ever_offline","order_num_total_ever_offline","order_num_total_ever_online"]

for i in features:
    replace_with_thresholds(df,i)

df.describe().T

import datetime as dt
df["first_order_date"] = pd.to_datetime(df["first_order_date"]).dt.normalize()
df["last_order_date"] = pd.to_datetime(df["last_order_date"]).dt.normalize()
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"]).dt.normalize()
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"]).dt.normalize()

df = df[~(df["Total_Payment"] == 0) | (df["Total_Order"] == 0)]

df["last_order_date"].max()

today_date = dt.datetime(2021,7,1)

new_df = pd.DataFrame({"CUSTOMER_ID": df["master_id"],
             "RECENCY_WEEKLY": ((df["last_order_date"] - df["first_order_date"]).dt.days)/7,
             "TENURE_WEEKLY": ((today_date - df["first_order_date"]).astype('timedelta64[D]'))/7,
             "FREQUENCY": df["Total_Order"],
             "MONETARY_AVG": df["Total_Payment"] / df["Total_Order"]})

##############################
# Forecasting expected purchases from customers in 3 months
##############################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(new_df['FREQUENCY'],
        new_df['RECENCY_WEEKLY'],
        new_df['TENURE_WEEKLY'])

new_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               new_df['FREQUENCY'],
                                               new_df['RECENCY_WEEKLY'],
                                               new_df['TENURE_WEEKLY'])

new_df.sort_values(by="exp_sales_3_month",ascending=False)[0:10]

##############################
# Forecasting expected purchases from customers in 6 months
##############################

new_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               new_df['FREQUENCY'],
                                               new_df['RECENCY_WEEKLY'],
                                               new_df['TENURE_WEEKLY'])

new_df.head()
# prediction validation
plot_period_transactions(bgf)
plt.show()


##############################
# GAMMA-GAMMA MODEL
##############################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(new_df['FREQUENCY'], new_df['MONETARY_AVG'])

new_df['EXP_AVERAGE_VALUE'] = ggf.conditional_expected_average_profit(new_df['FREQUENCY'], new_df['MONETARY_AVG'])

new_df.head()

##############################
# Calculation of CLTV with BG-NBD and GG model  - (6 MONTHS)
##############################

cltv = ggf.customer_lifetime_value(bgf,
                                   new_df['FREQUENCY'],
                                   new_df['RECENCY_WEEKLY'],
                                   new_df['TENURE_WEEKLY'],
                                   new_df['MONETARY_AVG'],
                                   time=6, # 6 MONTH
                                   freq="W",  # T's frequency information. (WEEKLY)
                                   discount_rate=0.01) # consider discounts that can be made over time (discount rate)

cltv = pd.DataFrame(cltv)

new_df["CLTV"] = cltv

new_df.sort_values(by='CLTV', ascending = False).head()

#########################################
# CREATING SEGMENTS BASED ON CLTV VALUES
#########################################

new_df["SEGMENT"] = pd.qcut(new_df["CLTV"], 4, labels=["D", "C", "B", "A"])

new_df.groupby("SEGMENT").agg({"count","mean","sum"})

#########################################
# FUNCTIONALIZE ALL PROCESS
#########################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
import researchpy as rp
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
from sklearn.preprocessing import MinMaxScaler
#pd.set_option('display.float_format', lambda x: '%.4f' % x)

from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_period_transactions
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv('/Users/serhandulger/Documents/Miuul_DS_Path/CRM Analitiği/FLO_RFM_Analizi/flo_data_20k.csv')

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def cltv_prediction(df):
    import datetime as dt
    df["Total_Order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["Total_Payment"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    df["first_order_date"] = pd.to_datetime(df["first_order_date"]).dt.normalize()
    df["last_order_date"] = pd.to_datetime(df["last_order_date"]).dt.normalize()
    df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"]).dt.normalize()
    df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"]).dt.normalize()

    num_cols = [col for col in df.columns if df[col].dtype != 'O' and col not in ["Total_Order", "Total_Payment"]]

    features = ["customer_value_total_ever_online", "customer_value_total_ever_offline", "order_num_total_ever_offline",
                "order_num_total_ever_online"]

    for i in features:
        replace_with_thresholds(df, i)

    df = df[~(df["Total_Payment"] == 0) | (df["Total_Order"] == 0)]

    today_date = dt.datetime(2021, 7, 1)

    new_df = pd.DataFrame({"CUSTOMER_ID": df["master_id"],
                           "RECENCY_WEEKLY": ((df["last_order_date"] - df["first_order_date"]).dt.days) / 7,
                           "TENURE_WEEKLY": ((today_date - df["first_order_date"]).astype('timedelta64[D]')) / 7,
                           "FREQUENCY": df["Total_Order"],
                           "MONETARY_AVG": df["Total_Payment"] / df["Total_Order"]})

    # SETTING UP BG-NBD MODEL
    bgf = BetaGeoFitter(penalizer_coef=0.001)

    bgf.fit(new_df['FREQUENCY'],
            new_df['RECENCY_WEEKLY'],
            new_df['TENURE_WEEKLY'])

    # Expected sales frequency in 3 months
    new_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                              new_df['FREQUENCY'],
                                              new_df['RECENCY_WEEKLY'],
                                              new_df['TENURE_WEEKLY'])

    # Expected sales frequency in 6 months
    new_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                              new_df['FREQUENCY'],
                                              new_df['RECENCY_WEEKLY'],
                                              new_df['TENURE_WEEKLY'])

    # Prediction validation
    plot_period_transactions(bgf)
    plt.show()

    # SETTING UP GAMMA-GAMMA MODEL
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(new_df['FREQUENCY'], new_df['MONETARY_AVG'])
    new_df['EXP_AVERAGE_VALUE'] = ggf.conditional_expected_average_profit(new_df['FREQUENCY'], new_df['MONETARY_AVG'])

    cltv = ggf.customer_lifetime_value(bgf,
                                       new_df['FREQUENCY'],
                                       new_df['RECENCY_WEEKLY'],
                                       new_df['TENURE_WEEKLY'],
                                       new_df['MONETARY_AVG'],
                                       time=6,  # 6 MONTH
                                       freq="W",  # T's frequency information. (WEEKLY)
                                       discount_rate=0.01)  # consider discounts that can be made over time (discount rate)
    cltv = pd.DataFrame(cltv)
    new_df["CLTV"] = cltv

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(new_df[["CLTV"]])
    new_df["SCALED_CLTV"] = scaler.transform(new_df[["CLTV"]])

    new_df["SEGMENT"] = pd.qcut(new_df["SCALED_CLTV"], 4, labels=["D", "C", "B", "A"])

    final_df = pd.merge(df, new_df[["CUSTOMER_ID", "CLTV", "SCALED_CLTV", "SEGMENT"]], left_on="master_id",
                        right_on="CUSTOMER_ID", how="left")

    return df, new_df, final_df

df,new_df,final_df  = cltv_prediction(df)

new_df.groupby("SEGMENT").agg({"count","mean","sum"})