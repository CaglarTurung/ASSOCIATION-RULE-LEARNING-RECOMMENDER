
# ASSOCIATION RULE LEARNING RECOMMENDATION

############################################
# DATA PREPARATION
############################################
import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

############################################
# DATA PREPROCESSING
############################################

df_ = pd.read_excel(r"E:\CAGLAR\datasets\online_retail_II.xlsx", sheet_name="Year 2010-2011")
df_.head()
df = df_.copy()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds (dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe
df = retail_data_prep(df)
df.head()

# Removed POST
df = df.loc[(df["StockCode"] != "POST")]

# ASSOCIATION RULES HAS BEEN CREATED (GERMANY)
def create_invoice_product_df(dataframe, id = False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def create_rules(dataframe, id = True, country = "France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id = True)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

rules_grm = create_rules(df, country="Germany")
rules_grm.head()

# WHAT ARE THE NAMES OF THE PRODUCTS WHOSE IDs ARE GIVEN ?
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df, 21987)
# (PACK OF 6 SKULL PAPER CUPS)
check_id(df, 23235)
# (STORAGE TIN VINTAGE LEAF)
check_id(df, 22747)
# (POPPY'S PLAYHOUSE BATHROOM)

# PRODUCT RECOMMENDATION

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.loc[i]["consequents"]))

    recommendation_list = list(dict.fromkeys(item for item_list in recommendation_list for item in item_list))

    return recommendation_list[:rec_count]

arl_recommender(rules_grm, 23235, 1)
arl_recommender(rules_grm, 22747, 1)
arl_recommender(rules_grm, 16237, 3)

# WHAT ARE THE NAMES OF THE RECOMMENDED PRODUCTS ?

check_id(df, arl_recommender(rules_grm, 21987, 2)[0])
check_id(df, arl_recommender(rules_grm, 23235, 1)[0])
check_id(df, arl_recommender(rules_grm, 22747, 1)[0])
