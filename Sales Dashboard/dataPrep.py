import pandas as pd
import numpy as np

# Read data
customer_info = pd.read_csv(
    "Data/Raw/customer_info.csv")
customer_cards = pd.read_csv(
    "Data/Raw/customer_cards.csv")
transaction_details = pd.read_csv(
    "Data/Raw/transaction_details.csv")
transaction_meta = pd.read_csv(
    "Data/Raw/transaction_meta_info.csv")

# Change false city information to nan
customer_info.loc[(customer_info.citycode > 81) | (
    customer_info.citycode < 1), "citycode"] = np.nan

# Add city names and drop the city codes
city_names = pd.read_csv("Data/Raw/cities.csv", usecols=["name", "citycode"])
customer_info = pd.merge(customer_info, city_names, on="citycode")
customer_info = customer_info.rename(columns={'name': 'cityname'})
customer_info = customer_info.drop(['citycode'], axis=1)

# Add age column
customer_info.loc[:, 'age'] = 2022 - customer_info.dateofbirth

# Remove outliers in age
# customer_info.boxplot("age")
customer_info.loc[(customer_info.age < 18) | (
    customer_info.age > 81), "age"] = np.nan

# Drop dateofbirth column
customer_info = customer_info.drop(['dateofbirth'], axis=1)

# Save
customer_info.to_csv("Data/Generated/customer_info.csv", index=False)

# Remove duplicates from transaction details
transaction_details = transaction_details.drop_duplicates()

# Remove zero or negative cost/quantity rows
transaction_details = transaction_details[(
    transaction_details.cost > 0) & transaction_details.quantity > 0]

# Change the date type in transaction meta info
transaction_meta['date'] = pd.to_datetime(transaction_meta['date'])

# Merge transaction details with transaction meta
transactions_merged = pd.merge(
    transaction_details, transaction_meta, on="basketid", how="inner")

# Merge customer card account with transaction details
transactions_merged = pd.merge(
    customer_cards, transactions_merged, on="cardnumber")

# Summarize transactions by individual baskets
transactions_merged.loc[:, 'n_products'] = 0
transactions_groupedby_basket = transactions_merged.groupby('basketid', as_index=False).agg({
    'individualnumber': 'max',
    'basketid': 'max',
    'cardnumber': 'max',
    'date': 'max',
    'n_products': 'count',
    'cost': 'sum',
    'quantity': 'sum',
    'discounttype1': 'sum',
    'discounttype2': 'sum',
    'discounttype3': 'sum',
    'issanal': 'max'
})

# Add city information to transactions
transactions_groupedby_basket = pd.merge(
    customer_info, transactions_groupedby_basket, on="individualnumber")

# Save
transactions_groupedby_basket.to_csv(
    "Data/Generated/transactions.csv", index=False)
