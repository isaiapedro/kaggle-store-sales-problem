import warnings
warnings.filterwarnings('ignore')

#Exploratory Data Analysis Libraries
import pandas as pd

def merge_dataframe(train, test, oil, holidays, transactions, stores):

  if 'SalePrice' not in test.columns:
    test['SalePrice'] = 0

  df = pd.concat([train, test], axis = 0)
  df = df.set_index('id')

  # --- Merge Oil Prices ---
  oil['dcoilwtico'] = oil['dcoilwtico'].ffill()
  df = df.merge(oil, on='date', how='left')

  # --- Merge Holidays ---
  holidays['is_holiday'] = 1
  holidays = holidays[['date', 'is_holiday']].drop_duplicates(subset='date')
  df = df.merge(holidays, on='date', how='left')
  df['is_holiday'] = df['is_holiday'].fillna(0)

  # --- Merge Transactions ---
  df = df.merge(transactions, on=['date', 'store_nbr'], how='left')
  df['transactions'] = df['transactions'].fillna(0)

  # --- Merge Store Metadata ---
  df = df.merge(stores, on='store_nbr', how='left')

  print("Dataset shape after merges:", df.shape)

  return df

def future_engineering(df):

  #Convert categorical column (family) to numeric codes
  df['family_encoded'] = df['family'].astype('category').cat.codes

  # Encode object columns to numeric
  for col in ['city', 'state', 'type']:
    if col in df.columns:
      df[col] = df[col].astype('category').cat.codes

  #Add date-based feature
  df['day_of_week'] = df['date'].dt.dayofweek
  df['month'] = df['date'].dt.month
  df['year'] = df['date'].dt.year
  df['day_of_month'] = df['date'].dt.day
  df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

  #Weekend Flag
  df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

  #Store type encoding
  df['store_type_encoded'] = df['type'].astype('category').cat.codes

  #Holiday Interaction
  # Higher sales expected just before holidays
  df['holiday_lag'] = df.groupby('store_nbr')['is_holiday'].shift(-1).fillna(0)

  return df