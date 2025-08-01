import warnings
warnings.filterwarnings('ignore')

#Exploratory Data Analysis Libraries
import pandas as pd

#Predictive Models
import xgboost as xgb
from sklearn.model_selection import train_test_split

def separate_train(df):

  #Separate train and test data
  df_dummies = pd.get_dummies(df, drop_first=True)

  train_start = '2015-01-01'
  train_end = '2017-08-15'

  train_data = df_dummies[(df_dummies['date'] >= train_start) & (df_dummies['date'] <= train_end)]

  feature_columns = [col for col in df_dummies.columns if col not in ['sales', 'date']]

  return train_data

def separate_test(df):

  #Separate train and test data
  df_dummies = pd.get_dummies(df, drop_first=True)

  test_start = '2017-08-16'
  test_end = '2017-08-31'

  test_data = df_dummies[(df_dummies['date'] >= test_start) & (df_dummies['date'] <= test_end)]

  return test_data

def prepare_data(df):

  #Import data
  train = separate_train(df)
  test = separate_test(df)

  #Prepare Features and Target
  # Target variable
  y = train['sales']

  # Features to use (add/remove as needed)
  # Drop columns not used as features
  drop_cols = ['id', 'date', 'sales', 'family']  # keep family_encoded

  # Select all other columns as features
  features = [col for col in train.columns if col not in drop_cols]

  # Prepare training and test data
  X = train[features]
  X_test = test[features]

  return X, y, X_test

def train_model(df):

  #Import data
  X, y, X_test = prepare_data(df)

  #Train/Test Split
  X_train, X_val, y_train, y_val = train_test_split(

      X, y, test_size=0.2, shuffle=False

      )  # No shuffling due to time-series nature

    #Train XGBoost Model
  model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
  )

  model.fit(X_train, y_train)

  return model, X_val, y_val, X_test