import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('weatherAUS_rainfall_prediction_dataset_cleaned.csv')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, r2_score, precision_score
from sklearn.model_selection import RandomizedSearchCV

models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'GradientBoost': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier()
}

df['Location'] = df['Location'].replace('MelbourneAirport', 'Melbourne')

df = df.drop(columns=['Date'])

y = df['RainTomorrow']
df = df.drop(columns = ['RainTomorrow'], axis = 1)
df.head()
x = df

numeric_cols = [cols for cols in df.columns if df[cols].dtype != 'O']
cat_cols = [cols for cols in df.columns if df[cols].dtype == 'O']

preprocessor = ColumnTransformer(transformers=[
    ('Scaler',StandardScaler(), numeric_cols),
    ('Encoder',OneHotEncoder(), cat_cols)
], remainder= 'passthrough'
)

y =LabelEncoder().fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state= 42)

x_train_transformed = preprocessor.fit_transform(x_train)
x_test_transformed = preprocessor.fit_transform(x_test)

rf_params = {
    'n_estimators': [100,200,300],
    'criterion': ['gini','entropy','log_loss'],
    'max_depth': [5, 8, 10, None],
    'min_samples_split': [2, 4, 5]
}
xgb_params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 300, 500],
    'colsample_bytree': [0.5, 0.8, 1],
    'subsample': [0.5, 0.8, 1],
    'gamma': [0, 1, 5],
    'tree_method': ['hist'],
    'predictor': ['auto']
}


randomcv = [
    ('XGBClassifier', XGBClassifier(), xgb_params)
]

model_params = {}
for name,model,params in randomcv:
    random = RandomizedSearchCV(estimator=model, param_distributions=params, n_jobs=-1,n_iter = 100, cv = 3, verbose = 2)
    random.fit(x_train_transformed, y_train)

    model_params[name] = random.best_params_

    print(f"\n\n.........{name} best params........")
    print(model_params[name])

