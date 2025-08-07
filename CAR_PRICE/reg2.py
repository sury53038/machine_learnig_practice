import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df =pd.read_csv('cardekho_imputated.csv')

df.head()

df.drop(columns=['car_name', 'brand', 'Unnamed: 0'], axis=1, inplace = True)

from sklearn.model_selection import train_test_split
x = df.drop(columns=['selling_price'],axis = 1)
y = df['selling_price']

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
odnl = OrdinalEncoder()

odn_col = ['model']

ohe_cols = ['seller_type', 'fuel_type', 'transmission_type']
scale_cols = [col for col in x.columns if col not in ohe_cols and col not in odn_col]

ohe = OneHotEncoder(sparse_output = False,drop = 'first')

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

scaler = StandardScaler()


preprocessor = ColumnTransformer(
    transformers=[
        ('OrdinalEncoder', odnl, odn_col),
        ('OneHotEncoder', ohe, ohe_cols),
        ('StandardScaler', scaler, scale_cols)
    ],
    remainder='passthrough'
)


x = preprocessor.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.28, random_state=42)


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_models(true, predicted):
    mae = mean_absolute_error(true,predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true,predicted)
    return mae, mse, r2_square

models = {
    "LinearRegressor" : LinearRegression(),
    "DecisionTreeRegressor" : DecisionTreeRegressor(),
    "RandomForestRegressor" : RandomForestRegressor(n_estimators=200, min_samples_split=2, max_depth= None, criterion= 'absolute_error'),
    "KNNRegressor" : KNeighborsRegressor(n_neighbors = 1),
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    model_mae, model_mse, model_r2_score = evaluate_models(y_pred, y_test)

    print(list(models.keys())[i])

    print("Model performance")
    print("Mean Squared Error : ",model_mae)
    print("Mean Absolute Error : ",model_mse)
    print("R2 Score : ", model_r2_score)

    print("\n.......................................")
    
    
    
from sklearn.pipeline import Pipeline
pipe = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor()) #(n_estimators=200, min_samples_split=2, max_depth= None, criterion= 'absolute_error'))
])
