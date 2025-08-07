import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df =pd.read_csv('cardekho_imputated.csv')

df.drop(columns=['car_name', 'brand', 'Unnamed: 0'], axis=1, inplace = True)

from sklearn.model_selection import train_test_split
x = df.drop(columns=['selling_price'],axis = 1)
y = df['selling_price']

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
label = LabelEncoder()

x['model'] = label.fit_transform(x['model'])


ohe_cols = ['seller_type', 'fuel_type', 'transmission_type']
scale_cols = [cols for cols in x.columns if cols not in ohe_cols]

ohe = OneHotEncoder(sparse_output = False,drop = 'first')


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    [
        ('OneHotEncoder', ohe, ohe_cols),
        ('StandardScaler', scaler, scale_cols)
    ],remainder='passthrough'
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
    "RandomForestRegressor" : RandomForestRegressor(),
    "KNNRegressor" : KNeighborsRegressor(),
    "LogisticRegressor" : LogisticRegression()
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



random_forest_param = {
    "n_estimators" : [100, 200, 500],
    "criterion" : ["squared_error", "absolute_error"],
    "max_depth" : [5, 15, None, 10],
    "min_samples_split" : [2, 3, 5]
}
knn_params = {"n_neighbors" : [5,10,15, 3, 2]}

from sklearn.model_selection import RandomizedSearchCV
randomcv_models = [('knn', KNeighborsRegressor(), knn_params),
                   ('rfr', RandomForestRegressor(), random_forest_param)
                  ]


model_params = {}

for name, model, param in randomcv_models:
    random = RandomizedSearchCV(estimator=model, param_distributions=param,
                                    n_iter=100,
                                    cv=3,
                                    verbose=2,
                                    n_jobs=-1)
    random.fit(x_train, y_train)
    model_params[name] = random.best_params_

for model_name in model_params:
    print("...................................")
    print(f"Best Parameters for {model_name}")
    print(model_params[model_name])

    
