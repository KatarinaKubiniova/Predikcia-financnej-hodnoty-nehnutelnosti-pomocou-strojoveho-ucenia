import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib
from sklearn.neighbors import KNeighborsRegressor
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet


# ************** PREDSPRACOVANIE **************************************************************
data = pd.read_excel('data.xlsx')
data = pd.get_dummies(data)

imputer_median = SimpleImputer(strategy='median')
stlpce_median = ['population', 'izby', 'vybavenie']
data[stlpce_median] = imputer_median.fit_transform(data[stlpce_median])

imputer_zero = SimpleImputer(strategy='constant', fill_value=0)
vsetky_stlpce = set(data.columns)
stlpce_zero = list(vsetky_stlpce - set(stlpce_median))
data[stlpce_zero] = imputer_zero.fit_transform(data[stlpce_zero])

X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)

# ************** škálovanie pre niektoré modely ******************************************************
'''
normalizer = Normalizer()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)


scaler = StandardScaler()
scaler_minmax = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

'''
# ********************************** LASSO regresia **************************************************
'''
param_grid = {
    "alpha": [500],
    "max_iter":  [3000],
}

lasso_regressor = Lasso(random_state=42)

grid_search = GridSearchCV(estimator=lasso_regressor,
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Najlepšie hyparametre:", grid_search.best_params_)
'''
# ********************************** RANDOM FOREST *************************************************************

'''
param_grid = {
    "n_estimators": [270],  
    "max_depth": [25],        
    "min_samples_split": [2], 
    "min_samples_leaf": [1],   
}

random_forest = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=random_forest,
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1, 
                           verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Najlepšie parametre:", grid_search.best_params_)
'''
# ********************************** DECISION TREES *************************************************************
'''

param_grid = {
    "max_depth": [16],
    "min_samples_split": [15], 
    "min_samples_leaf": [3], 
}


decision_tree = DecisionTreeRegressor()

grid_search = GridSearchCV(estimator=decision_tree,
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1, 
                           verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Najlepšie parmetre:", grid_search.best_params_)

'''
# ***********************************RGBBOOST*********************************************************************

param_grid = {
    "n_estimators": [210],
    "learning_rate": [0.1],
    "max_depth": [10],
    "gamma": [0.009],
    "subsample": [1.0],
    "colsample_bytree": [0.6],
}


grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', alpha=1),
                               param_grid=param_grid,
                               cv=5,
                               n_jobs=-1,
                               verbose=2)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
print("Najlepsie parmetre:", grid_search.best_params_)


feature_importance = best_model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Dôležitosť parametrov:")
print(feature_importance_df)

# ***************************************KNN regresia****************************************************************
'''

param_grid = {
    'n_neighbors': [8],
    'weights': ['distance'],
    'leaf_size': [1],
    'p': [1],
}

grid_search = GridSearchCV(estimator=KNeighborsRegressor(),
                               param_grid=param_grid,
                               cv=5,
                               n_jobs=-1,
                               verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("\nNajlepsie parmetre::")
for param_name in best_params:
    print(f"{param_name}: {best_params[param_name]}")
    
'''
# ***************************************Ridge regression****************************************************************
'''

param_grid = {
    'alpha': [300],
}


grid_search = GridSearchCV(estimator=Ridge(),
                                 param_grid=param_grid,
                                 cv=5,
                                 n_jobs=-1,
                                 verbose=2)


grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
best_params = grid_search.best_params_


print("\nNajlepsie parmetre::")
for param_name in best_params:
    print(f"{param_name}: {best_params[param_name]}")
'''
# *************************************** ElasticNet regression*********************************************************

'''
param_grid = {
    'alpha': [0.1],
    'l1_ratio': [0.6],
}


grid_search = GridSearchCV(estimator=ElasticNet(),
                                   param_grid=param_grid,
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=2)


grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("\nNajlepsie parmetre::")
for param_name in best_params:
    print(f"{param_name}: {best_params[param_name]}")
'''
# *******************************************************************************************************

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

print(f"R-squared (R²): {r2_score(y_test, y_pred_test)}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred_test))}")

plt.scatter(y_test, y_pred_test, c='#5D3FD3', alpha=0.7, s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '-', color='red')
plt.grid(True)
plt.show()
