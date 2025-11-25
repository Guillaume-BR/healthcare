import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

# Modèles de bases
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,RidgeCV, LassoCV, ElasticNetCV)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
    )
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Modèles avancés
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> RegressorMixin:
    """
    Entraîne un modèle de régression linéaire et renvoie le modèle entraîné.
    """
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

def train_ridge(X_train: np.ndarray, y_train:np.ndarray) -> RegressorMixin:
    """
    Entraîne un RidgeCV et renvoie le modèle Ridge final entraîné.
    """
    ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)
    ridge_cv.fit(X_train, y_train)
    print(f"Optimal Ridge alpha: {ridge_cv.alpha_}")
    
    # Crée le modèle final avec l'alpha optimal
    ridge_model = Ridge(alpha=ridge_cv.alpha_)
    ridge_model.fit(X_train, y_train)
    
    return ridge_model

def train_lasso(X_train: np.ndarray, y_train: np.ndarray) -> RegressorMixin:
    """
    Entraîne un LassoCV et renvoie le modèle Lasso final entraîné.
    """
    lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5, max_iter=10000)
    lasso_cv.fit(X_train, y_train)
    print(f"Optimal Lasso alpha: {lasso_cv.alpha_}")
    
    # Crée le modèle final avec l'alpha optimal
    lasso_model = Lasso(alpha=lasso_cv.alpha_)
    lasso_model.fit(X_train, y_train)
    
    return lasso_model

def train_elasticnet(X_train: np.ndarray, y_train: np.ndarray) -> RegressorMixin:
    """
    Entraîne un ElasticNetCV et renvoie le modèle ElasticNet final entraîné.
    """
    elasticnet_cv = ElasticNetCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=10000)
    elasticnet_cv.fit(X_train, y_train)
    print(f"Optimal ElasticNet alpha: {elasticnet_cv.alpha_}, l1_ratio: {elasticnet_cv.l1_ratio_}")
    
    # Crée le modèle final avec l'alpha et l1_ratio optimaux
    elasticnet_model = ElasticNet(alpha=elasticnet_cv.alpha_, l1_ratio=elasticnet_cv.l1_ratio_)
    elasticnet_model.fit(X_train, y_train)
    
    return elasticnet_model

def train_knn(X_train: np.ndarray, y_train: np.ndarray) -> RegressorMixin:
    """
    Entraîne un KNeighbors Regressor et renvoie le modèle entraîné.
    """
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    return knn_model

def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray) -> RegressorMixin:
    """
    Entraîne un Decision Tree Regressor et renvoie le modèle entraîné.
    """
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    return dt_model

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RegressorMixin:
    """
    Entraîne un Random Forest Regressor et renvoie le modèle entraîné.
    """
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    return rf_model

def gradient_boosting(X_train: np.ndarray, y_train: np.ndarray) -> RegressorMixin:
    """
    Entraîne un Gradient Boosting Regressor et renvoie le modèle entraîné.
    """
    gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    gb_model.fit(X_train, y_train)
    return gb_model

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> RegressorMixin:
    """
    Entraîne un modèle XGBoost Regressor et renvoie le modèle entraîné.
    """
    xgb_params = {
    'n_estimators': [300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

    xgb_random_search = RandomizedSearchCV(
    xgb_model,
    xgb_params,
    n_iter=20,
    cv=3,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=1
    )

    xgb_random_search.fit(X_train, y_train)
    print(f"Best XGBoost parameters: {xgb_random_search.best_params_}")
    return xgb_model

def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray) -> RegressorMixin:
    """
    Entraîne un modèle LightGBM Regressor et renvoie le modèle entraîné.
    """
    lgb_params = {
    'n_estimators': [300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100],
    'subsample': [0.8, 1.0]
    }
    
    lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)

    lgb_random_search = RandomizedSearchCV(
        lgb_model,
        lgb_params,
        n_iter=20,
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    lgb_random_search.fit(X_train, y_train)
    print(f"Best LightGBM parameters: {lgb_random_search.best_params_}")
    return lgb_model

def train_catboost(X_train: np.ndarray, y_train: np.ndarray) -> RegressorMixin:
    """
    Entraîne un modèle CatBoost Regressor et renvoie le modèle entraîné.
    """
    cat_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric='R2',
        random_seed=42,
        logging_level='Silent'
    )
    
    cat_model.fit(X_train, y_train)
    return cat_model

def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Entraîne tous les modèles et retourne un dictionnaire {nom: modèle entraîné}.
    """
    models = {}
    
    # Linear Regression
    models['LinearRegression'] = train_linear_regression(X_train, y_train)
    # Ridge
    models['Ridge'] = train_ridge(X_train, y_train)
    # Lasso
    models['Lasso'] = train_lasso(X_train, y_train)
    # ElasticNet
    models['ElasticNet'] = train_elasticnet(X_train, y_train)
    # K-Nearest Neighbors
    models['KNN'] = train_knn(X_train, y_train)
    # Decision Tree
    models['DecisionTree'] = train_decision_tree(X_train, y_train)
    # Random Forest
    models['RandomForest'] = train_random_forest(X_train, y_train)
    # Gradient Boosting
    models['GradientBoosting'] = gradient_boosting(X_train, y_train)
    # XGBoost
    models['XGBoost'] = train_xgboost(X_train, y_train)
    # LightGBM
    models['LightGBM'] = train_lightgbm(X_train, y_train)
    # CatBoost
    models['CatBoost'] = train_catboost(X_train, y_train)    
    
    return models



