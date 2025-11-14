import os
import joblib
from sklearn.metrics import root_mean_squared_error, r2_score

def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Calcule les métriques sur train et validation pour un modèle donné.
    Retourne un dictionnaire de métriques.
    """
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    metrics = {
        'train_rmse': root_mean_squared_error(y_train, y_train_pred),
        'val_rmse': root_mean_squared_error(y_val, y_val_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'val_r2': r2_score(y_val, y_val_pred)
    }
    
    return metrics

def choose_best_model(models, X_train, y_train, X_val, y_val):
    """
    Évalue tous les modèles sur le jeu de validation et renvoie le meilleur modèle
    selon val_rmse, ainsi que son nom et ses métriques.
    Retourne le modèle, son nom et un dictionnaire de résultats.
    """
    results = {}
    
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, y_train, X_val, y_val)
    
    best_name = min(results.items(), key=lambda x: x[1]['val_mse'])[0]
    best_model = models[best_name]
    
    return best_model, best_name, results

def evaluate_on_test(model, X_train, y_train, X_test, y_test):
    """
    Évalue le modèle final sur le test set.
    """
    return evaluate_model(model, X_train, y_train, X_test, y_test)

def save_model(model, model_name, folder='models'):
    os.makedirs(folder, exist_ok=True)
    joblib.dump(model, f'{folder}/{model_name}.joblib')
    print(f"Model '{model_name}' saved in '{folder}/'")