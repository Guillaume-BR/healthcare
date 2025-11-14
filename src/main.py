from src.data_loader import load_data
from src.preprocessing import create_preprocessor, preprocess_splits,train_test_val
from src.modeling import train_all_models
from src.evaluation import choose_best_model, evaluate_on_test, save_model

def main():
    data_path = "../data/raw/dirty_v3_path.csv"
    RANDOM_STATE = 42
    
    # 1️⃣ Chargement des données
    df = load_data(data_path)
    
    # 2️⃣ Split train / val / test -> dictionnaire de DataFrames
    splits = train_test_val(df)
    
    # 3️⃣ Prétraitement
    preprocessor = create_preprocessor(splits["X_train"])
    processed_splits = preprocess_splits(splits, preprocessor)
    
    # 4️⃣ Entraînement de tous les modèles depuis src/modeling.py
    models = train_all_models(
        processed_splits['X_train'],
        processed_splits['y_train']
    )
    
    # 5️⃣ Sélection du meilleur modèle sur la validation
    best_model, best_name, val_results = choose_best_model(
        models,
        processed_splits['X_train'], processed_splits['y_train'],
        processed_splits['X_val'], processed_splits['y_val']
    )
    
    # 6️⃣ Évaluation finale sur le test set
    test_metrics = evaluate_on_test(
        best_model,
        processed_splits['X_train'], processed_splits['y_train'],
        processed_splits['X_test'], processed_splits['y_test']
    )
    
    # 7️⃣ Sauvegarde du meilleur modèle
    save_model(best_model, best_name)
    
    # 8️⃣ Affichage des résultats
    print(f"\nBest model: {best_name}")
    print(f"Validation metrics: {val_results[best_name]}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
