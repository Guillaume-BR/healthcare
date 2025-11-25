import pandas as pd
import numpy as np
from utils.functions import normalize_char
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler , OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    # Normalisation des chaînes de caractères
    df = normalize_char(df)
    
    #suppression des doublons
    df = df.drop_duplicates()

    #On supprime les colonnes inutiles : noise_col et/ou random_notes si elles existent
    if 'noise_col' in df.columns:
        df.drop(columns=['noise_col'], inplace=True)
    if 'random_notes' in df.columns:
        df.drop(columns=['random_notes'], inplace=True)

    return df



def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df)
    
    #création de variables
    # BMI Categories (WHO Classification)
    df['bmi_category'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 25, 30, 35, 40, 100],
        labels=['Underweight', 'Normal', 'Overweight', 
               'Obese_I', 'Obese_II', 'Obese_III']
    )

    # Simplified BMI risk
    df['bmi_risk'] = df['bmi_category'].map({
        'Underweight': 1,
        'Normal': 0,
        'Overweight': 1,
        'Obese_I': 2,
        'Obese_II': 3,
        'Obese_III': 4
    })

    # Age Groups (Insurance Industry Standard) et mettre Nan si Nan dans Age
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    df['age_group'] = pd.cut(
        df['age'],
        bins=[17, 25, 35, 45, 55, 65],
        labels=['18-25', '26-35', '36-45', '46-55', '56-64'],
        include_lowest=True
    )

    # Age risk score
    df['age_risk'] = pd.cut(
        df['age'],
        bins=[17, 30, 40, 50, 60, 65],
        labels=[1, 2, 3, 4, 5]
    )

    return df


def preprocess_features(df: pd.DataFrame, target_col: str = 'lengthofstay') -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
    """
    Applique le feature engineering, gère les NaN et encode les colonnes catégorielles.
    Renvoie X_encoded, y et dictionnaire d'encoders.
    """
    df_feat = feature_engineering(df)  # la fonction existante
    df_feat.dropna(inplace=True)

    X = df_feat.drop([target_col], axis=1)
    y = df_feat[target_col]

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    X_encoded = X.copy()
    label_encoders = {}

    #Encodage de chaque variable catégorielle
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = X_encoded[col].astype(str).fillna('missing')
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le

    return X_encoded, y, label_encoders


def compute_feature_scores(X_encoded: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Calcule l'importance des features via Mutual Information et Random Forest,
    normalise les scores et renvoie un dataframe combiné à partir de variables encodées.
    """
    # Mutual Information
    mi_scores = mutual_info_regression(X_encoded, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': X_encoded.columns, 'MI Score': mi_scores})

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_encoded, y)
    rf_df = pd.DataFrame({'Feature': X_encoded.columns, 'RF Score': rf.feature_importances_})

    # Combine & normalize
    combined = mi_df.merge(rf_df, on='Feature')
    combined['MI_norm'] = combined['MI Score'] / combined['MI Score'].max()
    combined['RF_norm'] = combined['RF Score'] / combined['RF Score'].max()
    combined['Combined_Score'] = (combined['MI_norm'] + combined['RF_norm']) / 2

    return combined.sort_values('Combined_Score', ascending=False)

def select_features(scores_df: pd.DataFrame, threshold: float = 0.1, must_have: list = None) -> list:
    """
    Sélectionne les features selon un seuil et ajoute les features obligatoires.
    """
    selected = scores_df.loc[scores_df['Combined_Score'] > threshold, 'Feature'].tolist()

    if must_have:
        for f in must_have:
            if f not in selected:
                selected.append(f)
    
    return selected

def feature_selection(df: pd.DataFrame, target_col: str = 'lengthofstay', threshold: float = 0.1) -> list:
    """
    Pipeline complet de sélection de features.
    """
    must_have_features = ['age', 'bmi', 'smoking', 'alcohol', 'gender']

    X_encoded, y, _ = preprocess_features(df, target_col)
    scores_df = compute_feature_scores(X_encoded, y)
    selected_features = select_features(scores_df, threshold, must_have_features)

    return selected_features

def train_test_val(df: pd.DataFrame) -> dict:
    """
    Split the data into training, validation, and test sets.
    """
    selected_features = feature_selection(df, target_col='lengthofstay', threshold=0.1)
    df_feat = df[selected_features + ['lengthofstay']]


    # Split the data into features and target variable
    X = df_feat.drop(columns=['lengthofstay'])
    y = df_feat['lengthofstay']
    
    # First split: train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42    
    )
    
    # Second split: train en train + val
    X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42    
    )
    
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }

#imputation et encodage

def create_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Create and fit the preprocessing pipeline on training data only.
    """
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Fit only once
    preprocessor.fit(X_train)
    print("✅ Preprocessor fitted on training data")

    joblib.dump(preprocessor, '../models/preprocessor.pkl')
    print("✅ Preprocessor sauvegardé sous '../models/preprocessor.pkl'")
    return preprocessor


def preprocess_splits(splits: dict, preprocessor: ColumnTransformer) -> dict:
    """
    Transform train/val/test using a fitted preprocessor.
    """
    X_train_p = preprocessor.transform(splits["X_train"])
    X_val_p   = preprocessor.transform(splits["X_val"])
    X_test_p  = preprocessor.transform(splits["X_test"])
    
    return {
        "X_train": X_train_p,
        "X_val": X_val_p,
        "X_test": X_test_p,
        "y_train": splits["y_train"],
        "y_val": splits["y_val"],
        "y_test": splits["y_test"]
    }




