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

def clean_data(df):
    
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

def feature_engineering(df):
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



def feature_selection(df):
    """
    Sélection des features basées sur l'importance des variables
    en utilisant Mutual Information et Feature Importances d'un Random Forest.
    On ajoute aussi un encodage des variables catégorielles.
    """

    df_feat = feature_engineering(df)

    # Prepare data for feature selection en supprimant les na notamment
    df_feat.dropna(inplace=True)
    X = df_feat.drop(['lengthofstay'], axis=1)
    y = df_feat['lengthofstay']

    # IMPORTANT: Re-check for ALL categorical columns (including newly created ones)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    
    X_encoded = X.copy()
    label_encoders = {}

    # Encodage de chaque variable catégorielle
    for col in categorical_cols:
       le = LabelEncoder()
       # Handle different column types
       if X_encoded[col].dtype.name == 'category':
            # For categorical columns, convert to string first
            X_encoded[col] = X_encoded[col].astype(str)

        # Replace NaN with 'missing' string (now safe for all column types)
       X_encoded[col] = X_encoded[col].fillna('missing')

       # Encode the column
       X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
       label_encoders[col] = le

    
    #Choix des variables finales à l'aide de plusieurs critères
    
    # Mutual Information
    mi_scores = mutual_info_regression(X_encoded, y, random_state=42)
    mi_scores_df = pd.DataFrame({
        'Feature': X_encoded.columns,
        'MI Score': mi_scores
    }).sort_values('MI Score', ascending=False)

    # Feature Importances from Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_encoded, y)

    # Get feature importances
    rf_importance_df = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    importance_combined = pd.merge(
        mi_scores_df, 
        rf_importance_df, 
        on='Feature'
        )

    # Normalize scores to 0-1 range
    importance_combined['MI_normalized'] = (importance_combined['MI Score'] / 
                                            importance_combined['MI Score'].max())
    importance_combined['RF_normalized'] = (importance_combined['Importance'] / 
                                            importance_combined['Importance'].max())

    # Combined score (average of both)
    importance_combined['Combined_Score'] = (importance_combined['MI_normalized'] + 
                                             importance_combined['RF_normalized']) / 2

    importance_combined = importance_combined.sort_values('Combined_Score', ascending=False)

    # Sélection des features basées sur un seuil de score combiné
    threshold = 0.1  # Select features with combined score > 0.1
    selected_features = importance_combined[importance_combined['Combined_Score'] > threshold]['Feature'].tolist()

    # Always include original important features
    must_have_features = ['age', 'bmi', 'smoking', 'alcohol', 'gender']
    for feature in must_have_features:
        if feature not in selected_features:
            selected_features.append(feature)
    
    return selected_features

def train_test_val(df):
    """
    Split the data into training, validation, and test sets.
    """
    selected_features = feature_selection(df)
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

def create_preprocessor(X_train):
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

    return preprocessor


def preprocess_splits(splits, preprocessor):
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




