import os
import pandas as pd

def load_data(data_path: str = "../data/raw/dirty_v3_path.csv") -> pd.DataFrame:
    """
    Load the dataset from the specified CSV file path.

    Parameters:
    - data_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The loaded dataset.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Le fichier est introuvable à l'emplacement {data_path}")
    
    try:
        df_raw = pd.read_csv(data_path)
        df = df_raw.copy()
        return df
    except Exception as e:
        raise RuntimeError(f"❌ Une erreur s'est produite lors du chargement des données: {e}")