from src.data_loader import load_data
from scr.preprocessing import *


def main():
    data_path = "../data/raw/dirty_v3_path.csv"
    RANDOM_STATE = 42
    df = load_data(data_path)
    splits = train_test_val(df)
    preprocessor = create_preprocessor(splits["X_train"])
    processed_data = preprocess_splits(splits, preprocessor)


    df_feat = df[features_selection]

if __name__=="__main__":
    main()