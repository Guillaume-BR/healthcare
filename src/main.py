from src.data_loader import load_data

def main():
    data_path = "../data/raw/dirty_v3_path.csv"
    df = load_data(data_path)
    print(df.head())

if __name__=="__main__":
    main()