def describe_dataframe(df, name="DataFrame"):
    """
    Comprehensive DataFrame description with memory usage and data types
    """
    print_divider(f"{name} Overview")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing Values: {df.isnull().sum().sum()}")
    print(f"Duplicate Rows: {df.duplicated().sum()}")
    
    return df.info()

def normalize_char(df):
    """
    Normalize string columns by converting to lowercase, stripping whitespace,
    and replacing spaces with underscores.
    """
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace(r"[^a-zA-Z0-9_]", "")

    str_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in str_cols:
        df[col] = df[col].str.lower().str.strip().str.replace(" ", "_")

    return df
