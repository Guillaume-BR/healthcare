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

