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

def normalize_char(df)
    # Normalise les noms de colonnes en minuscules et remplace les espaces et tirets par des underscores
    df.columns = df.columns.lower().str .replace(' ', '_').str.replace('-', '_')

    #On effectue le même traitement pour les valeurs de type string
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    for cols in str_cols:
        df[cols] = df[cols].str.lower().str.replace(' ','_').str.replace('-','_')

    return df