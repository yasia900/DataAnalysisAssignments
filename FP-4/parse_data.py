import pandas as pd

def load_raw_csv(path: str) -> pd.DataFrame:
    """Load raw CSV file."""
    return pd.read_csv(path)


def select_relevant_columns(df0: pd.DataFrame) -> pd.DataFrame:
    """Select columns required for FP-3."""
    return df0[
        [
            'Value',
            'Value for 100k Of Affected Population',
            'Year',
            'Air Pollution Average [ug/m3]',
            'Air Pollution Population Weighted Average [ug/m3]',
            'Air Pollutant',
            'Population',
            'Affected Population',
            'Populated Area [km2]',
            'Degree Of Urbanisation',
            'Sex',
            'Description Of Age Group',
            'Category',
            'Outcome',
            'Health Indicator'
        ]
    ]


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename long column names to short snake_case names."""
    return df.rename(columns={
        'Value': 'value',
        'Value for 100k Of Affected Population': 'value_per_100k',
        'Year': 'year',
        'Air Pollution Average [ug/m3]': 'pollution_avg',
        'Air Pollution Population Weighted Average [ug/m3]': 'pollution_pop_avg',
        'Air Pollutant': 'pollutant',
        'Population': 'population',
        'Affected Population': 'affected_pop',
        'Populated Area [km2]': 'area_km2',
        'Degree Of Urbanisation': 'urban_degree',
        'Sex': 'sex',
        'Description Of Age Group': 'age_group',
        'Category': 'category',
        'Outcome': 'outcome',
        'Health Indicator': 'health_indicator'
    })


def get_clean_data(path: str) -> pd.DataFrame:
    """Load CSV, extract needed columns, rename them, return clean df."""
    df0 = load_raw_csv(path)
    df = select_relevant_columns(df0)
    df = rename_columns(df)
    return df
