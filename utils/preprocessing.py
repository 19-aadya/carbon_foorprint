import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        '2017 NAICS Title': 'Sector',
        'Supply Chain Emission Factors with Margins': 'EmissionFactor'
    })

    df = df[['Sector', 'EmissionFactor']]
    df.dropna(inplace=True)

    df['Sector_encoded'] = df['Sector'].astype('category').cat.codes
    X = df[['Sector_encoded']].values
    y = df['EmissionFactor'].values

    return X, y, df