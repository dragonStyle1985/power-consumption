import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Create a sample DataFrame
    data = {
        'Date': pd.date_range(start='2022-01-01', periods=10),
        'MT_001': np.random.randint(1, 10, size=10),
        'MT_002': np.random.randint(1, 10, size=10),
        'MT_003': np.random.randint(1, 10, size=10),
        'MT_004': np.random.randint(1, 10, size=10),
        'MT_005': np.random.randint(1, 10, size=10),
        'MT_006': np.random.randint(1, 10, size=10)
    }

    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)

    print(df)

    shift_unit = 3

    data_rolled = df.iloc[:, 3:6].rolling(window=shift_unit, min_periods=1).sum().shift(-shift_unit)
    df['target'] = data_rolled.sum(axis=1)
    df = df.iloc[:-shift_unit]
    print(df)
