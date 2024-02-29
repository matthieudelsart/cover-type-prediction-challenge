import numpy as np
import pandas as pd


def concat_Wilderness_Area(df: pd.DataFrame, new_name="WA_concat", drop_value=False):
    df = df.copy()
    wilderness_areas = [f"Wilderness_Area{i}" for i in range(1, 5)]
    df[new_name] = df[wilderness_areas] @ range(1, 5)
    if drop_value:
        df = df.drop(columns=wilderness_areas)
    return df


def concat_Soil_Type(df: pd.DataFrame, new_name="ST_concat", drop_value=False):
    df = df.copy()
    soil_types = [f"Soil_Type{i}" for i in range(1, 41)]
    df[new_name] = df[soil_types] @ range(1, 41)
    if drop_value:
        df = df.drop(columns=soil_types)
    return df


def group_climatic_zone(df: pd.DataFrame, one_hot=False):
    df = df.copy()
    col_names = 'ST_concat'
    if col_names not in df.columns:
        df = concat_Soil_Type(df)
    new_names = 'climatic_zone'
    zone2 = (df[col_names] >= 1) & (df[col_names] <= 6)
    zone3 = (df[col_names] >= 7) & (df[col_names] <= 8)
    zone4 = (df[col_names] >= 9) & (df[col_names] <= 13)
    zone5 = (df[col_names] >= 14) & (df[col_names] <= 15)
    zone6 = (df[col_names] >= 16) & (df[col_names] <= 18)
    zone7 = (df[col_names] >= 19) & (df[col_names] <= 34)
    zone8 = (df[col_names] >= 35) & (df[col_names] <= 40)
    cond = [zone2, zone3, zone4, zone5, zone6, zone7, zone8]
    if one_hot:
        for i in range(7):
            new_col_names = f'{new_names}_{i+2}'
            df[new_col_names] = np.select([cond[i]], [1], default=0)
    else:
        val = [i for i in range(2, 9)]
        df[new_names] = np.select(cond, val, default=0)
    return df


def group_geological_zone(df: pd.DataFrame, one_hot=False):
    df = df.copy()
    col_names = 'ST_concat'
    if col_names not in df.columns:
        df = concat_Soil_Type(df)

    new_names = 'geological_zone'
    zone1 = df[col_names].isin([14, 15, 16, 17, 19, 20, 21])
    zone2 = df[col_names].isin([9, 22, 23])
    zone5 = df[col_names].isin([7, 8])
    g7 = [i for i in range(1, 7)] + [i for i in range(10, 14)
                                     ] + [18] + [i for i in range(24, 40)]
    zone7 = df[col_names].isin(g7)
    cond = [zone1, zone2, zone5, zone7]
    if one_hot:
        df[f'{new_names}_{1}'] = np.select([cond[0]], [1], default=0)
        df[f'{new_names}_{2}'] = np.select([cond[1]], [1], default=0)
        df[f'{new_names}_{5}'] = np.select([cond[2]], [1], default=0)
        df[f'{new_names}_{7}'] = np.select([cond[3]], [1], default=0)
    else:
        val = [1, 2, 5, 7]
        df[new_names] = np.select(cond, val, default=0)
    return df


def preprocess(df):
    df = concat_Soil_Type(df)
    df = concat_Wilderness_Area(df)
    df = group_climatic_zone(df)
    df = group_geological_zone(df)
    return df
