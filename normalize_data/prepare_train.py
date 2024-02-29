import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('train.csv')

# Clean Wilderness Area
wild_areas = [f'Wilderness_Area{i}' for i in range(1, 5)]
df['Wilderness_Area'] = df[wild_areas].idxmax(axis=1).str.extract('(\d+)')
df['Wilderness_Area'] = df['Wilderness_Area'].astype(int)

# Clean Soil Type
soils = [f'Soil_Type{i}' for i in range(1, 41)]
df['Soil_Type'] = df[soils].idxmax(axis=1).str.extract('(\d+)')
df['Soil_Type'] = df['Soil_Type'].astype(int)

labels = df['Cover_Type']
df.drop(soils + wild_areas + ['Cover_Type'], axis=1, inplace=True)

# Normalize columns
with (open("stat_dict.pkl", "rb")) as openfile:
    stat_dict = pickle.load(openfile)

for column in df.columns:
    mean = stat_dict[column][0]
    std = stat_dict[column][1]
    df[column] = (df[column] - mean) / std

df['Cover_Type'] = labels

df.to_csv('train-cleaned.csv', index=False)