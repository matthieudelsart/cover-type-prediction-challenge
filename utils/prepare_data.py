import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('test-full.csv')

# Clean Wilderness Area
wild_areas = [f'Wilderness_Area{i}' for i in range(1, 5)]
df['Wilderness_Area'] = df[wild_areas].idxmax(axis=1).str.extract('(\d+)')
df['Wilderness_Area'] = df['Wilderness_Area'].astype(int)

# Clean Soil Type
soils = [f'Soil_Type{i}' for i in range(1, 41)]
df['Soil_Type'] = df[soils].idxmax(axis=1).str.extract('(\d+)')
df['Soil_Type'] = df['Soil_Type'].astype(int)

df.drop(soils + wild_areas, axis=1, inplace=True)

# Normalize columns
stat_dict = {}
for column in df.columns:
    mean = df[column].mean()
    std = df[column].std()
    df[column] = (df[column] - mean) / std
    stat_dict[column] = (mean, std)

with open('stat_dict.pkl', 'wb') as file:
    pickle.dump(stat_dict, file)

df.to_csv('test-cleaned.csv', index=False)