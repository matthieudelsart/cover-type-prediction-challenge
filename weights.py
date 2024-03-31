import numpy as np
import pandas as pd

predictions = pd.read_csv('with_all_features.csv')["Cover_Type"].to_list()
length = len(predictions)
weight_class = {i:length/predictions.count(i) for i in range(1, 8)}

print(weight_class)