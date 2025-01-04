import torch, numpy as np, pandas as pd

csv_path = './data/bangladesh_divisions_dataset.csv'

data_frame = pd.read_csv(csv_path)
# A lot of this data isn't relevant to us
cols_to_drop = ['Location', 'Land_Use_Type', 'Crop_Suitability', 'Season', 'Satellite_Observation_Date', 'Remarks']
for col in cols_to_drop:
    data_frame.drop(col, axis=1, inplace=True)

#Separate out columns that are categories
data_frame = pd.get_dummies(data_frame, columns=['Soil_Type'])
print(data_frame.columns)

#We only care about ethnicity hispanic/latino, 