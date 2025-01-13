import pandas as pd
from fastai.tabular.all import *
from matplotlib import pyplot

# Data
csv_path = './data/cars.csv'
data_frame = pd.read_csv(csv_path)
name_split = data_frame['name'].str.split(' ', expand=True)
brands = name_split[0]
data_frame['brand'] = brands
data_frame['model'] = ''
data_frame['selling_price'].astype(float)
data_frame['km_driven'].astype(float)
data_frame['year'].astype(float)
max = data_frame['selling_price'].max()

data_frame['selling_price_norm'] = data_frame['selling_price'] / max

for index in range(0, len(data_frame)):
    data_frame.loc[index, 'model'] = data_frame.loc[index, 'name'].replace(brands[index] + ' ', '')

# Machine Learning

splits = RandomSplitter(valid_pct=0.2)(range_of(data_frame))
to = TabularPandas(data_frame, procs=[Categorify, FillMissing,Normalize],
                   cat_names=['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'model'],
                   cont_names=['km_driven', 'year'],
                   y_names='selling_price_norm',
                   splits=splits)

dataloader = to.dataloaders(bs=256)

dataloader.device = torch.device("mps")




# Layers defines the number of hidden layers
learn = tabular_learner(dataloader, metrics=[rmse], layers=[1000, 500])


# First arg is no of epochs
learn.fit(30, lr=0.1)
test_feature = data_frame.iloc[:2]
test_dl = learn.dls.test_dl(test_feature)
predictions = learn.get_preds(dl=test_dl)

# Look on my works ye mighty and despair!
print(predictions)
print(max)
print(predictions[1] * max)

learn.export('car-linear-regression.pkl')
