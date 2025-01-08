
import torch, numpy as np, pandas as pd
from fastai.data.transforms import RandomSplitter
from torch import *
import tensorflow as tf

LEARNING_RATE = 1.0
EPOCHS = 30

# Data cleaning

csv_path = './data/bangladesh_divisions_dataset.csv'

data_frame = pd.read_csv(csv_path)
# A lot of this data isn't relevant to us
cols_to_drop = ['Location', 'Land_Use_Type', 'Crop_Suitability', 'Season', 'Satellite_Observation_Date', 'Remarks']
for col in cols_to_drop:
    data_frame.drop(col, axis=1, inplace=True)

# Separate out columns that are categories
data_frame = pd.get_dummies(data_frame, columns=['Soil_Type'], dtype="float")


# Machine Learning

# Add up the rows of (independent values * coefficients)
def calculate_predictions(coeffs, indep_tensor):
    return (indep_tensor * coeffs).sum(axis = 1)

# Generate loss by comparing the prediction and the dependent, then taking an average
def calculate_loss(coeffs, indep_tensor, dep_tensor):
    return torch.abs(calculate_predictions(coeffs, indep_tensor) - dep_tensor).mean()

# set the coefficient gradients back to zero
def zero_coefficients(coeffs):
    coeffs.sub_(coeffs.grad * LEARNING_RATE)
    coeffs.grad.zero_()

# Calculate one gradient descent step
def one_epoch(coeffs, indep_tensor, dep_tensor): # Might need to pass these in
    loss = calculate_loss(coeffs, indep_tensor, dep_tensor)
    loss.backward()
    with torch.no_grad(): zero_coefficients(coeffs)
    print(f"Loss: {loss:.3f}")

# This builds a Tensor of random numbers and calculates gradients on these coefficients
def initialize_coeffs(n_coeffs):
    return (torch.rand(n_coeffs)-0.5).requires_grad_()

# Training function
def train_model(coeffs, indep_tensor, dep_tensor):
    for i in arange(0, EPOCHS):
        print(f"Epoch {i}")
        one_epoch(coeffs, indep_tensor, dep_tensor)
    return coeffs

# Calculate accuracy
def calculate_accuracy(coeffs, val_indep, val_dep):
    # Accuracy within +/- 5% fertility index for now
    pred = calculate_predictions(coeffs, val_indep)
    above_threshold = (pred > val_dep - 5).bool()
    below_threshold = (pred < val_dep + 5).bool()
    mult = above_threshold * below_threshold
    return mult.float().mean()

# Variable we want to predict/measure
tensor_dependent = tensor(data_frame["Fertility_Index"])

# Variables we want to check for correlation with the dependent variable
independent_cols = ['Average_Rainfall(mm)', 'Temperature(°C)', 'Soil_Type_Clay', 'Soil_Type_Loamy', 'Soil_Type_Peaty', 'Soil_Type_Sandy', 'Soil_Type_Silt']
tensor_independent = tensor(data_frame[independent_cols].values, dtype=torch.float)

# Map all floats in independent cols to 0-1 based on their maximum- a percentage of maximum if you'd like
# Stops the sums of each row from being dominated by large value columns
values,indexes = tensor_independent.max(dim=0)
tensor_independent = tensor_independent / values

# We need a coefficient between -0.5/0.5 for each independent variable
num_coefficients = tensor_independent.shape[1]
coefficients = initialize_coeffs(num_coefficients)
    

#Separate out training and validation sets
train_split,validation_split=RandomSplitter()(data_frame)
train_independent, validation_independent = tensor_independent[train_split], tensor_independent[validation_split]
train_dependent, validation_dependent = tensor_dependent[train_split], tensor_dependent[validation_split]

# Train the model on the training split
train_model(coefficients, train_independent, train_dependent)
# Test for accuracy with the validation split
accuracy = calculate_accuracy(coefficients, validation_independent, validation_dependent)
print(f"Accuracy: {str(accuracy.item() * 100)}")