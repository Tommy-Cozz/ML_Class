import pandas as pd
import numpy as np
import matplotlib as plt



"""
Need to preproccess and clean up the data and ensure there are no blanks, duplicates, or noise(What would constitute as noise in a data set im not sure but maybe i can find out) 

"""


df_raw = pd.read_csv('USA_Housing.csv')              

# df_raw.drop_duplicates(subset = None, keep = 'first', inplace = True)




df_clean = df_raw.dropna()



#Indiviual Statistics 
mean_df = df_clean.mean()
median_df =df_clean.median()
min_df = df_clean.min()
#Pandas Describe function 
df_clean.describe()
df_clean.columns


x = np.random.randn(10,1)
y = 5*x + np.random.rand()

w = 0.0 
b = 0.0 
num_epochs = 1000
# Hyperparameter 
learning_rate = 0.01


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true- y_pred)**2)

for epoch in range(num_epochs):
    y_pred = np.dot(x, w) + b
    loss = mean_squared_error(y, y_pred)

    dw = -2 * np.dot(x.T, (y-y_pred)) / len(y)
    db = -2 * np.sum(y- y_pred) / len(y)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss}')


#Regression utalizing the housing data provided 

#Load Data/ Split into test and train 
#Test and Train split is Test = 8% Train = 92%, decimal ratio to be used 11.5


df = pd.read_csv('USA_Housing.csv')
ratio = 11.5

total_rows = df.shape[0]
train_size = int(total_rows*ratio)
train = df[0:train_size]
test = df[train_size:]


x = train
y = test

w = 0.0 
b = 0.0 
num_epochs = 1000
# Hyperparameter 
learning_rate = 0.01


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true- y_pred)**2)

for epoch in range(num_epochs):
    y_pred = np.dot(x, w) + b
    loss = mean_squared_error(y, y_pred)

    dw = -2 * np.dot(x.T, (y-y_pred)) / len(y)
    db = -2 * np.sum(y- y_pred) / len(y)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss}')









