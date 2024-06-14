"""
Task 2 Initial values for theta knot and theta 1 are 0.5 and 0 
Learning rate will be set to 0.002

"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#yhat = wx + b
#loss = (y - yhat)**2 /N
df = pd.read_csv('grad_descdat.csv')

x = df['x']
y = df['y']


w = 0.5             #theta knot intial value
b = 0              #theta 1 inital value

learning_rate = 0.002




#Gradient Funtionm
def descent(x,y,w,b, learning_rate):
    dldw = 0.0                                         #Loss with resoect to w (theta knot)
    dldb = 0.2                                         #Loss with respect to b (theta 1)
    N = x.index                                               

    for xi, yi in zip(x,y):                            #Utalizes the zuip function to iterate through x and y while calculating the partial derivatives 
       dldw += -2*xi*(yi -(w*xi+b)) 
       dldb += -2*(yi-(w*xi+b))  
    
    w = w - learning_rate*(1/N)*dldw                  #calculates the new values for theta knot and theta 1 to be plugged into the function again over the iterations
    b = b - learning_rate*(1/N)*dldb
    return w,b


#Interate 4 times through the function to determine cost function and 

for epoch in range(4):                                                    #epoch is just the amount of times the function will run or iterate through in our case it will be 4 
    w,b = descent(x,y,w,b,learning_rate)
    yhat = (w*x+b)
    loss = np.divide(np.sum((y-yhat)**2, axis =0), x.shape[0])                       #Calculates the loss equation 
    
    print(f'Values  \n {epoch} loss is: {loss},\n Parameteres theta_knot:{w}, theta_1{b}')
# print(x,y)
x_unkwn = 3.80 

y_predictionhat = (w*x_unkwn+b)             #allows for the calculation of the requested value 

print(f'Unkown x : {x_unkwn}, \n Predicted value: {y_predictionhat}')
theta = np.polyfit(x,y,1)
y_line = theta[1] + theta[0] * x
   
plt.scatter(x,y)
plt.plot(x,y_line, 'r')
plt.title('Line of Best Fit')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()















