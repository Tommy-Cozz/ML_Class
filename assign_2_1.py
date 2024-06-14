import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#yhat = mx + b
#loss = (y - yhat)**2 /N
df = pd.read_csv('grad_descdat.csv')

x = df['x']
y = df['y']


w = 0             #theta knot intial value
b = 0.2              #theta 1 inital value

learning_rate = 0.001




#Gradient Funtion
def descent(x,y,w,b, learning_rate):
    dldw = 0.0                                         #Loss with respect to m 
    dldb = 0.2                                          
    N = x.shape[0]                                               

    for xi, yi in zip(x,y):
       dldw += -2*xi*(yi -(w*xi+b)) 
       dldb += -2*(yi-(w*xi+b))  
    
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb
    return w,b


#Interate 4 times through the function to determine cost function and optimal values through for loop interation

for epoch in range(4):
    w,b = descent(x,y,w,b,learning_rate)
    yhat = (w*x+b)
    loss = np.divide(np.sum((y-yhat)**2, axis =0), x.shape[0])
    
    print(f'Values    \n {epoch} loss is P{loss},\n parameteres theta_knot: {w},\n theta_1: {b}')

# print(x,y)

x_unkwn = 3.80 

y_predictionhat = (w*x_unkwn+b)

print(f'Unkown x : {x_unkwn}, \n Predicted value: {y_predictionhat}')

#Use matplotlib to graph the line of best fit
theta = np.polyfit(x,y,1)
y_line = theta[1] + theta[0] * x
   
plt.scatter(x,y)
plt.plot(x,y_line, 'r')
plt.title('Line of Best Fit')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()






