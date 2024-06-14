import numpy as np




x = np.random.randn(10,1)
y = 2*x + np.random.rand()

w = 0.0
b = 0.0

learning_rate = 0.001


def descent(x,y ,w, b , learning_rate):
    dldw = 0.0          #initializing parameters (my guess this is the theta knot and theta 1)
    dldb = 0.0
    N = x.shape[0]      #calculates the size of the data set do the equivalent in pandas as you need the average so you need the total number of data points 
    #loss = (y-wx+b)))**2
    for xi, yi in zip(x,y):
        dldw += -2*xi(yi -(w*xi+b)) 
        dldb += -2*xi(yi-(w*xi+b))         


    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb
    return w,b



for epoch in range(100):
    w,b = descent(x,y,w,b,learning_rate)
    yhat = (w*x + b)
    loss = np.divide((y-yhat)**2, axis = 0), x.shape[0] 
    print(f'{epoch} loss is {loss}, parameters w:{w}, b:{b}')

























