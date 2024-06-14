import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('Data/assign_4_data.csv')



X = df[['x1','x2','x3']]
y = df['y']
# theta = [0,0,0,0]

"""
From the problem 2
Initial Values Theta0 =Theta1 = Theta2 = Theta 3 = 0
Learning rate = .5
Iterations N = 3
Compute both the parameters Theta0,1,2,3 and the cost after each iteration
"""




def hypothesis(X, theta):
   return np.dot(X, theta)

def multivar_cost(X, y, theta):
    m = len(y)
    error = hypothesis(X, theta) - y
    return (1/ (2*m)) * np.dot(error, error)


def grad_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    cost_history =[]

    for _ in range(num_iterations):
        error = hypothesis(X, theta) - y
        gradient = (1/m) * np.dot(X.T, error)
        theta = theta - alpha * gradient
        cost = multivar_cost(X, y, theta)
        cost_history.append(cost)
        print(f"Theta Array: {theta}, Cost Array: {cost}")
    return theta, cost_history



#Intializing theta and hyperparameters
theta = np.zeros(X.shape[1]) #Intalize coefficients to zeros
alpha = 0.01 # Learning Rate 
num_iterations = 1000


theta, cost_history = grad_descent(X, y, theta, alpha=.5, num_iterations=3)

print(f"Final Theta Array: {theta}, Final Cost Array: {cost_history}")
