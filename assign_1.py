import numpy as np
import matplotlib.pyplot as plt


"""
First problem need to clean up the output but the three funtions work as they are supposed to 
"""


v1 = np.array([[2],[4],[6]])
v2 = np.array([[0.6],[-1],[5]])
a1 = np.array([[1,-1,0],[2,1,5],[4,2,-3]])



sum = v1 + v2
dot = v1 * v2
trsp = np.transpose(a1)


mtx_vprod = a1 @ v1

trsp_mtrx_vprod = trsp@  v1

print(f'Sum of vectors 1 and 2: {sum} \n Dot product of vectors 1 and 2: {dot} \n Matrix Vector Product of Adjacency matrix and vector 1: {mtx_vprod}, \n Matrix vector product of Transpoed Adjancecy Matrix and Vector 1: {trsp_mtrx_vprod} \n')

"""
2. Create the adjacency matrix in numpy desribing the graph
"""

from numpy import linalg

adj_mtx = np.array([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]])


eig_vals, eig_vect = np.linalg.eig(adj_mtx)


fst_col = eig_vect[:,0]


prod_mtrx = fst_col * adj_mtx

print(f'Matrix product of first eigenvector and adjacency matrix: {prod_mtrx} ')

"""
Explain how this relates to the eigenvector equation 
    This relates due to the fact that when you multiply the egienvectors x by A it will keep the same direction ensuring the determinate to be 0 which also satisifes the equation 

"""




"""
Create Pandas prograsm that merges the provided data frames 

"""

import pandas as pd


df = pd.read_csv('dataA.csv')
df2 = pd.read_csv('dataB.csv')

combined_df = pd.concat([df,df2])
df_horz = pd.merge(df,df2, how ="outer" )
df_vert = pd.merge(df,df2,  how ="inner" )


print(f'\n Combined data frame horizantal{df_horz} \n\n\n Combined Dataframe Vertically {df_vert}')

"""
With these two sets of data graph the below graph 
"""



x = [1,2,31,5,60,17,18,9,10,12,13,4,15,16,18,19,21,22]
y = [10,9,80,60,6,55,60,65,70,7,75,76,8,79,10,9,9,10]







plt.scatter(x,y)
plt.title('List Given in Assignment 1')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()




