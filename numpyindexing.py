import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

array = np.arange(12) ** 2
four = array[2]

indices = [2,6,8]
values = array[indices] #can use list of indices
indices2 = np.array([[2,4],[8,10]])
values2 = array[indices2] #matches shape of indices
#can set values or reset matrix entries using this

data = pd.read_csv("data/gdp_pc.csv")['2016'].values
plt.plot(data)
#plt.show()

#remove nan values (fancy indexing) (data.median invalid but dataClean.median is)
dataClean = data[~np.isnan(data)]

#count elements that match a condition
#without count_nonzero would be an array of values over 40k
#np.nonzero will count all true values
num = np.count_nonzero(dataClean[dataClean > 40000])

#sort array and perform analysis
lowest = np.sort(dataClean)[:10]
highest = np.sort(dataClean)[-10:]

array2 = np.arange(16).reshape(4,4)
#get array of booleans based on original
indexBool = array2 > 9 #=array2[array2 > 9]
#get based on rows
indexBoolRows = np.sum(array2 < 6, axis=1)

#any and all value checks entire array to return 1 value
true = np.any(array2 > 8)
false = np.all(array2 > 8)

#numpy can do some things that pandas can as well

#broadcasting - working on matrices with mismatched dimensions
#with scalar, numpy broadcasts 1x1 scalar to size of matrix (smaller array broadcast to match larger dimensions)
#can only broadcast smaller array if shape matches in some way (3x1 can be broadcast to multiply with a 3x6 array, 3x2 cannot)
#need 1 dimension to match and other to be 1
heights = np.array([165,170,168,183,172,169])
weights = np.array([61,76,56,81,62,60])
student = np.array([heights, weights])
factors = np.array([[0.0328084], [2.20462]])
#only works if both are 2D arrays with 1 matching dimensions and one array with a dimension of 1
revised = student * factors

array3 = np.arange(30)
#can set shape, -1 is unknown (can only be for 1 dimension)
array3.shape = 2,3,-1

x = np.array([[1,2],[3,4]])
y = np.array([[4,3],[2,1]])
#can combine/stack arrays, options for axes available (vert equivalent to combined)
combined = np.concatenate((x,y))
horiz = np.hstack((x,y))
vert = np.vstack((x,y))

#histogram function with data and bins
hist = np.histogram([1,2,1], bins = [0,1,2,3])
data = np.random.normal(2, 0.5, 10000)
(n, binEdges) = np.histogram(data, bins=50)
#visualize with matplotlib
plt.plot(binEdges[1:], n)
#plt.show()

salaries = np.genfromtxt('data/salary.csv', delimiter=',')
#returns index of max/min value
argmax = np.argmax(salaries)
argmin = np.argmin(salaries)
#return array of indices by the order of corresponding value (sortIndex[-1] is argmax)
sortIndex = np.argsort(salaries)
sortSalaries = salaries[sortIndex]
#array of indices whose corresponding value matches condition
over100k = np.where(salaries > 100000)
aboveavg = np.where(salaries > np.mean(salaries))

