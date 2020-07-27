import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt

#can create from scratch or existing python array
array1 = np.array([1,2,3,4])

#create array of zeroes
zeroes = np.zeros((3,4))
ones = np.ones((3,4))

#can specify data type
onesInt = np.ones((3,4), dtype=np.int16)

#uninitialized
empty = np.empty((3,4))

#eye creates identity matrix of specified size
identity = np.eye(5)

#arange takes start/end and step to create array
evens = np.arange(2,20,2)

#create 2d array with list of lists
#shape is tuple array.shape
#reshape allows resizing of array
numbers = np.arange(6)
array2 = numbers.reshape(3,2)


#elementwise operations if same dimensions
tens = np.array([10,10,10])
fives = np.array([5,5,5])
fives2 = tens - fives
fifteens = fives + tens
fifties = fives * tens
twos = tens / fives

#using scalar or boolean
twenties = tens * 2
tens *= 1
tens += 0
falses = tens > 20

array2d = np.array([[0,1],[2,3]])
array2d2 = np.array([[4,5],[6,7]])
#same operations applicable
#matrix multiplication done using dot product
mult = array2d.dot(array2d2) #or np.dot(array, array)

#standard list operations
thirty = tens.sum()
ten = tens.min()
ten = tens.max()

array2d3 = np.arange(12).reshape(3,4)
#operations can be done on 2d array using axis
colSums = array2d3.sum(axis=0)
rowMins = array2d3.min(axis=1)

#also has np.pi, np.cosm, np.tan, np.degrees...
angles = np.arange(0,91,15)
radians = np.radians(angles)
sines = np.sin(radians)

#statistical values
avgs = np.mean(array1)
medians = np.median(array2)

#read in csv
#np.genfromtxt(filename, delimiter=',')

#indexing, slicing similar to lists
squares = np.arange(11) ** 2
four = squares[2]
hundred = squares[-1]
subset = squares[3:-2]
reverse = squares[::-1]

#iterating similar as well
for square in squares:
    root = square ** .5
flat = squares.flatten() # iterates over every value in array, relevant for non 1d matrices, order='F' to go down columns first
#nditer similar to flatten but doesnt create new array

geo = np.array([['Germany', 'France', 'Hungary', 'Austria'], ['Berlin', 'Paris', 'Budapest', 'Vienna']])
transpose = geo.T
#can use ravel to flatten as well

array2d4 = np.arange(15).reshape(5,3)
#reshape only works if same number of elements

#use -1 to specify an unknown value, will use as many as needed - must be divisible by the non -1 element
array2d4 = np.arange(15).reshape(-1,3)

array3 = np.arange(9)
triples = np.split(array3,3) #splits into array of 3 arrays

specific = np.split(array3, [4,7]) #divides array at specified indices

smallGeo = np.hsplit(geo, 2) #split can be done horizontally as well
longGeo = np.vsplit(geo, 1)

#image analysis
#each pixel has 3 values for RGB (0-255), image is matrix of pixels
#grayscale matrix of 0-1 depending how light or dark the pixel is
randimg = misc.face()
shape = randimg.shape #RGB
#type is np.array
#select portion of image
subimg = randimg[384:, 512:, :]
top,bottom = np.split(randimg,2)
left,right = np.split(randimg,2,axis=1)
original = np.concatenate((top, bottom)) #concatenates on columns
original = np.concatenate((left, right), axis=1)
plt.imshow(randimg)
plt.show()
plt.imshow(original)
plt.show()
#show that images are the same

#shallow copies, base is fruits
fruits = np.array('apple', 'banana', 'orange', 'grape')
basket1 = fruits.view()
basket2 = fruits.view()
basket2[0] = 'strawberry'
#fruits and basket1 will also change - but can reshape without the original being reshaped

basket = fruits.copy()
#deep copy, base not fruits















