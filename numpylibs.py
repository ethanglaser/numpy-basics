import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

titanic = pd.read_csv('data/titanic.csv')
#drop a column
dropcabin = titanic.drop(columns=['Cabin'])

unknownages = np.sum(titanic['Age'].isna())

#find average of all numeric values associated with each port
avgbyport = titanic.groupby('Embarked').aggregate(np.mean)
#find specific column data
dat = avgbyport[['Age', 'Fare']]

survivalrate = np.count_nonzero(titanic['Survived']) / np.count_nonzero(titanic['PassengerId'])

females = titanic[titanic['Sex'] == 'female']
femalesurvival = np.count_nonzero(females['Survived']) / np.count_nonzero(females['PassengerId'])
firstclass = titanic[titanic['Pclass'] == 1]
firstsurvival = np.count_nonzero(firstclass['Survived']) / np.count_nonzero(firstclass['PassengerId'])


#can skip header if header included
ages = np.genfromtxt('data/Ages.csv',delimiter=',',skip_header=1)
years = ages[:,0]
males = ages[:,1]
#interpolation function creation
inter = interp1d(years, males)
interpolated99 = inter(1999)

#can vectorize a function for integers using np.vectorize(function)

