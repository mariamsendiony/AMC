import glob
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit




# Get a list of all the files in the dataset
files = glob.glob('D:\Downloads_D\Datasetmodified - Copy\*')

# Initialize a dictionary to store the magnitudes,Cumulants
magnitudes = {}
Cumulants={}

# read the file into a DataFrame.
for file in files:
    df = pd.read_csv(file)

    # Get the second row ,here it skips the fiirst row automatically unless you type header=None
    row = df.iloc[0]

    # Create a dictionary.
    magnitudes[file] =[]
    

    # Iterate over the rows starting from the second row.
    for i in range(0, len(df)):

        # For each row, calculate the magnitude of the two elements in the row and add the magnitude to the dictionary.
        magnitudes[file].append(np.sqrt(df.iloc[i][0]**2 + df.iloc[i][1]**2))

# Print the dictionary.
for key, value in magnitudes.items():
    print(key, value)
#Calculating the cumulants
print('Cumulants are coming:')
for file in files:
 df = pd.read_csv(file)
 Cumulants[file]=[]
 Cumulants[file].append(sum(magnitudes[file])/len(magnitudes[file]))
 mean=sum(magnitudes[file])/len(magnitudes[file])
 
 variance = Cumulants[file].append(sum((x - mean)**2  for x in magnitudes[file]) / len(magnitudes[file]))
 variance=sum((x - mean)**2  for x in magnitudes[file]) / len(magnitudes[file])
 skewness=Cumulants[file].append(sum((x - mean)**3 for x in magnitudes[file] ) / ((len(magnitudes[file]) - 1) * variance**(3)))
 kurtosis =Cumulants[file].append((sum((x - mean)**4 for x in magnitudes[file] ) / ((len(magnitudes[file]) - 1) * variance**4 ))- 3)

 for key,value in Cumulants.items():
     print(key,value)

with open('D:\Downloads_D\Datatoprocess.txt', 'w') as f:

    

    # Write the keys in an output file
    writer = csv.writer(f)
    writer.writerow([ 'Modulation Type', 'SNR','Rotation','First Cumulant','Second Cumulant','Third Cumulant','Fourth Cumulant' ])
    for file in files:    
        df = pd.read_csv(file,header=None)
        temp = df.iloc[0]
        temp = list(temp)
        for i in Cumulants[file]:
            temp.append((i)) 
        writer.writerow((temp[0] + str(temp[1]), temp[2],temp[3],temp[4],temp[5],temp[6],temp[7])) 
    
bankdata=pd.read_csv('D:\Downloads_D\Datatoprocess.txt')
print(bankdata.shape)
#Data Preprocessing
x=bankdata.drop('Modulation Type',axis=1)
y=bankdata['Modulation Type']
print(x)
print(y)

param_grid = {'gamma': [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'C': [1, 10, 100, 1000, 10000, 100000,1000000,10000000,100000000,1000000000]}


cv = StratifiedShuffleSplit(n_splits=13, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
grid.fit(x, y)
print(
    "The best parameters are %s with a score of %f"
    % (grid.best_params_, (grid.best_score_))
)










# Print the best parameters and score
print(grid.best_params_, grid.best_score_)


      





