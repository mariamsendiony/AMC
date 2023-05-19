import glob
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


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


  
# Creating the classifier object
clf_gini = DecisionTreeClassifier(criterion='gini', splitter='best', 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_features=None, 
    random_state=None, 
    max_leaf_nodes=None, 
    min_impurity_decrease=0.0, 
    class_weight=None, 
    ccp_alpha=0.0)
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, random_state = 50)
# Performing training
clf_gini.fit(X_train, y_train)
# Testing
y_pred_gini =clf_gini.predict(X_test)
#Calculating Accuarcy
print ("Accuracy : ",
    accuracy_score(y_test,y_pred_gini)*100)
  


      





