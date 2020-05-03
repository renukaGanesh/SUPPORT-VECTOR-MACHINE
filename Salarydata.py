import pandas as pd 
import numpy as np 
import seaborn as sns

sal1 = pd.read_csv("C:\\Users\\user\\Downloads\\SalaryData_Train.csv")
sal1.head()
sal1.columns
# Converting non categorical values into categorical values
b1 = sal1.iloc[:,[1,2,4,5,6,7,8,12]]
b2 = sal1.iloc[:,[0,3,9,10,11]]
ds=b1

def convert(i): #x means the column frpm the dataset to be converted to category
    ds[i] = ds[i].astype('category')
    ds[i] = ds[i].cat.codes
for i in range(ds.shape[1]):
    convert(ds.columns[i])
print(ds)


sal1["Salary"] = sal1["Salary"].astype("category")
sal1["Salary"] = sal1["Salary"].cat.codes
c3=sal1["Salary"]

sal1_new = pd.concat([b2,ds,c3],axis =1)
sal1_new.head(10)

sns.pairplot(data=sal1_new)
############################
sal2 = pd.read_csv("C:\\Users\\user\\Downloads\\SalaryData_Test.csv")
sal2.head()
sal2.describe()
sal2.columns

# Converting non categorical values into categorical values
c1 = sal2.iloc[:,[1,2,4,5,6,7,8,12]]
c2 = sal2.iloc[:,[0,3,9,10,11]]
ds=c1

def convert(i): #x means the column frpm the dataset to be converted to category
    ds[i] = ds[i].astype('category')
    ds[i] = ds[i].cat.codes
for i in range(ds.shape[1]):
    convert(ds.columns[i])
print(ds)

sal2["Salary"] = sal2["Salary"].astype("category")
sal2["Salary"] = sal2["Salary"].cat.codes
c4=sal2["Salary"]

sal2_new = pd.concat([c2,ds,c4],axis =1)
# Finding the correlation by taking pairplot
sns.pairplot(data=sal2_new)

#####################################
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(sal1_new,test_size = 0.3)
train,test = train_test_split(sal2_new,test_size = 0.3)
train_X = sal1_new.iloc[0:100,:-1]
train_y = sal1_new.iloc[0:100,-1]
test_X = sal2_new.iloc[0:100,:-1]
test_y = sal2_new.iloc[0:100,-1]
# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 0.87%

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 0.82%

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 80%

