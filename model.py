# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



from sklearn.metrics import accuracy_score

dataset = pd.read_csv('studentrecord.csv')

dataset['Gender'] = dataset['Gender'].apply({'Male':0, 'Female':1}.get)
dataset['Backlogs'] = dataset['Backlogs'].apply({'No':0, 'Yes':1}.get)
dataset['placement_status'] = dataset['placement_status'].apply({'No (Not Placed)':0, 'Yes (Placed)':1}.get)
dataset['social_media_usage'] = dataset['social_media_usage'].apply({'No':0, 'Yes':1}.get)
dataset['physical_activites'] = dataset['physical_activites'].apply({'No':0, 'yes':1}.get)
dataset['core_concepts'] = dataset['core_concepts'].apply({'Beginner':1, 'Expert':2,'Proficient':3}.get)
dataset['coding'] = dataset['coding'].apply({'Beginner':1, 'Expert':2,'Proficient':3}.get)
dataset['communication'] = dataset['communication'].apply({'Beginner':1, 'Expert':2,'Proficient':3}.get)
dataset['stay'] = dataset['stay'].apply({'Hostel':0, 'Dayscholar':1}.get)
dataset['sleep(hrs)'] = dataset['sleep(hrs)'].apply({'Below 8':1, 'Exactly 8':2, 'Above 8':3}.get)
dataset['Library_usage'] = dataset['Library_usage'].apply({'Below 10':1, 'Between 11 to 20':2, 'Above 21':3}.get)
dataset['Attendance'] = dataset['Attendance'].apply({'Below 75':1, 'Between 76 to 89':2, 'Above 90 %':3}.get)
dataset['study_hours']=dataset['study_hours'].apply({ '0':1, 'Between 1 to 5':2, 'Above 5':3}.get)

 

dataset.drop(['Name'],axis = 1, inplace =True)

X = dataset.drop(["placement_status"] , axis=1)
Y = dataset.placement_status


from sklearn.feature_selection import SelectKBest, chi2, f_classif
modelF =  SelectKBest(chi2,k=12)
new = modelF.fit(X,Y)
x_new = new.transform(X)
cols = new.get_support(indices=True)
features_new = X.iloc[:,cols]
features_new.drop(['Gender','hrs_in_social_media'],axis=1, inplace=True)
features_new.head(5)

X = dataset.drop(["placement_status",'Gender','social_media_usage','core_concepts','sleep(hrs)','Library_usage','meal_intake','study_hours','hrs_in_social_media'] , axis=1)
Y = dataset.placement_status

#X = features_new.drop(["placement_status"] , axis=1)
#Y = features_new.placement_status

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.30,random_state=23,stratify=Y)



from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics



clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

clf = clf.fit(X_train,Y_train)


y_pred = clf.predict(X_test)

pickle.dump(clf, open('clf.pkl','wb'))


model = pickle.load(open('clf.pkl','rb'))


test_acc = accuracy_score(Y_test, y_pred)
print("The Accuracy for Test Set is {}".format(test_acc*100))






