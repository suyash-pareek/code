
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

#Step 1 Load Dataset
dataframe = pd.read_csv("C:\Users\suyas\OneDrive\Desktop\spam.csv")
print(dataframe.describe())


#Step 2 Split data into training and testing
x = dataframe["EmailText"]
y = dataframe["Label"]

x_train, y_train = x[0:4457], y[0:4457]
x_test, y_test = x[4457:], y[4457:]

#Step 3 Extract Features
cv = CountVectorizer()     
features = cv.fit_transform(x_train)     #Transforming string data into arrays of count of strings

#Step 4 Build Model
model = svm.SVC()       #Using support Vector Model beacause of works well in such classifications
model.fit(features,y_train)

#Step 5 Test the Model
features_test = cv.transform(x_test)
model.score(features_test,y_test)