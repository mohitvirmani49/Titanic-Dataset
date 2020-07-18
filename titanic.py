import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data = pd.read_csv('titanic_data.csv')

x = [x for x in data.columns if (x != 'Name') and (x != 'Cabin') and (x != 'Ticket')]
x1 = data[x]
y = data.iloc[:, 1].values

x1.isnull().sum()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x1.iloc[:, 4:5])
x1.iloc[:, 4:5] = imputer.fit_transform(x1.iloc[:, 4:5])

x1.isnull().sum()

imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer2 = imputer2.fit(x1.iloc[:, 8:9])
x1.iloc[:, 8:9] = imputer2.fit_transform(x1.iloc[:, 8:9])

x1.isnull().sum()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
x1.iloc[:, -1] = label_encoder.fit_transform(x1.iloc[:, -1])
x1.iloc[:, 3] = label_encoder.fit_transform(x1.iloc[:, 3])


def showData(x1):
    for i in range(2, len(x1.columns)):
        parameters = x1.columns[i]
        plt.hist(x=[x1[x1['Survived'] == 1][parameters], x1[x1['Survived'] == 0][parameters]],
                 label=['Survived', 'Dead'])
        plt.title('Survival')
        plt.xlabel(parameters)
        plt.ylabel('Number of Passengers')
        plt.legend()
        plt.style.use('fivethirtyeight')
        plt.show()


showData(x1)

newX = [newX for newX in x1.columns if x != 'Survived']
X = x1[newX]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=6)

from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression()
regressor.fit(x_train, y_train)
y_prediction = regressor.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)
print("You scored ", accuracy)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
acc = accuracy_score(y_test, y_prediction)
print(acc)
plt.scatter(x_test,y_test)
plt.show()