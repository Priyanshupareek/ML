import pandas as pd
import numpy as np
import sklearn as skl
from sklearn import linear_model
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1","G2","G3","studytime","failures","absences"]]

predict="G3"

x = np.array(data.drop([predict],'columns'))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x, y, test_size=0.1)

'''
best = 0
for _ in range(30000):

    x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x , y , test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    if acc > best:
        best = acc
        with open("student.pickle", "wb") as f:
            pickle.dump(linear, f)
        print(acc)
'''
pickle_in = open("student.pickle" , "rb")
linear = pickle.load(pickle_in)

'''print('Coef: ' , linear.coef_)
print('Intercept: ' , linear.intercept_)
'''
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(int(predictions[x]),x_test[x] , y_test[x])

p = "absences"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("G3")
pyplot.show()