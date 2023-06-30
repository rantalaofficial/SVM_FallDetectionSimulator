import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib


# read trough a file line by line

file = open("data.txt", "r")
lines = file.read()
file.close()

inertialData = []
labels = []

for line in lines.split('\n'):
    data = line.split("A")
    label = data[0]
    
    parts = data[1].split("B")
    inertialArray = (parts[0].strip() + ' ' + parts[1].strip()).split(' ')
    
    inertialArray = [float(x) for x in inertialArray]

    print(label + ':')
    print(inertialArray)

    inertialData.append(inertialArray)
    labels.append(label)

X = np.array(inertialData)
y = np.array(labels)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)

def predict(data):
    return int(clf.predict([data])[0])

print(predict([532.754, 619.754, 713.954, 815.354, 923.954, -26.0, -27.0, -28.0, -29.0, -30.0]))
print(predict([1322.073, 1415.752, 1512.485, 1612.781, 0.0, -75.0, -79.0, -83.0, -87.0, -91.0]))

joblib.dump(clf, "SVM_MODEL.joblib")