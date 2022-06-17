import numpy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
import numpy as np
import Lin
import Gip
from matplotlib import pyplot as plt

jL = Lin.LinDataSet(20)
jG = Gip.GipDataSet(20)

coefsL, absL, targetL = jL.Find()
coefsG, absG, targetG = jG.Find()

x = np.concatenate((coefsL, coefsG))
y = np.concatenate((targetL, targetG))

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

ppnrt = MLPClassifier(hidden_layer_sizes=(80, 14, 2 ), activation="relu", solver="adam",
         random_state=42, max_iter=100, learning_rate="constant").fit(xTrain, yTrain)
print(ppnrt.score(xTrain, yTrain))

train_sizesrl, train_scoresrl, test_scoresrl = learning_curve(estimator=ppnrt, X=xTrain, y=yTrain,cv=10, train_sizes=np.linspace(0.1, 1.0, 10))

train_meanrl = np.mean(train_scoresrl, axis=1)
train_stdrl = np.std(train_scoresrl, axis=1)
test_meanrl = np.mean(test_scoresrl, axis=1)
test_stdrl = np.std(test_scoresrl, axis=1)

plt.figure(1)
plt.plot(train_sizesrl, train_meanrl, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizesrl, train_meanrl + train_stdrl, train_meanrl - train_stdrl, alpha=0.15, color='blue')
plt.plot(train_sizesrl, test_meanrl, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizesrl, test_meanrl + test_stdrl, test_meanrl - test_stdrl, alpha=0.15, color='green')
plt.title('Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()