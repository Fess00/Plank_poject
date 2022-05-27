from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
import ReadLin as RL
import ReadTan as RT
from matplotlib import pyplot as plt


class Ui_Migration(object):
    def setupUi(self, Migration):
        Migration.setObjectName("Migration")
        Migration.resize(1122, 679)
        self.centralwidget = QtWidgets.QWidget(Migration)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 580, 291, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 580, 291, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(630, 580, 321, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.BaseSetLin = QtWidgets.QTableWidget(self.centralwidget)
        self.BaseSetLin.setGeometry(QtCore.QRect(10, 50, 491, 231))
        self.BaseSetLin.setObjectName("BaseSetLin")
        self.BaseSetLin.setColumnCount(4)
        self.BaseSetLin.setRowCount(10000)
        self.BaseSetHyp = QtWidgets.QTableWidget(self.centralwidget)
        self.BaseSetHyp.setGeometry(QtCore.QRect(540, 50, 491, 231))
        self.BaseSetHyp.setObjectName("BaseSetHyp")
        self.BaseSetHyp.setColumnCount(4)
        self.BaseSetHyp.setRowCount(10000)
        self.NetSetLin = QtWidgets.QTableWidget(self.centralwidget)
        self.NetSetLin.setGeometry(QtCore.QRect(10, 320, 491, 231))
        self.NetSetLin.setObjectName("NetSetLin")
        self.NetSetLin.setColumnCount(5)
        self.NetSetLin.setRowCount(10000)
        self.NetSeHyp = QtWidgets.QTableWidget(self.centralwidget)
        self.NetSeHyp.setGeometry(QtCore.QRect(540, 320, 491, 231))
        self.NetSeHyp.setObjectName("NetSeHyp")
        self.NetSeHyp.setColumnCount(5)
        self.NetSeHyp.setRowCount(10000)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 15, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(560, 10, 411, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 290, 451, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(560, 290, 471, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(960, 590, 141, 23))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.hide()
        Migration.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Migration)
        self.statusbar.setObjectName("statusbar")
        Migration.setStatusBar(self.statusbar)

        self.retranslateUi(Migration)
        QtCore.QMetaObject.connectSlotsByName(Migration)

        # поля и методы класса

        self.dataset_size = 1000

        self.base_input_data_rl = []
        self.input_data_rl = []
        self.output_data_rl = []
        self.base_input_data_rt = []
        self.input_data_rt = []
        self.output_data_rt = []

        self.xLrl = None
        self.yLrl = None
        self.xTrl = None
        self.yTrl = None
        self.yPrl = None

        self.xLrt = None
        self.yLrt = None
        self.xTrt = None
        self.yTrt = None
        self.yPrt = None

        self.ppnrl = None
        self.ppnrt = None

        self.rt = None
        self.lt = None

        self.clickBatton()

    def CreateDataSet(self):
        self.progressBar.show()
        self.progressBar.setValue(0)

        self.rt = RT.GipAprox(94.1, 16.4, 25.6, 0.3, 26, 2.7, 3, 0.25)
        self.base_input_data_rt = self.rt.GenSet(self.dataset_size)
        self.input_data_rt = self.rt.GenNetSet(self.dataset_size)
        self.output_data_rt = self.rt.GetTargets()

        self.progressBar.setValue(20)

        self.rl = RL.LinAprox(798, 3.6, 427, 0.291, 100)
        self.base_input_data_rl = self.rl.GenSet(self.dataset_size)
        self.input_data_rl = self.rl.GenNetSet(self.dataset_size)
        self.output_data_rl = self.rl.GetTargets()

        labelsBase = []
        labelsBase.append('C')
        labelsBase.append('C0')
        labelsBase.append('A')
        labelsBase.append('B')
        labelsNet = []
        labelsNet.append('E')
        labelsNet.append('Sx')
        labelsNet.append('St')
        labelsNet.append('G')
        labelsNet.append('Target')

        self.BaseSetLin.setHorizontalHeaderLabels(labelsBase)
        self.BaseSetHyp.setHorizontalHeaderLabels(labelsBase)
        self.NetSetLin.setHorizontalHeaderLabels(labelsNet)
        self.NetSeHyp.setHorizontalHeaderLabels(labelsNet)

        self.progressBar.setValue(40)

        for i in range(self.dataset_size):
            for j in range(5):
                if j == 4:
                    self.NetSetLin.setItem(i, j, QtWidgets.QTableWidgetItem(str(self.output_data_rl[i])))
                    self.NetSeHyp.setItem(i, j, QtWidgets.QTableWidgetItem(str(self.output_data_rt[i])))
                    break
                self.NetSetLin.setItem(i, j, QtWidgets.QTableWidgetItem(str(self.input_data_rl[i][j])))
                self.NetSeHyp.setItem(i, j, QtWidgets.QTableWidgetItem(str(self.input_data_rt[i][j])))
        
        self.progressBar.setValue(60)
        
        for i in range(self.dataset_size):
            for j in range(4):
                self.BaseSetLin.setItem(i, j, QtWidgets.QTableWidgetItem(str(self.base_input_data_rl[i][j])))
                self.BaseSetHyp.setItem(i, j, QtWidgets.QTableWidgetItem(str(self.base_input_data_rt[i][j])))

        self.progressBar.setValue(80)

        plt.figure(1)
        plt.hist(self.output_data_rl)
        plt.title("LinAprox")
        plt.show()
        plt.figure(2)
        plt.hist(self.output_data_rt)
        plt.title("HypAprox")
        plt.show()

        self.xLrl, self.xTrl, self.yLrl, self.yTrl = train_test_split(self.input_data_rl, self.output_data_rl,
         test_size=0.3, random_state=42, stratify=self.output_data_rl)

        self.xLrt, self.xTrt, self.yLrt, self.yTrt = train_test_split(self.input_data_rt, self.output_data_rt,
         test_size=0.3, random_state=42, stratify=self.output_data_rt)

        coef1 = np.corrcoef(self.input_data_rl, self.input_data_rl)
        print("LIN")
        print(coef1)

        coef2 = np.corrcoef(self.input_data_rt, self.input_data_rt)
        print("HYP")
        print(coef2)

        self.progressBar.setValue(100)
        self.progressBar.hide()

    def LoadNet(self):
        self.progressBar.show()
        self.progressBar.setValue(0)

        self.ppnrl = MLPClassifier(hidden_layer_sizes=(80, 60, 20 ), activation="relu", solver="adam",
         random_state=42, max_iter=100, learning_rate="constant").fit(self.xLrl, self.yLrl)
        print(self.ppnrl.score(self.xLrl, self.yLrl))

        self.progressBar.setValue(10)

        self.ppnrt = MLPClassifier(hidden_layer_sizes=(80, 14, 2 ), activation="relu", solver="adam",
         random_state=42, max_iter=100, learning_rate="constant").fit(self.xLrt, self.yLrt)
        print(self.ppnrt.score(self.xLrt, self.yLrt))

        self.progressBar.setValue(20)

        train_sizesrl, train_scoresrl, test_scoresrl = learning_curve(estimator=self.ppnrl, X=self.xLrl, y=self.yLrl,cv=10, train_sizes=np.linspace(0.1, 1.0, 10))

        train_meanrl = np.mean(train_scoresrl, axis=1)
        train_stdrl = np.std(train_scoresrl, axis=1)
        test_meanrl = np.mean(test_scoresrl, axis=1)
        test_stdrl = np.std(test_scoresrl, axis=1)

        self.progressBar.setValue(40)

        train_sizesrt, train_scoresrt, test_scoresrt = learning_curve(estimator=self.ppnrt, X=self.xLrt, y=self.yLrt,cv=10, train_sizes=np.linspace(0.1, 1.0, 10))

        train_meanrt = np.mean(train_scoresrt, axis=1)
        train_stdrt = np.std(train_scoresrt, axis=1)
        test_meanrt = np.mean(test_scoresrt, axis=1)
        test_stdrt = np.std(test_scoresrt, axis=1)

        self.progressBar.setValue(60)

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

        self.progressBar.setValue(80)

        plt.figure(2)
        plt.plot(train_sizesrt, train_meanrt, color='blue', marker='o', markersize=5, label='Training Accuracy')
        plt.fill_between(train_sizesrt, train_meanrt + train_stdrt, train_meanrt - train_stdrt, alpha=0.15, color='blue')
        plt.plot(train_sizesrt, test_meanrt, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
        plt.fill_between(train_sizesrt, test_meanrt + test_stdrt, test_meanrt - test_stdrt, alpha=0.15, color='green')
        plt.title('Learning Curve')
        plt.xlabel('Training Data Size')
        plt.ylabel('Model accuracy')
        plt.grid()
        plt.legend(loc='lower right')
        plt.show()

        self.progressBar.setValue(100)
        self.progressBar.hide()

    def TestNet(self):
        self.progressBar.show()
        self.progressBar.setValue(0)

        self.yPrl = self.ppnrl.predict(self.xTrl)
        self.yPrt = self.ppnrt.predict(self.xTrt)

        self.progressBar.setValue(50)

        accPrl = metrics.accuracy_score(self.yTrl, self.yPrl) * 100
        accNrl = 100.0 - accPrl

        accPrt = metrics.accuracy_score(self.yTrt, self.yPrt) * 100
        accNrt = 100.0 - accPrt

        # self.label_14.setText("{0:.2f}%".format(accN))
        # self.label_15.setText("{0:.2f}%".format(accP))

        plt.figure(1)
        fprrl, tprrl, thresholdsrl = metrics.roc_curve(self.yTrl, self.yPrl)
        roc_aucrl = metrics.auc(fprrl, tprrl)
        displayrl = metrics.RocCurveDisplay(fpr=fprrl, tpr=tprrl, roc_auc=roc_aucrl, estimator_name='example estimator')
        displayrl.plot()
        plt.show()

        plt.figure(2)
        fprrt, tprrt, thresholdsrt = metrics.roc_curve(self.yTrt, self.yPrt)
        roc_aucrt = metrics.auc(fprrt, tprrt)
        displayrt = metrics.RocCurveDisplay(fpr=fprrt, tpr=tprrt, roc_auc=roc_aucrt, estimator_name='example estimator')
        displayrt.plot()
        plt.show()

        self.progressBar.setValue(50)
        self.progressBar.hide()



    def clickBatton(self):
        self.pushButton.clicked.connect(lambda: self.CreateDataSet())

        self.pushButton_3.clicked.connect(lambda: self.TestNet())
        
        self.pushButton_2.clicked.connect(lambda: self.LoadNet())

    def retranslateUi(self, Migration):
        _translate = QtCore.QCoreApplication.translate
        Migration.setWindowTitle(_translate("Migration", "Migration"))
        self.pushButton.setText(_translate("Migration", "Сгенерировать выборки"))
        self.pushButton_2.setText(_translate("Migration", "Обучить нейронную сеть"))
        self.pushButton_3.setText(_translate("Migration", "Протестировать нейронную сеть"))
        self.label.setText(_translate("Migration", "Исходная выборка для Л-К аппроксимации"))
        self.label_2.setText(_translate("Migration", "Исходная выборка для гиперб. аппроксимации"))
        self.label_3.setText(_translate("Migration", "Адпатированная Л-К выборка под нейронную сеть"))
        self.label_4.setText(_translate("Migration", "Адпатированная гиперб. выборка под нейронную сеть"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Migration()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
