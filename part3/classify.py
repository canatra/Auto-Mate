import csv
import math
import os
from sklearn import tree,svm, datasets
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

def precision_recall(true, pred, title="2-class Precision-Recall curve"):
	precision,recall,thresholds = precision_recall_curve(true,pred)
	avgprec = average_precision_score(true,pred)
	print("Precision= "+str(precision))
	print("Recall= "+str(recall))
	print("thresholds= " +str(thresholds))
	print("average_precision_score= "+str(avgprec))
	plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
	plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title(title+' AP={0:0.2f}'.format(
          avgprec))


def plot_confusion_matrix (cm, classes, normalize=False, title ='Confusion matrix', cmap=plt.cm.Blues):

		if normalize:
			cm = cm.astype('float')/cm.sum(axis=1)[:np.newaxis]

		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks=np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)
		fmt ='.2f' if normalize else 'd'
		thresh = cm.max()/2

		for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
			plt.text(j,i,format(cm[i,j],fmt),
				horizontalalignment="center",
				color="white" if cm[i,j] > thresh else "black")
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')


def split(Tset):
	X =[]
	Y = []
	for row in Tset:
		X.append(row[:-1])
		Y.append(row[-1])
	return X, Y


def decisionTree(X,Y, X_test, Y_test):
	print("Decision Tree Classifier")
	clf = tree.DecisionTreeClassifier()
	clf =  clf.fit(X, Y)
	pred = clf.predict(X_test)
	precision_recall(Y_test, pred,"DecisionTreeClassifier Precision-Recall")	
	print(pred)
	print(Y_test)
	cnf_matrix = confusion_matrix(Y_test, pred)
	np.set_printoptions(precision=2)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=('Calm','Aggressive'), title='Decision Tree Confusion matrix')

	
	plt.show()


def svm_model(X, Y, X_test, Y_test):
	clf= svm.SVC()
	clf.fit(X,Y)
	pred = clf.predict(X_test)
	print("SVM prediction")
	print(pred)
	print(Y_test)
	precision_recall(Y_test, pred, "SVM Precision-Recall")
	cnf_matrix = confusion_matrix(Y_test, pred)
	np.set_printoptions(precision=2)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=('Calm','Aggressive'), title='SVM Confusion matrix')

	plt.show()

def MLP_model(X, Y, X_test, Y_test):
	print("MLPClassifier ")
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                   	hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(X,Y)
	pred = clf.predict(X_test)
	precision_recall(Y_test, pred, "MLPClassifier Precision-Recall")	
	print(pred)
	print(Y_test)
	cnf_matrix = confusion_matrix(Y_test, pred)
	np.set_printoptions(precision=2)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=('Calm','Aggressive'), title='MLP Confusion matrix')

	
	plt.show()

					




def newList(filename, curr,size):

	#1
	#print('AccelerationExplorer-angry3.csv')
	with open(filename) as csvfile:
		reader = csv.reader(csvfile)
		prevx = 0.0
		prevy = 0.0
		count = 0
		time=0.0
		for row in reader:

			#acceleration = change in velocity/change in time
			#acceleration * change in time = change in velocity
			if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
				count+= 1
				if 'time_tick' not in row[0]:
					time = float(row[0])
					curr.append(float(row[1]) - prevx)
					curr.append(float(row[2]) - prevy)
					prevx = float(row[1])
					prevy = float(row[2])
					previous = curr
					prevtime = float(row[0])
					
					if len(curr) == size or time >= 60:
						return(len(curr))
					
					
def newMicroList(filename, curr,size):

	#1
	#print('AccelerationExplorer-angry3.csv')
	with open(filename) as csvfile:
		reader = csv.reader(csvfile)
		prevx = 0.0
		prevy = 0.0
		count = 0
		time=0.0
		for row in reader:

			if 'Max' not in row[3]: 
				count+= 1
				curr.append(float(row[3]))
				curr.append(float(row[4]))
				time =row[2].split(':')[2].split('PM')[0]
					#prevx = float(row[1])
					#prevy = float(row[2])
					#previous = curr
					#prevtime = float(row[0])
					
			if len(curr) == size or float(time) + 1 >= 60:
				return(len(curr))
					
					



trainingset  =[]
curr =[]
size = 0.0

print('calm5-H')
test1 = newList('calm5-H.csv', curr,15000)
test2 = newList('calm5-Hgyro.csv', curr,15000)
test3 =newMicroList('calm5-Hmicro.csv', curr,150000)		
curr.append(1)	
trainingset.append(curr)
minsize =test3
minfile ='calm5-H'

curr =[]


for root, dirs, filenames in os.walk('.'):
	for filename in filenames:
		if ".csv" in filename and 'calm5-H' not in filename and 'gyro' not in filename and 'micro' not in filename:
			print(filename)
			newList(filename, curr,test1)
			filename = filename.split('.csv')[0] +'gyro.csv'
			print(filename)
			newList(filename, curr, test2)
			filename = filename.split('gy')[0]+'micro.csv'
			print(filename)
			size = newMicroList(filename, curr, test3)
		
			if 'calm' in filename:
				curr.append(0)
			else:		
				curr.append(1)
			if int(size) < minsize:
				minsize = size
				minfile = filename
			if size == test3:
				trainingset.append(curr)		
				
			curr =[]


print(minsize)
print(minfile)
print(len(trainingset))

#split data into traing set & validation set
trainingset, testset = train_test_split(trainingset)
#split class from other features
X,Y = split(trainingset)
#print(X[0])
#print(Y[0])
X_test, Y_test = split(testset)
print(len(trainingset))
print(len(testset))
#calling models: decision tree, svm, and MLP (multilayer perceptron)
decisionTree(X,Y, X_test, Y_test)
svm_model(X, Y, X_test, Y_test)
MLP_model(X, Y, X_test, Y_test)
