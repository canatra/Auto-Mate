import csv
import math
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


trainingset  =[]

size = 0.0
#1
#print('AccelerationExplorer-angry3.csv')
with open('AccelerationExplorer-angry3.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx = 0.0
	prevy = 0.0
	count = 0
	time=0.0
	for row in calmreader:

		#acceleration = change in velocity/change in time
		#acceleration * change in time = change in velocity
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			count+= 1
			time = float(row[0])
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) - prevy)
			prevx = float(row[1])
			prevy = float(row[2])
		if time >= 48:
			size = len(curr)
			curr.append(1)
			trainingset.append(curr)
			break
				
			previous = curr
			prevtime = float(row[0])
			timechange += float(row[0])

#	print(size)
	print(len(curr))
#2
#print('AccelerationExplorer-calm1.csv')
with open('AccelerationExplorer-calm1.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			
			if len(curr) == size:
				curr.append(0)
				trainingset.append(curr)
				break	
	#print(len(curr))

#3
#print('AccelerationExplorer-calm2.csv')
with open('AccelerationExplorer-calm2.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			time = float(row[0])
			if len(curr) == size:
				curr.append(0)
				trainingset.append(curr)
				break	
				
	#print(len(curr))		
#4
#print('AccelerationExplorer-H-calm1.csv')
with open('AccelerationExplorer-H-calm1.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			time = float(row[0])
			if len(curr) == size:
				curr.append(0)
				trainingset.append(curr)
				break	
	#print(len(curr))
#print('AccelerationExplorer-H-calm2.csv')
#5
with open('AccelerationExplorer-H-calm2.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			time = float(row[0])
			if len(curr) == size:
				curr.append(0)
				trainingset.append(curr)
				break	
	#print(len(curr))
#6

#print('AccelerationExplorer-L-calm3.csv')
with open('AccelerationExplorer-L-calm3.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			time = float(row[0])
			if len(curr) == size:
				curr.append(0)
				trainingset.append(curr)
				break	
	#print(len(curr))
#7
#print('AccelerationExplorer-L-H-calm3.csv')
with open('AccelerationExplorer-L-H-calm3.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			time = float(row[0])
			if len(curr) == size:
				#print("too short")
				curr.append(0)
				trainingset.append(curr)
				break	
	#print(len(curr))
#8
#print('AccelerationExplorer-L-calm4.csv')
with open('AccelerationExplorer-L-calm4.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			time = float(row[0])
			if len(curr) == size:
				curr.append(0)
				trainingset.append(curr)
				break	
	#print(len(curr))
#9
#print('AccelerationExplorer-L-H-calm4.csv')
with open('AccelerationExplorer-L-H-calm4.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			time = float(row[0])
			if len(curr) == size:
				curr.append(0)
				trainingset.append(curr)
				break	
	#print(len(curr))

#10

#print('AccelerationExplorer-H-angry3.csv')
with open('AccelerationExplorer-H-angry3.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			
			if len(curr) == size:
				
				curr.append(1)
				trainingset.append(curr)
				break	
	#print(len(curr))

#11
#print('AccelerationExplorer-H-angry4.csv')
with open('AccelerationExplorer-H-angry4.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			
			if len(curr) == size:
				curr.append(1)
				trainingset.append(curr)
				break	

	
	#print(len(curr))
	#12
#print('AccelerationExplorer-L-H-angry1.csv')
with open('AccelerationExplorer-L-H-angry1.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			
			if len(curr) == size:
				curr.append(1)
				trainingset.append(curr)
				break	
	#print(len(curr))

#13
#print('AccelerationExplorer-L-H-angry2.csv')
with open('AccelerationExplorer-L-H-angry2.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			
			if len(curr) == size:
				curr.append(1)
				trainingset.append(curr)
				break	
	#print(len(curr))
#14
#print('AccelerationExplorer-L-angry1.csv')
with open('AccelerationExplorer-L-angry1.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			
			if len(curr) == size:
				curr.append(1)
				trainingset.append(curr)
				break	
	#print(len(curr))
#15
#print('AccelerationExplorer-L-angry2.csv')
with open('AccelerationExplorer-L-angry2.csv') as csvfile:	
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			
			if len(curr) == size:
				curr.append(1)
				trainingset.append(curr)
				break	
	#print(len(curr))

#16
#print('AccelerationExplorer-angry4.csv')
with open('AccelerationExplorer-angry4.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			
			if len(curr) == size:
				curr.append(1)
				trainingset.append(curr)
				break	
	#print(len(curr))

#17
#print('AccelerationExplorer-LL-H-angry5.csv')
with open('AccelerationExplorer-LL-H-angry5.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			
			if len(curr) == size:
				curr.append(1)
				trainingset.append(curr)
				break	
	#print(len(curr))
#18
#print('AccelerationExplorer-LL-angry5.csv')
with open('AccelerationExplorer-LL-angry5.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			
			if len(curr) == size:
				curr.append(1)
				trainingset.append(curr)
				break	
	#print(len(curr))

#19
#print('AccelerationExplorer-LL-calm5.csv')
with open('AccelerationExplorer-LL-calm5.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			time = float(row[0])
			if len(curr) == size:
				curr.append(0)
				trainingset.append(curr)
				break	
	#print(len(curr))

#20
#print('AccelerationExplorer-LL-H-calm5.csv')
with open('AccelerationExplorer-LL-H-calm5.csv') as csvfile:
	calmreader = csv.reader(csvfile)
	curr = []
	prevx =0.0
	prevy= 0.0
	count = 0
	time = 0.0
	for row in calmreader:
		#print(row)
		if len(row) == 4 and row[1] != 'X' and row[1] != '' and row[2] != '' :
			curr.append(float(row[1]) - prevx)
			curr.append(float(row[2]) -prevy)		
			prevx =float(row[1])
			prevy =float(row[2])
			count+= 1
			time = float(row[0])
			if len(curr) == size:
				curr.append(0)
				trainingset.append(curr)
				break	
	#print(len(curr))
	


#print(len(trainingset))
trainingset, testset = train_test_split(trainingset)
X,Y = split(trainingset)
X_test, Y_test = split(testset)
print(len(trainingset))
print(len(testset))
decisionTree(X,Y, X_test, Y_test)
svm_model(X, Y, X_test, Y_test)
MLP_model(X, Y, X_test, Y_test)