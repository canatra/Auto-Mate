import csv
import math
from sklearn import tree
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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



def decisionTree(trainingset,testset):

	X =[]
	Y = []
	for row in trainingset:
		X.append(row[:-1])
		Y.append(row[-1])


	clf = tree.DecisionTreeClassifier()
	clf =  clf.fit(X, Y)
	X_test =[]
	Y_test = []
	for row in testset:
		X_test.append(row[:-1])
		Y_test.append(row[-1])

	pred = clf.predict(X_test)
	print(pred)
	print(Y_test)
	cnf_matrix = confusion_matrix(Y_test, pred)
	np.set_printoptions(precision=2)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=('Aggressive', 'Calm'), title='Confusion matrix, without normalize')

	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=('Aggressive', 'Calm'), normalize=True ,title='Confusion matrix, with normalize')
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
		if time >= 30:
			size = len(curr)
			curr.append(1)
			trainingset.append(curr)
			break
				
			previous = curr
			prevtime = float(row[0])
			timechange += float(row[0])

#	print(size)
#	print(len(curr))
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
print(len(trainingset))
print(len(testset))
decisionTree(trainingset,testset)