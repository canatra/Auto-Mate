
# 0 means calm.
# 1 means angry

#resoce : https://www.youtube.com/watch?v=tU3Adlru1Ng
#prepare the data
data1 <- angry_h_1_COM
str(data1)
summary(data1)
data1$X8F <- factor(data1$X8)

#partition the data into training and validation datasets
set.seed(1234)
pd <- sample(2,nrow(data1),replace = TRUE, prob = c(0.75,0.25))
train <- data1[pd==1,]
validate <-data1[pd==2,]

#decision tree with party
library(party)
myfor <- X8F ~ `Max & Avg`
tree <- ctree(myfor, data = train)
#table for prediction
table(predict(tree, validate))
plot(tree)
tree
#prob. prediction
predict(tree, validate, type= "prob")


#using different R library
library(rpart)
library(rpart.plot)
tree1 <- rpart(myfor,train)
rpart.plot(tree1)

#prediction
predict(tree1, validate)

#Misclassification error for "tarin" data
#we will use a tree model to calculate isclassification error for "tarin" data
tab <- table(predict(tree),train$X8F)
print(tab)
#Misclassification error
1-sum(diag(tab))/sum(tab)

#Misclassification error for "validation" data
testPred <- predict(tree,newdata=validate)
tab<- table(testPred,validate$X8F)
print(tab)
1-sum(diag(tab))/sum(tab)



