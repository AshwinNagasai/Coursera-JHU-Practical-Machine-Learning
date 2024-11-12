---
title: "Practical Machine Learning Project"
author: "Ashwin Sai Murali Neelakandan"
date: "11/11/2024"
output: 
  html_document: 
    keep_md: true
---



## Introduction  

The goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways, to predict the manner in which the participants did the exercise.  

## Data  
  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively.   

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: 
http://groupware.les.inf.puc-rio.br/har
  
### Loading the necessary packages  

``` r
library(caret)
```

```
## Warning: package 'caret' was built under R version 4.4.2
```

```
## Loading required package: ggplot2
```

```
## Loading required package: lattice
```

``` r
library(lattice)
library(ggplot2)
library(kernlab)
```

```
## 
## Attaching package: 'kernlab'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     alpha
```

``` r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 4.4.2
```

```
## randomForest 4.7-1.2
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

``` r
library(corrplot)
```

```
## Warning: package 'corrplot' was built under R version 4.4.2
```

```
## corrplot 0.95 loaded
```

``` r
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 4.4.2
```

```
## Loading required package: rpart
```

### Loading and Data Preprocessing  

``` r
raw_train <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
raw_test <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
# Looking at the dimensions of the training and test data  
dim(raw_train)
```

```
## [1] 19622   160
```

``` r
dim(raw_test)
```

```
## [1]  20 160
```

Seeing if the data has any NA values and then replacing most of them with 0

``` r
# Seeing which cases have no missing values
sum(complete.cases(raw_train))
```

```
## [1] 406
```
  
  
Selecting and replacing the NA values in both training and test sets  

``` r
raw_train <- raw_train[, colMeans(is.na(raw_train)) < .9]
test_data <- raw_test[, colMeans(is.na(raw_test))<.9]
# Removing columns which are irrelevant to the outcome from the raw training data
raw_train <- raw_train[, -c(1:7)]
```

Removing near zero variance variables from the raw training set

``` r
# For training set
nearzvar1 <- nearZeroVar(raw_train)
raw_train <- raw_train[, -nearzvar1]
# Looking at both training and testing set dimensions after pre-processing
dim(raw_train)
```

```
## [1] 19622    53
```

### Splitting the Training dataset  


``` r
# Setting a seed for reproducibility
set.seed(6583)
# Partitioning  the cleaned dataset into training and validation data sets
inData <- createDataPartition(y=raw_train$classe, p=0.7, list = FALSE)
Traindata <- raw_train[inData,]
Validdata <- raw_train[-inData,]
```
The cleaned data was split into a training set (70%) and a validation set (30%) which will be used for cross validation purposes.  


``` r
# Converting the classe variable into a factor variable
Traindata$classe <- as.factor(Traindata$classe)
Validdata$classe <- as.factor(Validdata$classe)
# Setting up a control to use 5-fold cross validation for Decision Trees, Random Forests, and Support Vector Machine models
crossvalid_control <- trainControl(method="cv", number=5, verboseIter=FALSE)
```

## Model Building 

Trying fit the prediction models based on a few popular models:  
i. Decision Trees  
ii. Random Forests    
iii. Support Vector Machines   
iv. Generalized Boosting   
  
### Decision Trees  


``` r
# Creating a Decision tree prediction model the rpart method
Tree_mod <- train(classe ~ ., data = Traindata, method = "rpart", 
                  trControl = crossvalid_control)
# Applying the model to the validation set for the prediction
Tree_pred <- predict(Tree_mod, Validdata)
Tree_cfm <- confusionMatrix(Tree_pred, Validdata$classe)
Tree_cfm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1511  459  484  449  148
##          B   18  296   15  161   66
##          C  141  384  527  354  387
##          D    0    0    0    0    0
##          E    4    0    0    0  481
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4783          
##                  95% CI : (0.4655, 0.4912)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.319           
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9026  0.25988  0.51365   0.0000  0.44455
## Specificity            0.6343  0.94522  0.73945   1.0000  0.99917
## Pos Pred Value         0.4952  0.53237  0.29392      NaN  0.99175
## Neg Pred Value         0.9425  0.84181  0.87805   0.8362  0.88870
## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
## Detection Rate         0.2568  0.05030  0.08955   0.0000  0.08173
## Detection Prevalence   0.5184  0.09448  0.30467   0.0000  0.08241
## Balanced Accuracy      0.7685  0.60255  0.62655   0.5000  0.72186
```

``` r
Tree_accuracy <- Tree_cfm$overall[1]
Tr_outsamperror <- 1 - Tree_accuracy
```

The accuracy obtained for the Decision trees model is 0.4783347 and the out of sample error rate is 0.5216653   

### Random Forests


``` r
# Creating a Random Forests prediction model with 3-fold cross validation
RF_mod <- train(classe~., Traindata, method = "rf", 
                trControl = crossvalid_control)
# Applying the model to the validation set for the prediction
RF_pred <- predict(RF_mod, Validdata)
RF_cfm <- confusionMatrix(RF_pred, Validdata$classe)
RF_cfm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    2    0    0    0
##          B    2 1135   11    0    0
##          C    1    2 1013   18    0
##          D    0    0    2  946    2
##          E    0    0    0    0 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9932          
##                  95% CI : (0.9908, 0.9951)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9914          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9965   0.9873   0.9813   0.9982
## Specificity            0.9995   0.9973   0.9957   0.9992   1.0000
## Pos Pred Value         0.9988   0.9887   0.9797   0.9958   1.0000
## Neg Pred Value         0.9993   0.9992   0.9973   0.9964   0.9996
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1929   0.1721   0.1607   0.1835
## Detection Prevalence   0.2843   0.1951   0.1757   0.1614   0.1835
## Balanced Accuracy      0.9989   0.9969   0.9915   0.9903   0.9991
```

``` r
RF_accuracy <- RF_cfm$overall[1]
RF_outsamperror <- 1 - RF_accuracy
```
The accuracy obtained for the Random Forests model is0.9932031 and the out of sample error rate is 0.0067969  
  
### Support Vector Machine  


``` r
# Creating a Support Vector Machine prediction model with 3-fold cross validation
SVM_mod <- train(classe~., Traindata, method = "svmLinear",
                 trControl = crossvalid_control)
# Applying the model to the validation set for the prediction
SVM_pred <- predict(SVM_mod, Validdata)
SVM_cfm <- confusionMatrix(SVM_pred, Validdata$classe)
SVM_cfm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1553  152   80   66   55
##          B   26  811   80   38  134
##          C   48   74  811  107   74
##          D   42   24   31  702   59
##          E    5   78   24   51  760
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7879          
##                  95% CI : (0.7773, 0.7983)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7304          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9277   0.7120   0.7904   0.7282   0.7024
## Specificity            0.9162   0.9414   0.9376   0.9683   0.9671
## Pos Pred Value         0.8148   0.7447   0.7280   0.8182   0.8279
## Neg Pred Value         0.9696   0.9316   0.9549   0.9479   0.9352
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2639   0.1378   0.1378   0.1193   0.1291
## Detection Prevalence   0.3239   0.1850   0.1893   0.1458   0.1560
## Balanced Accuracy      0.9219   0.8267   0.8640   0.8483   0.8348
```

``` r
SVM_accuracy <- SVM_cfm$overall[1]
SVM_outsamperror <- 1 - SVM_accuracy
```
The accuracy obtained for the Support Vector Machine model is0.7879354 and the out of sample error rate is 0.2120646  
    
### Generalized Boosting  


``` r
# Setting up a control to use repeated 5-fold cross validation for the Generalized Boosting model
Gbm_Control <- trainControl(method = "repeatedcv", number = 5, verboseIter = FALSE)
# Creating a Generalized Boosting prediction model with 3-fold cross validation
Gbm_mod <- train(classe~., Traindata, method = "gbm", 
                 trControl = Gbm_Control, 
                 verbose = FALSE)
# Applying the model to the validation set for the prediction
Gbm_pred <- predict(Gbm_mod, Validdata)
Gbm_cfm <- confusionMatrix(Gbm_pred, Validdata$classe)
Gbm_cfm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1641   27    0    1    1
##          B   18 1071   30    4   13
##          C   10   41  983   41   12
##          D    5    0   12  908   13
##          E    0    0    1   10 1043
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9594         
##                  95% CI : (0.954, 0.9643)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9486         
##                                          
##  Mcnemar's Test P-Value : 4.063e-09      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9803   0.9403   0.9581   0.9419   0.9640
## Specificity            0.9931   0.9863   0.9786   0.9939   0.9977
## Pos Pred Value         0.9826   0.9428   0.9043   0.9680   0.9896
## Neg Pred Value         0.9922   0.9857   0.9910   0.9887   0.9919
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2788   0.1820   0.1670   0.1543   0.1772
## Detection Prevalence   0.2838   0.1930   0.1847   0.1594   0.1791
## Balanced Accuracy      0.9867   0.9633   0.9683   0.9679   0.9808
```

``` r
Gbm_accuracy <- Gbm_cfm$overall[1]
Gbm_outsamperror <- 1 - Gbm_accuracy
```
The accuracy obtained for the Gradient Boosted Trees model is0.9593883 and the out of sample error rate is 0.0406117 

### Selecting the optimal model  

Creating a table with Accuracy and out of sample error rates for all the above models  


``` r
model_names = c("Tree", "RandomForests", "SupportVectorMachine", "GeneralizedBoosting")
Accuracy <- round(c(Tree_accuracy, RF_accuracy, SVM_accuracy, Gbm_accuracy), 3)
Out_of_Sample_Error <- 1-Accuracy
data.frame(Accuracy=Accuracy, Out_of_Sample_Error_Rate = Out_of_Sample_Error, row.names = model_names)
```

```
##                      Accuracy Out_of_Sample_Error_Rate
## Tree                    0.478                    0.522
## RandomForests           0.993                    0.007
## SupportVectorMachine    0.788                    0.212
## GeneralizedBoosting     0.959                    0.041
```

Based on the results observed, the optimal model is the Random Forests model with accuracy of 0.9932031 and the out of sample error rate of 0.0067969 . The Gradient Tree with Boosting model comes second best with accuracy of 0.9593883 and the out of sample error rate of 0.0406117. 
  
Based on the results observed, the Random Forests model is selected as the optimal model.  

##  Predicting on the Test Set  


``` r
Pred_Results <- predict(RF_mod, test_data)
Pred_Results
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
  
## Appendix: Plots  
  
Correlation Matrix of variables in the training set  


``` r
corr_matrix <- cor(Traindata[, -length(names(Traindata))])
corrplot(corr_matrix, method = "circle")
```

![](Practical-Machine-Learning-Project-Report_files/figure-html/unnamed-chunk-14-1.png)<!-- -->

Plotting the different models  

``` r
# Plotting Decision Tree model
plot(Tree_mod)
```

![](Practical-Machine-Learning-Project-Report_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

``` r
# Visualizing the Decision Tree model
rpart.plot(Tree_mod$finalModel, main = "Decision Tree Model")
```

![](Practical-Machine-Learning-Project-Report_files/figure-html/unnamed-chunk-15-2.png)<!-- -->

``` r
# Plotting Random Forests model
plot(RF_mod)
```

![](Practical-Machine-Learning-Project-Report_files/figure-html/unnamed-chunk-15-3.png)<!-- -->

``` r
# Plotting the Generalized Boosting Model
plot(Gbm_mod)
```

![](Practical-Machine-Learning-Project-Report_files/figure-html/unnamed-chunk-15-4.png)<!-- -->


