---
title: "Practical Machine Learning Project"
author: "John Doe"
date: "October 26, 2014"
output: html_document
---
This project aims to create an algorithm to analyze whether data from devices with accelerometers can correctly predict the exact movement the person wearing is doing. 

## Background: Are they performing barbell lifts correctly?
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

## Data
### Getting Data
The data for this project come from [here](http://groupware.les.inf.puc-rio.br/har). The dataset used in training and cross-validation is stored in `totalRaw`, and the other small dataset used to predict values submitted to Coursera is stored in `testRaw`. When loading data, any empty values or those with "NA" or "#DIV/0!" would be treated as `NA`. We have 19622 observations of 160 features in `totalRaw`.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(ggplot2)
library(lattice)
library(kernlab)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
if (!file.exists("pml-training.csv")) {
  download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv")
}
if (!file.exists("pml-testing.csv")) {
  download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv")
}

totalRaw <- read.csv("pml-training.csv", sep = ",", na.strings = c("", "NA", "#DIV/0!"))
testRaw <- read.csv("pml-testing.csv", sep = ",", na.strings = c("", "NA", "#DIV/0!"))
```

## Data Cleaning
First exclude all those columns with `NA` values, as those insignificant features would interfere with algorithm's accuracy and efficiency. Doing so would eliminate 100 variables. 
Then we remove columns corresponding to descriptive contents, such as `user_name`, and anything related to `timestamp`. These are stored in the first seven columns in the data set.


```r
totalRaw <- totalRaw[,colSums(is.na(totalRaw))==0]
totalRaw <- totalRaw[,-1:-7]
```

## Data Partition
Next, we partition data to have 70% into training and 30% into testing in order to have good cross-validation.


```r
inTrain <- createDataPartition(y=totalRaw$classe, p = 0.7, list = FALSE)
training <- totalRaw[inTrain,]
testing <- totalRaw[-inTrain,]
```

# Model Building
We utilize `randomForest` model to build the classification tree, since it has highest accuracy with minimum overfitting issues. During training, cross-validation (`cv`) has been chosen to control the process.


```r
# Register for parallel computing
library(doParallel);
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
rCluster <- makePSOCKcluster(4);
registerDoParallel(rCluster);

# Command to train machine
trControl = trainControl(method = "cv", number = 4, allowParallel =TRUE);
rffit <- train(training$classe ~.,data = training,method="rf",trControl=trControl);
```

# Cross-validation
Now we want to learn the error of in-sample and out-of-sample. We achieve such goal by drawing confusion matrix.

First is in-sample, i.e. by seeing how well the algorithm is predicting training set itself.

```r
prediction <- predict(rffit, training)
confusionMatrix(prediction, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

Then is out-of-sample error, i.e. by seeing how well it can predict the small test part:

```r
prediction <- predict(rffit, testing)
confusionMatrix(prediction, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    5    0    0    0
##          B    1 1134   13    0    0
##          C    0    0 1011   21    3
##          D    0    0    2  942    4
##          E    0    0    0    1 1075
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9915          
##                  95% CI : (0.9888, 0.9937)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9893          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9956   0.9854   0.9772   0.9935
## Specificity            0.9988   0.9971   0.9951   0.9988   0.9998
## Pos Pred Value         0.9970   0.9878   0.9768   0.9937   0.9991
## Neg Pred Value         0.9998   0.9989   0.9969   0.9955   0.9985
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1927   0.1718   0.1601   0.1827
## Detection Prevalence   0.2851   0.1951   0.1759   0.1611   0.1828
## Balanced Accuracy      0.9991   0.9963   0.9902   0.9880   0.9967
```
We can see from both confusion matrices, we have pretty high accuracy for both cases.

# How does it go with the testing 20 samples?

```r
answer <- predict(rffit, testRaw)

answer
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
answer <- as.vector(answer)

pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

pml_write_files(answer)
```

