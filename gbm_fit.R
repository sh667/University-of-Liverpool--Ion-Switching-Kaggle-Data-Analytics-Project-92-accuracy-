library(rsample)      # data splitting 
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # a java-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization
require(caTools)
data <- read.csv("train.csv",header=TRUE)
test_data <- read.csv("test.csv",header=TRUE,colClasses = c("character", "character"))
shuffle_index <- sample(1:nrow(data))
data <- data[shuffle_index, ]
head(data)
summary(data)
data <- transform(data, open_channels =as.factor(open_channels))
sapply(data, class)
data[ data == "?"] <- NA
colSums(is.na(data))
data <- data[!(data$open_channels %in% c(NA)),]
colSums(is.na(data))
sample = sample.split(data$open_channels, SplitRatio = .85)
train = subset(data, sample == TRUE)
test  = subset(data, sample == FALSE)
dim(train)
dim(test)
# train GBM model
gbm.fit <- gbm(formula = open_channels ~ .,distribution = "gaussian",data = train,n.trees = 10000,interaction.depth = 1,shrinkage = 0.001,cv.folds = 5,n.cores = NULL,verbose = FALSE) 
predictions_train = predict(gbm.fit, newdata=train[-3])
predictions_test = predict(gbm.fit, newdata=test[-3])
accuracy_train <- mean(predictions_train == train$open_channels)
accuracy_train
accuracy_test <- mean(predictions_test == test$open_channels)
accuracy_test
predictions_test_data  <- predict(gbm.fit, newdata=test_data)
test_data$open_channels <- predictions_test_data
str(test_data)
test_data$signal <- NULL
write.table(test_data, file = 'gbmfit_predicted_test_final.csv', sep=",",row.names=FALSE)
