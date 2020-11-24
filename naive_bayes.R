library(e1071)
library(caret)
library(caTools)
data<-read.csv(file="train.csv",header = TRUE)
test_data <- read.csv("test.csv",header=TRUE,colClasses = c("character", "character"))
shuffle_index <- sample(1:nrow(data))
data <- data[shuffle_index, ]
data<-transform(data,open_channels=as.factor(open_channels))
sapply(data,class)
data[data=="?"]<-NA
colSums(is.na(data))
split_values<-sample.split(data$open_channels,SplitRatio=0.85)
train<-subset(data,split_values==TRUE)
test<-subset(data,split_values==FALSE)
nb<-naiveBayes(open_channels~.,train)
predictions_train = predict(nb, newdata=train[-3])
predictions_test = predict(nb, newdata=test[-3])
accuracy_train <- mean(predictions_train == train$open_channels)
accuracy_train
accuracy_test <- mean(predictions_test == test$open_channels)
accuracy_test
predictions_test_data  <- predict(nb, newdata=test_data)
test_data$open_channels <- predictions_test_data
str(test_data)
test_data$signal <- NULL
write.table(test_data, file = 'nb_predicted_test_final.csv', sep=",",row.names=FALSE)


