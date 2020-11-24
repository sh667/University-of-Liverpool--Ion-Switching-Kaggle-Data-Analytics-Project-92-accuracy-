library(randomForest)
require(caTools)
data <- read.csv("train_clean.csv",header=TRUE)
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
sample = sample.split(data$open_channels, SplitRatio = .75)
train = subset(data, sample == TRUE)
test  = subset(data, sample == FALSE)
dim(train)
dim(test)
rf <- randomForest(open_channels ~., data=train)
predictions_train = predict(rf, newdata=train[-3])
predictions_test = predict(rf, newdata=test[-3])
accuracy_train <- mean(predictions_train == train$open_channels)
accuracy_train
accuracy_test <- mean(predictions_test == test$open_channels)
accuracy_test
predictions_test_data  <- predict(rf, newdata=test_data)
test_data$open_channels <- predictions_test_data
str(test_data)
test_data$signal <- NULL
write.table(test_data, file = 'rf_predicted_test_final.csv', sep=",",row.names=FALSE)
