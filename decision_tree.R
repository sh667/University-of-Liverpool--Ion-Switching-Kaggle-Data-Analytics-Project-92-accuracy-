install.packages("rpart.plot")
require(caTools)
library(rpart)
library(rpart.plot)
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
sample = sample.split(data$open_channels, SplitRatio = 0.85)
train = subset(data, sample == TRUE)
test  = subset(data, sample == FALSE)
dim(train)
dim(test)
fit <- rpart(open_channels~., data = test, method = 'class')
predictions_train = predict(fit, newdata=train[-3])
predictions_test = predict(fit, newdata=test[-3])
accuracy_train <- mean((colnames(predictions_train)[apply(predictions_train,1,which.max)]) == train$open_channels)
accuracy_train
accuracy_test <- mean((colnames(predictions_test)[apply(predictions_test,1,which.max)]) == test$open_channels)
accuracy_test
test_data1 <- read.csv("test.csv",header=TRUE)
predictions_test_data  <- predict(fit, newdata=test_data1)
test_data$open_channels <- (colnames(predictions_test_data)[apply(predictions_test_data,1,which.max)])
str(test_data)
test_data$signal <- NULL
write.table(test_data, file = 'dt_predicted_test_final.csv', sep=",",row.names=FALSE)