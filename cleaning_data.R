library(data.table)
library(dplyr)
data <- read.csv("../input/liverpool-ion-switching/train.csv",header=TRUE)
test_data <- read.csv("../input/liverpool-ion-switching/test.csv",header=TRUE)
train2 = copy(data)
a<-500000
b<-600000 
train2[a:b,2] <-  train2[a:b, 2] - (3*(train2$time[a:b] - 50)/10)
f <- function(x,low,high,mid){
  return(-((-low+high)/625)*(x-mid)**2+high -low)
}

# CLEAN TRAIN BATCH 7
batch <- 7 
a <- 500000*(batch-1) 
b <- 500000*batch
train2[a:b,2] <-  train2[a:b, 2] - f(data$time[a:b], -1.817,3.186,325)
# CLEAN TRAIN BATCH 8
batch <- 8
a <- 500000*(batch-1) 
b <- 500000*batch
train2[a:b,2] <-  train2[a:b, 2] - f(data$time[a:b],-0.094,4.936,375)
# CLEAN TRAIN BATCH 9
batch = 9;
a <- 500000*(batch-1) 
b <- 500000*batch
train2[a:b,2] <-  train2[a:b, 2] - f(data$time[a:b],1.715,6.689,425)
# CLEAN TRAIN BATCH 10
batch = 10; 
a <- 500000*(batch-1) 
b <- 500000*batch
train2[a:b,2] <-  train2[a:b, 2] - f(data$time[a:b],3.361,8.45,475)
# Training batch 1 and 2(1 Slow Open Channel)
batch <- 1
a <- 500000*(batch-1) 
b <- 500000*batch
batch <- 2
c <- 500000*(batch-1) 
d <- 500000*batch
abc <- c(train2$signal[a:b],train2$signal[c:d])
X_train <- c()
for (i in 1:length(abc)){
  X_train[i] <- list(abc[i])
}
xyz <- c()
y_train <- c(train2$open_channels[a:b],train2$open_channels[c:d])
for (i in 1:length(abc)){
  Y_train[i] <- list(xyz[i])
}
