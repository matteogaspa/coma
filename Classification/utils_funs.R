require(dplyr)
require(glmnet)
require(randomForest)
require(polspline)
require(e1071)
require(ggplot2)
require(mlr)
require(gridExtra)
require(mvtnorm)


# NNet -----
cp_class_nn <- function(train_data, test_point, alpha = 0.1, split = 0.5, target = "y", nclass = 8, seed = 1){
  n     <- NROW(train_data)
  set.seed(seed)
  indc  <- sample(1:2, n, replace = T, prob = c(split, 1-split))
  ntask <- makeClassifTask(data = train_data, target = target)
  
  # train in the proper training set 
  lrn.nn <- makeLearner("classif.nnet", predict.type = "prob")
  m1     <- mlr::train(lrn.nn, ntask, subset = which(indc == 1))
  
  # scores in the calibration set 
  p1 <- predict(m1, task = ntask, subset = which(indc == 2))$data
  ys <- p1$truth
  p1 <- p1[,3:(2+nclass)]
  
  scores <- rep(NA, sum(indc==2))
  for(i in 1:length(scores)){
    scores[i] <- p1[i,as.numeric(ys[i])]
  }
  
  # quantile of reference 
  n2     <- sum(indc==2)
  scores <- scores[order(scores)]
  qn2    <- scores[floor(alpha * (n2 - 1))]
  
  # output 
  p3 <- predict(m1, newdata = test_point)$data[,1:nclass]
  p3 <- I(p3 >= qn2)
  
  return(p3)
}

## Random Forest -----
cp_class_rf <- function(train_data, test_point, alpha = 0.1, split = 0.5, target = "y", nclass = 8, seed = 1){
  n     <- NROW(train_data)
  set.seed(seed)
  indc  <- sample(1:2, n, replace = T, prob = c(split, 1-split))
  ntask <- makeClassifTask(data = train_data, target = target)
  
  # train in the proper training set 
  lrn.rf <- makeLearner("classif.randomForest", predict.type = "prob")
  m1     <- mlr::train(lrn.rf, ntask, subset = which(indc == 1))
  
  # scores in the calibration set 
  p1 <- predict(m1, task = ntask, subset = which(indc == 2))$data
  ys <- p1$truth
  p1 <- p1[,3:(2+nclass)]
  
  scores <- rep(NA, sum(indc==2))
  for(i in 1:length(scores)){
    scores[i] <- p1[i,as.numeric(ys[i])]
  }
  
  # quantile of reference 
  n2     <- sum(indc==2)
  scores <- scores[order(scores)]
  qn2    <- scores[floor(alpha * (n2 - 1))]
  
  # output 
  p3 <- predict(m1, newdata = test_point)$data[,1:nclass]
  p3 <- I(p3 >= qn2)
  
  return(p3)
}

# QDA -----
cp_class_qda <- function(train_data, test_point, alpha = 0.1, split = 0.5, target = "y", nclass = 8, seed = 1){
  n     <- NROW(train_data)
  set.seed(seed)
  indc  <- sample(1:2, n, replace = T, prob = c(split, 1-split))
  ntask <- makeClassifTask(data = train_data, target = target)
  
  # train in the proper training set 
  lrn.qda <- makeLearner("classif.qda", predict.type = "prob")
  m1     <- mlr::train(lrn.qda, ntask, subset = which(indc == 1))
  
  # scores in the calibration set 
  p1 <- predict(m1, task = ntask, subset = which(indc == 2))$data
  ys <- p1$truth
  p1 <- p1[,3:(2+nclass)]
  
  scores <- rep(NA, sum(indc==2))
  for(i in 1:length(scores)){
    scores[i] <- p1[i,as.numeric(ys[i])]
  }
  
  # quantile of reference 
  n2     <- sum(indc==2)
  scores <- scores[order(scores)]
  qn2    <- scores[floor(alpha * (n2 - 1))]
  
  # output 
  p3 <- predict(m1, newdata = test_point)$data[,1:nclass]
  p3 <- I(p3 >= qn2)
  
  return(p3)
}

# LDA -----
cp_class_lda <- function(train_data, test_point, alpha = 0.1, split = 0.5, target = "y", nclass = 8, seed = 1){
  n     <- NROW(train_data)
  set.seed(seed)
  indc  <- sample(1:2, n, replace = T, prob = c(split, 1-split))
  ntask <- makeClassifTask(data = train_data, target = target)
  
  # train in the proper training set 
  lrn.lda <- makeLearner("classif.lda", predict.type = "prob")
  m1     <- mlr::train(lrn.lda, ntask, subset = which(indc == 1))
  
  # scores in the calibration set 
  p1 <- predict(m1, task = ntask, subset = which(indc == 2))$data
  ys <- p1$truth
  p1 <- p1[,3:(2+nclass)]
  
  scores <- rep(NA, sum(indc==2))
  for(i in 1:length(scores)){
    scores[i] <- p1[i,as.numeric(ys[i])]
  }
  
  # quantile of reference 
  n2     <- sum(indc==2)
  scores <- scores[order(scores)]
  qn2    <- scores[floor(alpha * (n2 - 1))]
  
  # output 
  p3 <- predict(m1, newdata = test_point)$data[,1:nclass]
  p3 <- I(p3 >= qn2)
  
  return(p3)
}

# Hedge Algorithm -----

# Given a matrix T x k of losses and a learning parameter returns the weigths 
# and the hedge loss over time

hedge <- function(l, eta){
  # l: matrix TxK containing the loss of the K experts during rounds
  # eta: learning parameter
  N <- nrow(l)
  K <- ncol(l)
  h <- rep(NA, N)
  L <- rep(0, K)
  
  weights <- matrix(NA, N, K)
  w <- rep(1/K, K)
  
  for(t in 1:N){
    weights[t,] <- w
    w <- w * exp(-eta * l[t,])
    w <- w / sum(w)
    h[t] <- sum(w * l[t,])
  }
  return(list(h=h, weights=weights))
}

# AdaHedge algorithm -----

# Given a matrix T x k of losses and returns the weigths and the hedge loss 
# over time of the adaHedge procedure

mix <- function(L, eta) {
  # eta: learning parm.
  # L: cumulative loss
  mn <- min(L)
  
  if (eta == Inf) {
    w <- as.numeric(L == mn)
  } else {
    w <- exp(-eta * (L - mn))
  }
  
  s <- sum(w)
  w <- w / s
  M <- mn - log(s / length(L)) / eta
  
  return(list(w = w, M = M))
}

adahedge <- function(l) {
  # l: matrix TxK containing the loss of the K experts during rounds
  N <- nrow(l)
  K <- ncol(l)
  h <- rep(NA, N)
  L <- rep(0, K)
  etas <- rep(NA, N)
  weights <- matrix(NA, N, K)
  Delta <- 0
  
  for (t in 1:N) {
    eta <- log(K) / Delta
    result <- mix(L, eta=eta)
    w <- result$w
    Mprev <- result$M
    h[t] <- sum(w * l[t,])
    weights[t,] <- w
    L <- L + l[t,]
    result <- mix(L, eta=eta)
    delta <- max(0, h[t] - (result$M - Mprev))
    Delta <- Delta + delta
    etas[t] <- eta
  }
  
  return(list("h"=h, "weights"=weights, "eta"=etas))
}

