require(nnet)
require(rpart)

# nnet ------
nnet.train <- function(x, y, ...){
  nnet(x, y, ...)
}

nnet.preds <- function(out, newx, ...){
  predict(out, newx, ...)
}

nnet.funs <- list(
  train = nnet.train,
  predict = nnet.preds
)

# COMA -----
# Hedge Algorithm
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
  return(list("h"=h, "weights"=weights))
}
  
# AdaHedge algorithm
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
    result <- mix(L, eta)
    w <- result$w
    Mprev <- result$M
    etas[t] <- eta
    h[t] <- sum(w * l[t,])
    weights[t,] <- w
    L <- L + l[t,]
    result <- mix(L, eta)
    delta <- max(0, h[t] - (result$M - Mprev))
    Delta <- Delta + delta
  }
  
  return(list("h"=h, "weights"=weights, "eta"=etas))
}

# Other Functions -----
loss_fun_ls <- function(c, sets, a, b){
  # c: target point
  # sets: list of sets
  # a: parm
  size_int <- lapply(sets, function(x) x$up - x$lo) %>% unlist()
  cov_int  <- lapply(sets, function(x) c(x$lo <= c & c <= x$up)) %>% unlist()
  loss_vec <- a * size_int - b * cov_int
  return(loss_vec)
}

weighted_sum <- function(x, w){
  sum(x * w)
}

sizes <- function(sets){
  # sets: list of sets
  lapply(sets, function(x) x$up - x$lo) %>% unlist()
}

# Majority vote -----

# Given a matrix of containing the lower and the upper bounds of the intervals
# return a set using a majority vote procedure
# majority vote
# counts in a interval 
counts_int <- function(M, a, b, w){
  k <- ncol(M)
  num_int <- weighted.mean(I(M[,1]<=((a+b)*0.5) & ((a+b)*0.5)<=M[,2]), w)
  return(num_int)
}


# M: matrix of interval (each interval lower and upper)
# q: quantile
majority_vote <- function(M, w, rho=0.5){
  k <- nrow(M)
  breaks <- as.vector(M)
  breaks <- unique(breaks)
  breaks <- breaks[order(breaks)]
  i <- 1
  lower <- upper <- NULL
  
  while(i < length(breaks)) {
    cond <- (counts_int(M, breaks[i], breaks[i+1], w)>rho)
    if(cond){
      lower <- c(lower, breaks[i])
      j <- i
      while(j < length(breaks) & cond){
        j <- j+1
        cond <- counts_int(M, breaks[j], breaks[j+1], w)>rho
      }
      i <- j
      upper <- c(upper, breaks[i])
    }
    i <- i+1
  }
  if(is.null(lower)){
    return(NA)
  }else{
    return(cbind(lower, upper))
  }
} 



# coverage function
covr_fun <- function(ci, target){
  if(sum(is.na(ci))==1){
    return(0)
  }else{
    covr <- 0
    l   <- nrow(ci)
    if(is.null(l)){
      covr <- covr + as.numeric(I((ci[1] <= target) && (target <= ci[2])))
    }else{
      for(i in 1:l){
        covr <- covr + as.numeric(I((ci[i,1] <= target) && (target <= ci[i,2])))
      }
    }
    return(covr)
  }
}

# loss function
loss_fun <- function(ci){
  if(sum(is.na(ci))==1){
    return(0)
  }else{
    sz  <- 0
    l   <- nrow(ci)
    if(is.null(l)){
      sz <- sz + (ci[2]-ci[1])
    }else{
      for(i in 1:l){
        sz <- sz + (ci[i,2]-ci[i,1])
      }
    }
    return(sz)
  }
}




