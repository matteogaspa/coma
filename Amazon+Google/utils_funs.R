# Majority vote -----

# Given a matrix of containing the lower and the upper bounds of the intervals
# return a set using a majority vote procedure
counts_int <- function(M, a, b, w){
  k <- ncol(M)
  num_int <- weighted.mean(I(M[,1]<=((a+b)*0.5) & ((a+b)*0.5)<=M[,2]), w)
  return(num_int)
}

majority_vote <- function(M, w, rho=0.5){
  k <- nrow(M)
  breaks <- as.vector(M)
  breaks <- unique(breaks)
  breaks <- breaks[order(breaks)]
  i <- 1
  lower <- upper <- NULL
  
  if(mean(M[,2]==Inf)>rho){
    return(c(-Inf, Inf))
  }
  
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

# Create matrix -----

# Create a (list of) matrix(ces) from a list obtained from conf.pred

create.matrix <- function(l){
  # l: list conf.pred
  k <- length(l)
  n <- length(l[[1]]$lo)
  list.mat <- vector("list", n)
  for(i in 1:n){
    mat.res <- matrix(NA, nrow = k, ncol = 2)
    for(j in 1:k){
      mat.res[j, 1] <- l[[j]]$lo[i]
      mat.res[j, 2] <- l[[j]]$up[i]
    }
    list.mat[[i]] <- mat.res
  }
  if(n == 1){
    return(list.mat[[1]])
  }
  else{
    return(list.mat)
  }
}



