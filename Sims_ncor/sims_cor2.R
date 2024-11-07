remove(list = ls())
library(conformalInference)

# AdaHedge algorithm -----
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
  weights <- matrix(NA, N, K)
  Delta <- 0
  
  for (t in 1:N) {
    eta <- log(K) / Delta
    result <- mix(L, eta)
    w <- result$w
    Mprev <- result$M
    h[t] <- sum(w * l[t,])
    weights[t,] <- w
    L <- L + l[t,]
    result <- mix(L, eta)
    delta <- max(0, h[t] - (result$M - Mprev))
    Delta <- Delta + delta
  }
  
  return(list("h"=h, "weights"=weights))
}



# simulation function -----
sims_data <- function(N, rho, p, beta){
  eps    <- rep(NA, N)
  eps[1] <- rnorm(1, 0, sqrt(1-rho^2))
  for(i in 2:N){
    eps[i] <- rho*eps[i-1] + rnorm(1, 0, sqrt(1-rho^2))
  }
  X <- matrix(rnorm(N*p), nrow = N, ncol = p)
  y <- X %*% beta + eps
  return(list(X=X, y=y))
}

# set the pars -----
funs <- list()
funs[[1]] <- lasso.funs(standardize = F, lambda = exp(-4))
funs[[2]] <- lasso.funs(standardize = F, lambda = exp(-3))
funs[[3]] <- lasso.funs(standardize = F, lambda = exp(-2))
funs[[4]] <- lasso.funs(standardize = F, lambda = exp(-1))

# Example ------
N    <- 150
rho  <- 0.1
p    <- 100
beta <- c(rep(2/sqrt(5), 5), rep(0, p-5))

set.seed(1234)
dati <- sims_data(N, rho, p, beta)

tinit <- 100
alpha <- 0.1
k     <- length(funs)

covs_t <- matrix(NA, nrow = N-tinit+1, ncol = k)
loss_t <- matrix(NA, nrow = N-tinit+1, ncol = k)
cis_t  <- vector("list", N-tinit)


for(i in tinit:N){
  X  <- dati$X[1:(i-1),]
  y  <- dati$y[1:(i-1)]
  X0 <- dati$X[i,]
  y0 <- dati$y[i]
  # prediction intervals
  conf.pred <-lapply(funs, function(z) conformal.pred(X, y, X0, alpha = alpha,
                                                                 train.fun = z$train, predict.fun = z$predict))
  
  cis <- matrix(NA, nrow = k, ncol = 2) 
  for(l in 1:k){
    cis[l,1]    <- conf.pred[[l]]$lo
    cis[l,2]    <- conf.pred[[l]]$up
    covs_t[i-tinit+1,l] <- as.numeric(I(conf.pred[[l]]$lo <= y0 & y0 <= conf.pred[[l]]$up))
    loss_t[i-tinit+1,l] <- as.numeric(conf.pred[[l]]$up - conf.pred[[l]]$lo)
  }
  cis_t[[i-tinit+1]]   <- cis

  #if(i %% 10 == 0) cat(i, "\n")
}
colMeans(covs_t); colMeans(loss_t)


# Simulations -----
B <- 1000         # number of replications
N <- 200          # number of observations
k <- length(funs) # number of algorithms

w_alg1 <- w_alg2 <- w_alg3 <- w_alg4 <- matrix(NA, nrow = N-tinit+1, ncol = B)
m_alg1 <- m_alg2 <- m_alg3 <- m_alg4 <- matrix(NA, nrow = N-tinit+1, ncol = B)

set.seed(123)
for(j in 1:B){
  # simulate data 
  dati <- sims_data(N, rho, p, beta)
  covs_t <- matrix(NA, nrow = N-tinit+1, ncol = k)
  loss_t <- matrix(NA, nrow = N-tinit+1, ncol = k)
  
  for(i in tinit:N){
    X  <- dati$X[1:(i-1),]
    y  <- dati$y[1:(i-1)]
    X0 <- dati$X[i,]
    y0 <- dati$y[i]
    # prediction intervals
    conf.pred <-lapply(funs, function(z) conformal.pred(X, y, X0, alpha = alpha,
                                                        train.fun = z$train, predict.fun = z$predict))
    
    cis <- matrix(NA, nrow = k, ncol = 2) 
    for(l in 1:k){
      cis[l,1]    <- conf.pred[[l]]$lo
      cis[l,2]    <- conf.pred[[l]]$up
      covs_t[i-tinit+1,l] <- as.numeric(I(conf.pred[[l]]$lo <= y0 & y0 <= conf.pred[[l]]$up))
      loss_t[i-tinit+1,l] <- as.numeric(conf.pred[[l]]$up - conf.pred[[l]]$lo)
    }
    cis_t[[i-tinit+1]]   <- cis
  }
  
  ada_alg <- adahedge(loss_t)
  
  w_alg1[,j] <- ada_alg$weights[,1]
  w_alg2[,j] <- ada_alg$weights[,2]
  w_alg3[,j] <- ada_alg$weights[,3]
  w_alg4[,j] <- ada_alg$weights[,4]
  
  m_alg1[,j] <- 1-covs_t[,1]
  m_alg2[,j] <- 1-covs_t[,2]
  m_alg3[,j] <- 1-covs_t[,3]
  m_alg4[,j] <- 1-covs_t[,4]
  
  if(j %% 10 == 0) cat(j, "\n")
}

save(w_alg1, w_alg2, w_alg3, w_alg4,
     m_alg1, m_alg2, m_alg3, m_alg4,
     file = "sims_cor.RData")


