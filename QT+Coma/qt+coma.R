rm(list = ls())
library(glmnet)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(mvtnorm)
source("utils_funs.R")
set.seed(12345)

# Create data -----
# indicators of the data
ind_1 <- c(1:50, 101:150, 251:350 ,451:550, 651:750, 851:950, 1051:1150, 1251:1350)
ind_2 <- c(51:100, 151:250, 351:450, 551:650, 751:850, 951:1050, 1151:1250, 1351:1450)

N     <- length(ind_1) + length(ind_2)
Sigma <- matrix(c(1, .0, .0, 1), 2, 2)
X     <- rmvnorm(N, sigma = Sigma)
x1    <- X[,1]
x2    <- X[,2]
beta  <- 2

y        <- rep(NA, N)
y[ind_1] <- beta*x1[ind_1] + rnorm(length(ind_1))
y[ind_2] <- beta*x2[ind_2] + rnorm(length(ind_2))


# Linear Model -----
run_lm <- function(Y,X,alpha,nu,q,tinit=100){
  # Y: outcome
  # X: covariates
  # alpha: conf.level
  # nu: learning parameter
  # q: initial quantiles
  
  T <- length(Y)
  ## Initialize data storage variables
  qTrajectory <- rep(NA, T-tinit+1)
  adaptErrSeq <- rep(NA, T-tinit+1)
  piAdapt     <- matrix(NA, nrow = T-tinit+1, ncol=2)
  qT          <- q

  for(t in tinit:T){
    newX <- X[(t-tinit+1):(t-1)]
    newY <- Y[(t-tinit+1):(t-1)]
    
    ### Fit regression
    lmfit <- lm(newY ~ newX)
    
    
    predt <- c(1, X[t])%*%lmfit$coef
    
    ### Compute errt
    adaptErrSeq[t-tinit+1] <- 1 - I(abs(predt - Y[t]) <= qT)
    piAdapt[t-tinit+1,]    <- c(predt - qT, predt + qT)

    ## update qT
    qTrajectory[t-tinit+1] <- qT
    qT <- qT + nu*(adaptErrSeq[t-tinit+1]-alpha)
    
  }
  return(list(q_t=qTrajectory,
              AdaptErr=adaptErrSeq,
              piAdapt=piAdapt))
}

lm1 <- run_lm(Y=y, X=x1, alpha=0.1, q=1, nu=1)
lm2 <- run_lm(Y=y, X=x2, alpha=0.1, q=1, nu=1)
mean(lm1$AdaptErr); mean(lm2$AdaptErr)


# simulation -----
n_iter <- 1000
weights_B1 <- weights_B2 <- cumsum_lm1 <- cumsum_lm2 <- matrix(NA, nrow = N-99, ncol = n_iter)
lcov_1 <- lcov_2 <- matrix(NA, nrow = N-99, ncol = n_iter)
for(i in 1:n_iter){
  # generate data
  X <- rmvnorm(N, sigma = Sigma)
  x1 <- X[,1]
  x2 <- X[,2]
  beta <- 2
  y <- rep(NA, N)
  y[ind_1] <- beta*x1[ind_1] + rnorm(length(ind_1))
  y[ind_2] <- beta*x2[ind_2] + rnorm(length(ind_2))
  
  lm1 <- run_lm(Y=y, X=x1, alpha=0.1, q=1, nu=1)
  lm2 <- run_lm(Y=y, X=x2, alpha=0.1, q=1, nu=1)
  loss.matrix <- cbind(lm1$piAdapt[,2] - lm1$piAdapt[,1], 
                       lm2$piAdapt[,2] - lm2$piAdapt[,1])
  cumsum_lm1[,i] <- cumsum(loss.matrix[,1])
  cumsum_lm2[,i] <- cumsum(loss.matrix[,2])
  hedAlg <- adahedge(loss.matrix)
  weights_B1[,i] <- hedAlg$weights[,1]
  weights_B2[,i] <- hedAlg$weights[,2]
  lcov_1[,i] <- lm1$AdaptErr
  lcov_2[,i] <- lm2$AdaptErr
  if(i %% 10 == 0){
    print(sprintf("Done %i time steps",i))
  }
}


data.plot1 <- data.frame(
  t = 1:(N-99),
  w1 = rowMeans(weights_B1),
  w2 = rowMeans(weights_B2)
)

data.plot2 <- data.frame(
  t = 1:(N-99),
  L1 = rowMeans(cumsum_lm1),
  L2 = rowMeans(cumsum_lm2),
  a1 = I((rowMeans(cumsum_lm1) - rowMeans(cumsum_lm2))>0)
)



pl1<-ggplot(data.plot2, aes(x = t)) +
  geom_line(aes(y = L1, color = "L1"), size = 1) +
  geom_line(aes(y = L2, color = "L2"), size = 1) +
  labs(title = "",
       x = "t",
       y = "Cumulative loss") +
  scale_color_manual(
    values = c("L1" = "tan2", "L2" = "seagreen"),
    labels = c("L1" = expression(L[1]), "L2" = expression(L[2]))) +
  geom_segment(data = subset(data.plot2, a1), aes(x = t, xend = t, y = 0, yend = max(L1)), color = "gray", alpha = 0.02) +
  theme_classic() +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(title = "")) 

pl2<-ggplot(data.plot1, aes(x = t)) +
  geom_line(aes(y = w1, color = "w1"), size = 1) +
  geom_line(aes(y = w2, color = "w2"), size = 1) +
  labs(title = "",
       x = "t",
       y = "Weights") +
  scale_color_manual(values = c("w1" = "tan2", "w2" = "seagreen"),
                     labels = c("w1" = expression(w[1]), "L2" = expression(w[2]))) +
  theme_classic() +
  theme(legend.position = "bottom") +
  geom_segment(data = subset(data.plot2, a1), aes(x = t, xend = t, y = 0, yend = 1), color = "gray", alpha = 0.02) +
  guides(color = guide_legend(title = ""))
grid.arrange(pl1, pl2, ncol = 2)

