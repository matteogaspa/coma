rm(list = ls())
library(glmnet)
library(ggplot2)
library(ggtext)
library(gridExtra)
library(dplyr)
source("utils_elec2qt.R")

set.seed(1)
# Load data -----
elec <- read.csv("electricity-normalized.csv")
head(elec)
str(elec)
# delete constant transfer
const.ind <- min(which(elec$transfer != elec$transfer[1]))
elec <- elec[const.ind:NROW(elec),]
const.ind <- min(which(elec$transfer != elec$transfer[1]))
elec <- elec[const.ind:NROW(elec),]
# from 9 to 12 am
time.ran <- c(9*2/48, 0.5)
elec <- elec[(elec$period >= time.ran[1] & elec$period <= time.ran[2]),]
# select the covariates
elec <- elec[,4:8]
# create lag matrix
elec_lag <- embed(elec$transfer, 6)
elec_lag <- elec_lag[-NROW(elec_lag),]
# delete the first 6 obs.
elec <- elec[7:NROW(elec),]
# create outcome and covariates 
elec_y <- elec$transfer
elec_x <- as.matrix(elec[,1:4])

NROW(elec); NROW(elec_y); NROW(elec_lag); NROW(elec_x)


# Linear Model -----
run_lm <- function(Y,X,alpha,q,nu,eps=0.05,tinit=100){
  # Y: outcome
  # X: covariates
  # alpha: conf.level
  # q: initial quantiles
  # nu: learning parameter (\nu^t=\nu*t^(-.5-eps))
  # eps: epsilon parameter described in Angelopoulos et al. (2024). If =1/2 then \nu^t = \nu
  # tinit: burn in
  
  T <- length(Y)
  ## Initialize data storage variables
  qTrajectory <- rep(NA, nrow = T-tinit+1)
  adaptErrSeq <- rep(NA, nrow = T-tinit+1)
  piAdapt     <- matrix(NA, nrow=T-tinit+1, ncol=2)
  qT          <- q
  nuT         <- nu
  
  for(t in tinit:T){
    newX <- X[1:(t-1),]
    newY <- Y[1:(t-1)]
    
    ### Fit model 
    lmfit <- lm(newY ~ newX)
    
    ### Compute prediction
    predt <- as.numeric(c(1, X[t,])%*%lmfit$coef)
    
    ### Compute errt for both methods using the score (abs.resds)
    adaptErrSeq[t-tinit+1] <- 1 - I(abs(Y[t] - predt) <= qT)
    piAdapt[t-tinit+1,] <- c(predt - qT, predt + qT)
    
    ## update qT and nuT
    qTrajectory[t-tinit+1] <- qT
    nuT <- nu*(t-tinit+1)^(-0.5-eps)
    qT  <- qT + nuT*(adaptErrSeq[t-tinit+1]-alpha)
    
    if(t %% 100 == 0){
      print(sprintf("Done %i time steps",t))
    }
  }
  return(list(q_t=qTrajectory,
              AdaptErr=adaptErrSeq,
              piAdapt=piAdapt))
}

# Linear Model using covriates at time t
m1 <- run_lm(elec_y, elec_x, alpha = 0.1, q = 0, nu = 2)
mean(m1$AdaptErr)
# Linear Model using lag as covariates AR(2)
m2 <- run_lm(elec_y, elec_lag[,1:2], alpha = 0.1, q = 0, nu = 2)
mean(m2$AdaptErr)
# Linear Model using lag as covariates AR(2) + 1covariate
m3 <- run_lm(elec_y, cbind(elec_lag[,1:2], elec_x[,1]), alpha = 0.1, q = 0, nu = 2)
mean(m3$AdaptErr)

data.plot <- data.frame(
  iter = 1:length(m1$AdaptErr),
  m1 = 1-stats::filter(m1$AdaptErr, rep(1/200, 200)),
  m2 = 1-stats::filter(m2$AdaptErr, rep(1/200, 200)),
  m3 = 1-stats::filter(m3$AdaptErr, rep(1/200, 200))
)

p_lm1<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = m1), color = "blue", linetype = "solid") +
  geom_hline(yintercept = 0.90, color = "black", linetype = "dashed") +
  labs(title = "LM", x = "Iter", y = "Local Level Coverage", color = "") +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0.75, 1)
p_ar2<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = m2), color = "blue", linetype = "solid") +
  geom_hline(yintercept = 0.90, color = "black", linetype = "dashed") +
  labs(title = "AR(2)", x = "Iter", y = "Local Level Coverage", color = "") +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0.75, 1)
p_armax1<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = m3), color = "blue", linetype = "solid") +
  geom_hline(yintercept = 0.90, color = "black", linetype = "dashed") +
  labs(title = "ARMAX(2,0)", x = "Iter", y = "Local Level Coverage", color = "") +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0.75, 1)
plot_models <- grid.arrange(p_lm1, p_ar2, p_armax1, ncol = 3)

# AdaHedge + Hedge
# AdaHedge -----
# compute the loss matrix 
loss.matrix <- cbind(m1$piAdapt[,2]-m1$piAdapt[,1], m2$piAdapt[,2]-m2$piAdapt[,1], m3$piAdapt[,2]-m3$piAdapt[,1])
adahedAlg <- adahedge(loss.matrix)

N <- NROW(adahedAlg$weights)
K <- NCOL(adahedAlg$weights)
t <- NROW(elec_lag)-N
pi.m <- pi.wm <- vector("list", N)
m.cov <- rep(NA, N)
wm.cov <- rep(NA, N)

for(i in 1:N){
  conf.t <- rbind(m1$piAdapt[i,], m2$piAdapt[i,], m3$piAdapt[i,])
  maj.int <- majority_vote(conf.t, w=adahedAlg$weights[i,], 0.5)
  wmaj.int <- majority_vote(conf.t, w=adahedAlg$weights[i,], runif(1, 0.5, 1))
  pi.m[[i]] <- maj.int
  pi.wm[[i]] <- wmaj.int
  m.cov[i] <- covr_fun(maj.int, elec_y[i+t])
  wm.cov[i] <- covr_fun(wmaj.int, elec_y[i+t])
  if(i %% 100 == 0){
    print(sprintf("Done %i time steps",i))
  }
}

mean(m.cov)
mean(wm.cov)

data.plot <- data.frame(
  iter = 1:N,
  local.m = stats::filter(m.cov, rep(1/200, 200)),
  local.wm = stats::filter(wm.cov, rep(1/200, 200))
)

pM1<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = local.m, color = "Weighted Majority"), linetype = "solid") +
  geom_line(aes(y = local.wm, color = "Rand. Weighted Majority"), linetype = "solid") +
  geom_hline(yintercept = 0.80, color = "black", linetype = "dashed") +
  labs(title = expression("Adaptive"~eta), x = "Iter", y = "Local Level Coverage", color = "") +
  scale_color_manual(values = c("Weighted Majority" = "forestgreen", "Rand. Weighted Majority" = "cyan3", "Bern 0.9" = "orange")) +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0.75, 1)
pM1

# loss
sz.m <- unlist(lapply(pi.m, loss_fun))
sz.wm <- unlist(lapply(pi.wm, loss_fun))

data.plot <- data.frame(
  iter = 1:N,
  lm = stats::filter(loss.matrix[,1], rep(1/100, 100)),
  ar = stats::filter(loss.matrix[,2], rep(1/100, 100)),
  armax = stats::filter(loss.matrix[,3], rep(1/100, 100)), 
  sz.m = stats::filter(sz.m, rep(1/100, 100)),
  sz.wm = stats::filter(sz.wm, rep(1/100, 100))
)

plot_loss1<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = lm, color = "LM"), linetype = "solid", size = 1) +
  geom_line(aes(y = ar, color = "AR(2,0)"), linetype = "solid", size = 1) +
  geom_line(aes(y = armax, color = "ARMAX(2,0)"), linetype = "solid", size = 1) +
  geom_line(aes(y = sz.m, color = "Weighted Maj."), linetype = "dashed", size = 1) +
  #geom_line(aes(y = sz.wm, color = "Rand. Weighted Maj."), linetype = "solid") +
  labs(title = expression("Adaptive"~eta), x = "Iter", y = "Loss", color = "") +
  scale_color_manual(values = c("LM" = "red", "AR(2,0)" = "forestgreen", "ARMAX(2,0)" = "blue", "Weighted Maj." = "orange", "Rand. Weighted Maj." = "orange")) +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0.1, 0.75)
plot_loss1


# hedge (fixed eta) ------
hedAlg <- hedge(loss.matrix, eta = 0.1)
N <- NROW(hedAlg$weights)
K <- NCOL(hedAlg$weights)
t <- NROW(elec_lag)-N
pi.m <- pi.wm <- vector("list", N)
m.cov <- rep(NA, N)
wm.cov <- rep(NA, N)

for(i in 1:N){
  conf.t <- rbind(m1$piAdapt[i,], m2$piAdapt[i,], m3$piAdapt[i,])
  maj.int <- majority_vote(conf.t, w=hedAlg$weights[i,], 0.5)
  wmaj.int <- majority_vote(conf.t, w=hedAlg$weights[i,], runif(1, 0.5, 1))
  pi.m[[i]] <- maj.int
  pi.wm[[i]] <- wmaj.int
  m.cov[i] <- covr_fun(maj.int, elec_y[i+t])
  wm.cov[i] <- covr_fun(wmaj.int, elec_y[i+t])
  if(i %% 100 == 0){
    print(sprintf("Done %i time steps",i))
  }
}

mean(m.cov)
mean(wm.cov)

data.plot <- data.frame(
  iter = 1:N,
  local.m = stats::filter(m.cov, rep(1/200, 200)),
  local.wm = stats::filter(wm.cov, rep(1/200, 200))
)

pM2<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = local.m, color = "Weighted Majority"), linetype = "solid") +
  geom_line(aes(y = local.wm, color = "Rand. Weighted Majority"), linetype = "solid") +
  geom_hline(yintercept = 0.80, color = "black", linetype = "dashed") +
  labs(title = expression(eta==0.1), x = "Iter", y = "Local Level Coverage", color = "") +
  scale_color_manual(values = c("Weighted Majority" = "forestgreen", "Rand. Weighted Majority" = "cyan3", "Bern 0.9" = "orange")) +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0.75, 1)
pM2

# loss
sz.m <- unlist(lapply(pi.m, loss_fun))
sz.wm <- unlist(lapply(pi.wm, loss_fun))

data.plot <- data.frame(
  iter = 1:N,
  lm = stats::filter(loss.matrix[,1], rep(1/100, 100)),
  ar = stats::filter(loss.matrix[,2], rep(1/100, 100)),
  armax = stats::filter(loss.matrix[,3], rep(1/100, 100)), 
  sz.m = stats::filter(sz.m, rep(1/100, 100)),
  sz.wm = stats::filter(sz.wm, rep(1/100, 100))
)

plot_loss2<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = lm, color = "LM"), linetype = "solid", size = 1) +
  geom_line(aes(y = ar, color = "AR(2,0)"), linetype = "solid", size = 1) +
  geom_line(aes(y = armax, color = "ARMAX(2,0)"), linetype = "solid", size = 1) +
  geom_line(aes(y = sz.m, color = "Weighted Maj."), linetype = "dashed", size = 1) +
  #geom_line(aes(y = sz.wm, color = "Rand. Weighted Maj."), linetype = "solid") +
  labs(title = expression(eta==0.1), x = "Iter", y = "Loss", color = "") +
  scale_color_manual(values = c("LM" = "red", "AR(2,0)" = "forestgreen", "ARMAX(2,0)" = "blue", "Weighted Maj." = "orange", "Rand. Weighted Maj." = "purple")) +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0.1, 0.75)
plot_loss2

plot_merge <- grid.arrange(pM1, pM2, ncol = 2)
plot_tot   <- grid.arrange(plot_models, plot_merge)


# weights
data.plot1 <- as.matrix(adahedAlg$weights)
colnames(data.plot1) <- c("lm", "ar", "armax")
data.plot1 <- as.data.frame(data.plot1)
data.plot1$iter <- 1:nrow(data.plot1)

p1<-ggplot(data.plot1, aes(x = iter)) +
  geom_line(aes(y = lm, color = "LM"), linetype = "solid", size = 1) +
  geom_line(aes(y = ar, color = "AR(2)"), linetype = "solid", size = 1) +
  geom_line(aes(y = armax, color = "ARMAX(2,0)"), linetype = "solid", size = 1) +
  labs(title = expression("Adaptive"~eta), x = "Iter", y = "Weights", color = "") +
  scale_color_manual(values = c("LM" = "red", "AR(2)" = "forestgreen", "ARMAX(2,0)" = "blue")) +
  theme_minimal() + theme(legend.position = "bottom") 
p1


data.plot2 <- as.matrix(hedAlg$weights)
colnames(data.plot2) <- c("lm", "ar", "armax")
data.plot2 <- as.data.frame(data.plot2)
data.plot2$iter <- 1:nrow(data.plot2)
p2<-ggplot(data.plot2, aes(x = iter)) +
  geom_line(aes(y = lm, color = "LM"), linetype = "solid", size = 1) +
  geom_line(aes(y = ar, color = "AR(2)"), linetype = "solid", size = 1) +
  geom_line(aes(y = armax, color = "ARMAX(2,0)"), linetype = "solid", size = 1) +
  labs(title = expression(eta==0.1), x = "Iter", y = "Weights", color = "") +
  scale_color_manual(values = c("LM" = "red", "AR(2)" = "forestgreen", "ARMAX(2,0)" = "blue")) +
  theme_minimal() + theme(legend.position = "bottom") 
p2

grid.arrange(p1, p2, plot_loss1, plot_loss2, nrow = 2, ncol = 2)







