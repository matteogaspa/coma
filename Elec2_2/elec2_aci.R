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

NROW(elec_y); NROW(elec_x); NROW(elec_lag)


# Hedge with fixed learning rate -----
runMix <- function(Y, X1, X2, X3, alpha, gamma, tinit = 200, splitSize = 0.6, nu = 0.1){
  T <- length(Y)
  ## Initialize data storage variables
  alphaTrajectory <- rep(alpha,T-tinit+1)
  adaptErrSeq <-  rep(1,T-tinit+1)
  noAdaptErrorSeq <-  rep(1,T-tinit+1)
  alphat <- alpha
  piAdapt <- vector("list", T-tinit+1)
  piNoAdapt <- vector("list", T-tinit+1)
  wTrajectory <- matrix(NA, nrow=T-tinit+1, ncol=3)
  lossTrajectory <- matrix(NA, nrow=T-tinit+1, ncol=3)
  wt <- rep(1/3, 3)
  for(t in tinit:T){
    ### Split data into training and calibration set
    trainPoints <- sample(1:(t-1),round(splitSize*(t-1)))
    calpoints <- (1:(t-1))[-trainPoints]
    newX1 <- X1[1:(t-1),]
    newX2 <- X2[1:(t-1),]
    newX3 <- X3[1:(t-1),]
    newY <- Y[1:(t-1)]
    X1train <- newX1[trainPoints,]
    X2train <- newX2[trainPoints,]
    X3train <- newX3[trainPoints,]
    Ytrain <- newY[trainPoints]
    X1Cal <- newX1[calpoints,]
    X2Cal <- newX2[calpoints,]
    X3Cal <- newX3[calpoints,]
    YCal <- newY[calpoints]
    
    ### Fit regression algorithms on training setting
    mfit1 <- lm(Ytrain ~ X1train)
    mfit2 <- lm(Ytrain ~ X2train)
    mfit3 <- lm(Ytrain ~ X3train)
    
    ### Compute conformity score on calibration set and on new data example for model 1
    predForCal1 <- cbind(rep(1,nrow(X1Cal)),X1Cal)%*%mfit1$coef
    scores1 <- abs(predForCal1 - YCal)
    predt1 <- as.numeric(c(1, X1[t,])%*%mfit1$coef)
    
    ### Compute conformity score on calibration set and on new data example for model 2
    predForCal2 <- cbind(rep(1,nrow(X2Cal)),X2Cal)%*%mfit2$coef
    scores2 <- abs(predForCal2 - YCal)
    predt2 <- as.numeric(c(1, X2[t,])%*%mfit2$coef)
    
    ### Compute conformity score on calibration set and on new data example for model 3
    predForCal3 <- cbind(rep(1,nrow(X3Cal)),X3Cal)%*%mfit3$coef
    scores3 <- abs(predForCal3 - YCal)
    predt3 <- as.numeric(c(1, X3[t,])%*%mfit3$coef)
    
    ### Compute errt for both methods
    confQuantnoAdapt1 <- quantile(scores1,probs=1-alpha)
    confQuantnoAdapt2 <- quantile(scores2,probs=1-alpha)
    confQuantnoAdapt3 <- quantile(scores3,probs=1-alpha)
    confQuantsnoAdapt  <- matrix(NA, nrow = 3, ncol = 2)
    confQuantsnoAdapt[1,] <- c(predt1 - confQuantnoAdapt1, predt1 + confQuantnoAdapt1)
    confQuantsnoAdapt[2,] <- c(predt2 - confQuantnoAdapt2, predt2 + confQuantnoAdapt2)
    confQuantsnoAdapt[3,] <- c(predt3 - confQuantnoAdapt3, predt3 + confQuantnoAdapt3)
    
    piNoAdapt[[t-tinit+1]] <- majority_vote(confQuantsnoAdapt, wt/sum(wt), 0.5)
    noAdaptErrorSeq[t-tinit+1] <- 1 - covr_fun(piNoAdapt[[t-tinit+1]], Y[t])
    
    if(alphat >=1){
      adaptErrSeq[t-tinit+1] <- 1
      confQuantAdapt <- 0
      piAdapt[[t-tinit+1]] <- c(predt1 - confQuantnoAdapt1, predt1 + confQuantnoAdapt1)
      wTrajectory[t-tinit+1,] <- wt/sum(wt)
      lossTrajectory[t-tinit+1,] <- rep(0, 3) 
    }else if (alphat <=0){
      adaptErrSeq[t-tinit+1] <- 0
      confQuantAdapt <- Inf
      piAdapt[[t-tinit+1]] <- c(-Inf, Inf)
      wTrajectory[t-tinit+1,] <- wt/sum(wt)
      lossTrajectory[t-tinit+1,] <- rep(1, 3)
    }else{
      confQuantAdapt1 <- quantile(scores1,probs=1-alphat)
      confQuantAdapt2 <- quantile(scores2,probs=1-alphat)
      confQuantAdapt3 <- quantile(scores3,probs=1-alphat)
      confQuants  <- matrix(NA, nrow = 3, ncol = 2)
      confQuants[1,] <- c(predt1 - confQuantAdapt1, predt1 + confQuantAdapt1)
      confQuants[2,] <- c(predt2 - confQuantAdapt2, predt2 + confQuantAdapt2)
      confQuants[3,] <- c(predt3 - confQuantAdapt3, predt3 + confQuantAdapt3)
      piAdapt[[t-tinit+1]] <- majority_vote(confQuants, wt/sum(wt), 0.5)
      adaptErrSeq[t-tinit+1] <- 1 - covr_fun(piAdapt[[t-tinit+1]],Y[t])
      ## update weights
      losst <- atan(confQuants[,2]-confQuants[,1])
      lossTrajectory[t-tinit+1,] <- losst
      wTrajectory[t-tinit+1,] <- wt/sum(wt)
      wt <- (wt * exp(-nu*losst)) /sum(wt)
    }
    
    ## update alphat
    alphaTrajectory[t-tinit+1] <- alphat
    alphat <- alphat + gamma*(alpha-adaptErrSeq[t-tinit+1])
    
    if(t %% 100 == 0){
      print(sprintf("Done %i time steps",t))
    }
  }
  return(list(alpha_t=alphaTrajectory,
              AdaptErr=adaptErrSeq,
              noAdaptErr=noAdaptErrorSeq,
              piAdapt=piAdapt,
              piNoAdapt=piNoAdapt,
              weights=wTrajectory,
              losses=lossTrajectory))
}

rMix <- runMix(elec_y, elec_x, elec_lag[,1:2], cbind(elec_lag[,1:2], elec_x[,1]), alpha = 0.05, gamma = 0.01)

data.plot <- data.frame(
  "iter" = 1:length(rMix$AdaptErr),
  "alpha_ad" = 1 - stats::filter(rMix$AdaptErr, rep(1/200, 200)),
  "alpha_noad" = 1 - stats::filter(rMix$noAdaptErr, rep(1/200, 200))
)

f_eta_err <- ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = alpha_ad), color = "blue", linetype = "solid") +
  #geom_line(aes(y = alpha_noad), color = "red", linetype = "solid") +
  geom_hline(yintercept = 0.95, color = "black", linetype = "dashed") +
  labs(title = expression(eta==0.1), x = "Iter", y = "Local Level Coverage", color = "") +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0.75, 1)

data.plot <- data.frame(
  "iter" = 1:length(rMix$AdaptErr),
  "lm" = rMix$weights[,1],
  "ar" = rMix$weights[,2],
  "armax" = rMix$weights[,3]
)

f_eta_wts <- ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = lm, color = "LM"), linetype = "solid", size = 1) +
  geom_line(aes(y = ar, color = "AR(2,0)"), linetype = "solid", size = 1) +
  geom_line(aes(y = armax, color = "ARMAX(2,0)"), linetype = "solid", size = 1) +
  labs(title = expression(eta==0.1), x = "Iter", y = "Weights", color = "") +
  scale_color_manual(values = c("LM" = "red", "AR(2,0)" = "forestgreen", "ARMAX(2,0)" = "blue", "Weighted Maj." = "orange", "Rand. Weighted Maj." = "orange")) +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0,1)

data.plot <- data.frame(
  "iter" = 1:length(rMix$AdaptErr),
  "lm" = stats::filter(rMix$losses[,1], rep(1/100, 100)),
  "ar" = stats::filter(rMix$losses[,2], rep(1/100, 100)),
  "armax" = stats::filter(rMix$losses[,3], rep(1/100, 100)),
  "aci" = stats::filter(unlist(lapply(rMix$piAdapt, function(x) atan(x[2]-x[1]))),rep(1/100, 100))
)

f_eta_loss <- ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = lm, color = "LM"), linetype = "solid", size = 1) +
  geom_line(aes(y = ar, color = "AR(2,0)"), linetype = "solid", size = 1) +
  geom_line(aes(y = armax, color = "ARMAX(2,0)"), linetype = "solid", size = 1) +
  geom_line(aes(y = aci, color = "Weighted Maj."), linetype = "dashed", size = 1) +
  labs(title = expression(eta==0.1), x = "Iter", y = "Loss", color = "") +
  scale_color_manual(values = c("LM" = "red", "AR(2,0)" = "forestgreen", "ARMAX(2,0)" = "blue", "Weighted Maj." = "orange", "Rand. Weighted Maj." = "orange")) +
  theme_minimal() + theme(legend.position = "bottom") 



# Adahedge with adaptive learning rate
runMixAdapt <- function(Y, X1, X2, X3, alpha, gamma, tinit = 200, splitSize = 0.6){
  T <- length(Y)
  ## Initialize data storage variables
  alphaTrajectory <- rep(alpha,T-tinit+1)
  adaptErrSeq <-  rep(1,T-tinit+1)
  noAdaptErrorSeq <-  rep(1,T-tinit+1)
  alphat <- alpha
  piAdapt <- vector("list", T-tinit+1)
  piNoAdapt <- vector("list", T-tinit+1)
  wTrajectory <- matrix(NA, nrow=T-tinit+1, ncol=3)
  lossTrajectory <- matrix(NA, nrow=T-tinit+1, ncol=3)
  wt <- rep(1/3, 3)
  for(t in tinit:T){
    ### Split data into training and calibration set
    trainPoints <- sample(1:(t-1),round(splitSize*(t-1)))
    calpoints <- (1:(t-1))[-trainPoints]
    newX1 <- X1[1:(t-1),]
    newX2 <- X2[1:(t-1),]
    newX3 <- X3[1:(t-1),]
    newY <- Y[1:(t-1)]
    X1train <- newX1[trainPoints,]
    X2train <- newX2[trainPoints,]
    X3train <- newX3[trainPoints,]
    Ytrain <- newY[trainPoints]
    X1Cal <- newX1[calpoints,]
    X2Cal <- newX2[calpoints,]
    X3Cal <- newX3[calpoints,]
    YCal <- newY[calpoints]
    
    ### Fit regression algorithms on training setting
    mfit1 <- lm(Ytrain ~ X1train)
    mfit2 <- lm(Ytrain ~ X2train)
    mfit3 <- lm(Ytrain ~ X3train)
    
    ### Compute conformity score on calibration set and on new data example for model 1
    predForCal1 <- cbind(rep(1,nrow(X1Cal)),X1Cal)%*%mfit1$coef
    scores1 <- abs(predForCal1 - YCal)
    predt1 <- as.numeric(c(1, X1[t,])%*%mfit1$coef)
    
    ### Compute conformity score on calibration set and on new data example for model 2
    predForCal2 <- cbind(rep(1,nrow(X2Cal)),X2Cal)%*%mfit2$coef
    scores2 <- abs(predForCal2 - YCal)
    predt2 <- as.numeric(c(1, X2[t,])%*%mfit2$coef)
    
    ### Compute conformity score on calibration set and on new data example for model 3
    predForCal3 <- cbind(rep(1,nrow(X3Cal)),X3Cal)%*%mfit3$coef
    scores3 <- abs(predForCal3 - YCal)
    predt3 <- as.numeric(c(1, X3[t,])%*%mfit3$coef)
    
    ### Compute errt for both methods
    confQuantnoAdapt1 <- quantile(scores1,probs=1-alpha)
    confQuantnoAdapt2 <- quantile(scores2,probs=1-alpha)
    confQuantnoAdapt3 <- quantile(scores3,probs=1-alpha)
    confQuantsnoAdapt <- matrix(NA, nrow = 3, ncol = 2)
    confQuantsnoAdapt[1,] <- c(predt1 - confQuantnoAdapt1, predt1 + confQuantnoAdapt1)
    confQuantsnoAdapt[2,] <- c(predt2 - confQuantnoAdapt2, predt2 + confQuantnoAdapt2)
    confQuantsnoAdapt[3,] <- c(predt3 - confQuantnoAdapt3, predt3 + confQuantnoAdapt3)
    
    piNoAdapt[[t-tinit+1]] <- majority_vote(confQuantsnoAdapt, wt/sum(wt), 0.5)
    noAdaptErrorSeq[t-tinit+1] <- 1 - covr_fun(piNoAdapt[[t-tinit+1]], Y[t])
    
    if(alphat >=1){
      adaptErrSeq[t-tinit+1] <- 1
      confQuantAdapt <- 0
      piAdapt[[t-tinit+1]] <- c(predt1 - confQuantAdapt, predt1 + confQuantAdapt)
      wTrajectory[t-tinit+1,] <- wt/sum(wt)
      losst <- rep(0, 3)
      lossTrajectory[t-tinit+1,] <- losst
    }else if (alphat <= 0){
      adaptErrSeq[t-tinit+1] <- 0
      piAdapt[[t-tinit+1]] <- c(-Inf, Inf)
      wTrajectory[t-tinit+1,] <- wt/sum(wt)
      losst <- rep(1, 3)
      lossTrajectory[t-tinit+1,] <- losst
    }else{
      confQuantAdapt1 <- quantile(scores1,probs=1-alphat)
      confQuantAdapt2 <- quantile(scores2,probs=1-alphat)
      confQuantAdapt3 <- quantile(scores3,probs=1-alphat)
      confQuants  <- matrix(NA, nrow = 3, ncol = 2)
      confQuants[1,] <- c(predt1 - confQuantAdapt1, predt1 + confQuantAdapt1)
      confQuants[2,] <- c(predt2 - confQuantAdapt2, predt2 + confQuantAdapt2)
      confQuants[3,] <- c(predt3 - confQuantAdapt3, predt3 + confQuantAdapt3)
      piAdapt[[t-tinit+1]] <- majority_vote(confQuants, wt/sum(wt), 0.5)
      adaptErrSeq[t-tinit+1] <- 1-covr_fun(piAdapt[[t-tinit+1]], Y[t])
      ## update weights
      losst <- atan(confQuants[,2]-confQuants[,1])
      lossTrajectory[t-tinit+1,] <- losst
      wTrajectory[t-tinit+1,] <- wt/sum(wt)
      wt <- adahedge(matrix(lossTrajectory[1:(t-tinit+1),], t-tinit+1, 3))$weights[t-tinit+1,]
    }
    
    ## update alphat
    alphaTrajectory[t-tinit+1] <- alphat
    alphat <- alphat + gamma*(alpha-adaptErrSeq[t-tinit+1])
    
    if(t %% 100 == 0){
      print(sprintf("Done %i time steps",t))
    }
  }
  return(list(alpha_t=alphaTrajectory,
              AdaptErr=adaptErrSeq,
              noAdaptErr=noAdaptErrorSeq,
              piAdapt=piAdapt,
              piNoAdapt=piNoAdapt,
              weights=wTrajectory,
              losses=lossTrajectory))
}

rMixAdapt <- runMixAdapt(elec_y, elec_x, elec_lag[,1:2], cbind(elec_lag[,1:2], elec_x[,1]), alpha = 0.05, gamma = 0.005)
mean(rMixAdapt$AdaptErr)
rMixAdapt$losses

data.plot <- data.frame(
  "iter" = 1:length(rMixAdapt$AdaptErr),
  "alpha_ad" = 1 - stats::filter(rMixAdapt$AdaptErr, rep(1/200, 200)),
  "alpha_noad" = 1 - stats::filter(rMixAdapt$noAdaptErr, rep(1/200, 200))
)

a_eta_err <- ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = alpha_ad), color = "blue", linetype = "solid") +
  #geom_line(aes(y = alpha_noad), color = "red", linetype = "solid") +
  geom_hline(yintercept = 0.95, color = "black", linetype = "dashed") +
  labs(title = expression("Adaptive"~eta), x = "Iter", y = "Local Level Coverage", color = "") +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0.75, 1)

data.plot <- data.frame(
  "iter" = 1:length(rMixAdapt$AdaptErr),
  "lm" = rMixAdapt$weights[,1],
  "ar" = rMixAdapt$weights[,2],
  "armax" = rMixAdapt$weights[,3]
)

a_eta_wts <- ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = lm, color = "LM"), linetype = "solid", size = 1) +
  geom_line(aes(y = ar, color = "AR(2,0)"), linetype = "solid", size = 1) +
  geom_line(aes(y = armax, color = "ARMAX(2,0)"), linetype = "solid", size = 1) +
  labs(title = expression("Adaptive"~eta), x = "Iter", y = "Weights", color = "") +
  scale_color_manual(values = c("LM" = "red", "AR(2,0)" = "forestgreen", "ARMAX(2,0)" = "blue", "Weighted Maj." = "orange", "Rand. Weighted Maj." = "orange")) +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0,1)

data.plot <- data.frame(
  "iter" = 1:length(rMixAdapt$AdaptErr),
  "lm" = stats::filter(rMixAdapt$losses[,1], rep(1/100, 100)),
  "ar" = stats::filter(rMixAdapt$losses[,2], rep(1/100, 100)),
  "armax" = stats::filter(rMixAdapt$losses[,3], rep(1/100, 100)),
  "aci" = stats::filter(unlist(lapply(rMixAdapt$piAdapt, function(x) atan(x[2]-x[1]))),rep(1/100, 100))
)

a_eta_loss <- ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = lm, color = "LM"), linetype = "solid", size = 1) +
  geom_line(aes(y = ar, color = "AR(2,0)"), linetype = "solid", size = 1) +
  geom_line(aes(y = armax, color = "ARMAX(2,0)"), linetype = "solid", size = 1) +
  geom_line(aes(y = aci, color = "Weighted Maj."), linetype = "dashed", size = 1) +
  labs(title = expression("Adaptive"~eta), x = "Iter", y = "Loss", color = "") +
  scale_color_manual(values = c("LM" = "red", "AR(2,0)" = "forestgreen", "ARMAX(2,0)" = "blue", "Weighted Maj." = "orange", "Rand. Weighted Maj." = "orange")) +
  theme_minimal() + theme(legend.position = "bottom") 

grid.arrange(a_eta_err, a_eta_wts, a_eta_loss, f_eta_err, f_eta_wts, f_eta_loss, ncol= 3, nrow = 2)
