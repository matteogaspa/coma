# WMA for high-dimensional regression
rm(list = ls())
library(conformalInference)
library(dplyr)
library(tidyverse)
library(glmnet)
library(ggplot2)
library(gridExtra)
library(limSolve)
source("utilsFuns.R")

# Load data -----
dati <- read.table("affitti.txt", header = T)
dati <- dati %>% select(price, latitude, longitude, accommodates, bathrooms, bedrooms,
                        extra_people, minimum_nights, maximum_nights, number_of_reviews,
                        review_scores_rating, review_scores_accuracy, reviews_per_month,
                        aggressioni_1000, droga_1000, furti_1000)


# creation of the outcome and the X matrix
y <- log(dati$price)
X <- dati[,-1]
X <- scale(X)

set.seed(123)
train <- sample(1:NROW(y), 75000)
test  <- sample(setdiff(1:NROW(y), train), 75)

yt <- y[train]
Xt <- X[train,]

y0 <- y[test]
X0 <- X[test,]


# set the functions
nnet.funs <- list(
  train = function(x, y, ...) nnet.train(x, y, size = 8, decay = 1, linout = T, maxit = 2000),
  predict  = function(out, newx) nnet.preds(out, newx)
)

funs <- list()
funs[[1]] <- rf.funs(ntree = 250, varfrac = 4/16)
funs[[2]] <- lm.funs()
funs[[3]] <- lasso.funs(standardize = F, lambda = 0.1)
funs[[4]] <- ridge.funs(standardize = F, lambda = 0.1)
funs[[5]] <- nnet.funs

# Simulation ----
t     <- 75          # number of rounds
k     <- length(funs)# number of methods
w     <- rep(1, k)   # initial weights
a     <- 1           # lin. combination
b     <- 0           # lin. combination
alpha <- 0.1         # confidence level

mat_best <- rep(NA, t)
weights  <- losses <- matrix(NA, ncol = k, nrow = t)
cis_t    <- vector("list", t)

set.seed(321)
for(i in 1:t){
  sel<- (1+1000*(i-1)):(1000*i) 
  Xa <- Xt[sel,]
  ya <- yt[sel]
  
  # prediction intervals
  conf.pred.ints <-lapply(funs, function(z) conformal.pred.split(Xa, ya, X0[i,], alpha = alpha/2,
                                                                 train.fun = z$train, predict.fun = z$predict))
  
  cis <- matrix(NA, nrow = k, ncol = 2) 
  for(l in 1:k){
    cis[l,1] <- conf.pred.ints[[l]]$lo
    cis[l,2] <- conf.pred.ints[[l]]$up
  }
  cis_t[[i]]   <- cis
  mat_best[i]  <- min(sizes(conf.pred.ints))
    
  # update weights
  loss        <- loss_fun_ls(y0[i], conf.pred.ints, a, b)
  losses[i,]  <- loss
  
  if(i %% 10 == 0) cat(i, "\n")
} 


# Compute the algorithms -----
## Without NN ----
hed_alg1 <- hedge(losses[,1:4], 0.1)
hed_alg2 <- hedge(losses[,1:4], 1)
ada_alg  <- adahedge(losses[,1:4])

results1 <- data.frame(iter  = 1:length(losses[,1] %>% stats::filter(., rep(1/5,5))),
                      RF = losses[,1] %>% stats::filter(., rep(1/5,5)),
                      Hedge_0.1 = hed_alg1$h %>% stats::filter(., rep(1/5,5)),
                      Hedge_1   = hed_alg2$h %>% stats::filter(., rep(1/5,5)),
                      AdaHedge  = ada_alg$h%>% stats::filter(., rep(1/5,5)))


p1<-ggplot(results1, aes(x = iter)) +
  geom_line(aes(y = RF, color = "RF"), linetype = "solid", size = 1) +
  geom_line(aes(y = Hedge_0.1, color = "eta = 0.1"), linetype = "longdash", size = 1) +
  geom_line(aes(y = Hedge_1, color = "eta = 1"), linetype = "longdash", size = 1) +
  geom_line(aes(y = AdaHedge, color =  "Adaptive eta"), linetype = "dashed", size = 1) + 
  labs(title = "", x = "Iter", y = "Hedge Loss", color = "") +
  scale_color_manual(values = c("RF" = "blue", "eta = 0.1" = "red", "eta = 1" = "forestgreen", "Adaptive eta" = "orange")) +
  theme_minimal() + theme(legend.position = "bottom") +
  guides(linetype = guide_legend(keywidth = 3, keyheight = 1)) +
  ylim(1.55, 1.9)


## With NN -----
hed_alg1_nn <- hedge(losses, 0.1)
hed_alg2_nn <- hedge(losses, 1)
ada_alg_nn  <- adahedge(losses)

results2 <- data.frame(iter  = 1:length(losses[,1] %>% stats::filter(., rep(1/5,5))),
                      RF = losses[,1] %>% stats::filter(., rep(1/5,5)),
                      NN = losses[,5] %>% stats::filter(., rep(1/5,5)),
                      Hedge_0.1 = hed_alg1_nn$h %>% stats::filter(., rep(1/5,5)),
                      Hedge_1   = hed_alg2_nn$h %>% stats::filter(., rep(1/5,5)),
                      AdaHedge  = ada_alg_nn$h%>% stats::filter(., rep(1/5,5)))

p2<-ggplot(results2, aes(x = iter)) +
  geom_line(aes(y = RF, color = "RF"), linetype = "solid", size = 1) +
  geom_line(aes(y = NN, color = "NN"), linetype = "solid", size = 1) +
  geom_line(aes(y = Hedge_0.1, color = "eta = 0.1"), linetype = "longdash", size = 1) +
  geom_line(aes(y = Hedge_1, color = "eta = 1"), linetype = "longdash", size = 1) +
  geom_line(aes(y = AdaHedge, color = "Adaptive eta"), linetype = "dashed", size = 1) + 
  labs(title = "", x = "Iter", y = "Hedge Loss", color = "") +
  scale_color_manual(values = c("RF" = "blue", "eta = 0.1" = "red", "eta = 1" = "forestgreen", "Adaptive eta" = "orange", "NN" = "pink")) +
  theme_minimal() + theme(legend.position = "bottom") +
  ylim(1.55, 1.9)

grid.arrange(p1, p2, nrow = 1)



# weights -----
plots.w <- vector("list", 6)
num.t   <- c(1, 15, 30, 45, 60, 75)
for(i in num.t){
  data.p <- data.frame(
    algorithm = c(rep("eta = 0.1", 4), rep("eta = 1", 4), rep("Adaptive eta", 4)),
    method = rep(c("RF", "LM", "Lasso", "Ridge"), 3),
    values = c(hed_alg1$weights[i,], hed_alg2$weights[i,], ada_alg$weights[i,])
  )
  j <- ifelse(i==1, 1, i/15 + 1)
  plots.w[[j]] <- ggplot(data.p, aes(x = method, y = values, fill = algorithm)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_grid(. ~ algorithm, scales = "free", space = "free") +
    labs(title = paste("t =",i), x = "", y = "Weight") +
    theme_minimal() + ylim(0, 1) + theme(legend.position = "none") +
    theme(axis.text.x = element_text(angle = 90, hjust = 0.5, vjust = 0.5))
}

grid.arrange(plots.w[[1]], plots.w[[2]], plots.w[[3]], plots.w[[4]], plots.w[[5]], plots.w[[6]], nrow = 2)


plots.w <- vector("list", 6)
num.t   <- c(1, 15, 30, 45, 60, 75)
for(i in num.t){
   data.p <- data.frame(
     algorithm = c(rep("eta = 0.1", 5), rep("eta = 1", 5), rep("Adaptive eta", 5)),
     method = rep(c("RF", "LM", "Lasso", "Ridge", "NN"), 3),
     values = c(hed_alg1_nn$weights[i,], hed_alg2_nn$weights[i,], ada_alg_nn$weights[i,])
   )
   j <- ifelse(i==1, 1, i/15 + 1)
   plots.w[[j]] <- ggplot(data.p, aes(x = method, y = values, fill = algorithm)) +
     geom_bar(stat = "identity", position = "dodge") +
     facet_grid(. ~ algorithm, scales = "free", space = "free") +
     labs(title = paste("t =",i), x = "", y = "Weight") +
     theme_minimal() + ylim(0, 1) + theme(legend.position = "none") +
     theme(axis.text.x = element_text(angle = 90, hjust = 0.5, vjust = 0.5))
}

grid.arrange(plots.w[[1]], plots.w[[2]], plots.w[[3]], plots.w[[4]], plots.w[[5]], plots.w[[6]], nrow = 2)


# obtained ints ----
cov_mv  <- cov_rmv <- ln_mv <- ln_rmv <- matrix(NA, ncol = 3, nrow = t)

for(i in 1:t){
  ci_mv_hed1 <- majority_vote(cis_t[[i]][1:4,], hed_alg1$weights[i,])
  ci_mv_hed2 <- majority_vote(cis_t[[i]][1:4,], hed_alg2$weights[i,])
  ci_mv_ada  <- majority_vote(cis_t[[i]][1:4,], ada_alg$weights[i,])
  
  ci_rv_hed1 <- majority_vote(cis_t[[i]][1:4,], hed_alg1$weights[i,], runif(1, 0.5, 1))
  ci_rv_hed2 <- majority_vote(cis_t[[i]][1:4,], hed_alg2$weights[i,], runif(1, 0.5, 1))
  ci_rv_ada  <- majority_vote(cis_t[[i]][1:4,], ada_alg$weights[i,], runif(1, 0.5, 1))
  
  cov_mv[i,1] <- covr_fun(ci_mv_hed1, y0[i])
  cov_mv[i,2] <- covr_fun(ci_mv_hed2, y0[i])
  cov_mv[i,3] <- covr_fun(ci_mv_ada, y0[i])
  
  cov_rmv[i,1] <- covr_fun(ci_rv_hed1, y0[i])
  cov_rmv[i,2] <- covr_fun(ci_rv_hed2, y0[i])
  cov_rmv[i,3] <- covr_fun(ci_rv_ada, y0[i])
  
  ln_mv[i,1] <- loss_fun(ci_mv_hed1)
  ln_mv[i,2] <- loss_fun(ci_mv_hed2)
  ln_mv[i,3] <- loss_fun(ci_mv_ada)
  
  ln_rmv[i,1] <- loss_fun(ci_rv_hed1)
  ln_rmv[i,2] <- loss_fun(ci_rv_hed2)
  ln_rmv[i,3] <- loss_fun(ci_rv_ada)
} 

colMeans(cov_mv); colMeans(cov_rmv)


cov_mv <- cov_rmv <- ln_mv <- ln_rmv <- matrix(NA, ncol = 3, nrow = t)

for(i in 1:t){
  ci_mv_hed1 <- majority_vote(cis_t[[i]][1:5,], hed_alg1_nn$weights[i,])
  ci_mv_hed2 <- majority_vote(cis_t[[i]][1:5,], hed_alg2_nn$weights[i,])
  ci_mv_ada  <- majority_vote(cis_t[[i]][1:5,], ada_alg_nn$weights[i,])
  
  ci_rv_hed1 <- majority_vote(cis_t[[i]][1:5,], hed_alg1_nn$weights[i,], runif(1, 0.5, 1))
  ci_rv_hed2 <- majority_vote(cis_t[[i]][1:5,], hed_alg2_nn$weights[i,], runif(1, 0.5, 1))
  ci_rv_ada  <- majority_vote(cis_t[[i]][1:5,], ada_alg_nn$weights[i,], runif(1, 0.5, 1))
  
  cov_mv[i,1] <- covr_fun(ci_mv_hed1, y0[i])
  cov_mv[i,2] <- covr_fun(ci_mv_hed2, y0[i])
  cov_mv[i,3] <- covr_fun(ci_mv_ada, y0[i])
  
  cov_rmv[i,1] <- covr_fun(ci_rv_hed1, y0[i])
  cov_rmv[i,2] <- covr_fun(ci_rv_hed2, y0[i])
  cov_rmv[i,3] <- covr_fun(ci_rv_ada, y0[i])
  
  ln_mv[i,1] <- loss_fun(ci_mv_hed1)
  ln_mv[i,2] <- loss_fun(ci_mv_hed2)
  ln_mv[i,3] <- loss_fun(ci_mv_ada)
  
  ln_rmv[i,1] <- loss_fun(ci_rv_hed1)
  ln_rmv[i,2] <- loss_fun(ci_rv_hed2)
  ln_rmv[i,3] <- loss_fun(ci_rv_ada)
} 

colMeans(cov_mv); colMeans(cov_rmv)



