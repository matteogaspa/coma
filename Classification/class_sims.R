rm(list = ls())
library(dplyr)
library(glmnet)
library(randomForest)
library(polspline)
library(e1071)
library(ggplot2)
library(mlr)
library(gridExtra)
library(mvtnorm)
source("utils_funs.R")

# Generate data -----
nclass <- 8
npoint <- 250
dati   <- NULL
set.seed(123)
for(i in 1:nclass){
  cors <- runif(1, -1, 1)
  sds <- runif(1, 3, 5)
  pclass <- cbind(rep(i, npoint), rmvnorm(npoint, mean = c(i,i), sigma = matrix(c(sds^2, sds*cors, sds*cors, sds^2), 2, 2)), rnorm(npoint), rnorm(npoint))
  dati <- rbind(dati, pclass)
}
plot(dati[,2:3], col = dati[,1], pch = 20)

colnames(dati) <- c("y", "x1", "x2", "x3", "x4")
dati <- as.data.frame(dati)
dati$y <- as.factor(dati$y)
indx <- sample(1:NROW(dati), NROW(dati), replace = F)
dati <- dati[indx,]

# Analyses
N      <- NROW(dati)
losses <- coverage <- matrix(NA, nrow = N/2, ncol = 4)

for(i in 1:(N/2-1)){
  sel        <- 1:(N/2+i)
  dati_n     <- dati[sel,]
  test_point <- dati[N/2+i+1,2:5]
  
  # train models
  m_nn  <- cp_class_nn(dati_n, test_point, seed = i)
  m_rf  <- cp_class_rf(dati_n, test_point, seed = i)
  m_qda <- cp_class_qda(dati_n, test_point, seed = i)
  m_lda <- cp_class_lda(dati_n, test_point, seed = i)
  
  # cardinality
  losses[i,1] <- rowSums(m_nn)
  losses[i,2] <- rowSums(m_rf)
  losses[i,3] <- rowSums(m_qda)
  losses[i,4] <- rowSums(m_lda)
  
  y_n <- as.numeric(dati$y[N/2+i+1])
  
  coverage[i,1] <- I(m_nn[1,y_n]==T)
  coverage[i,2] <- I(m_rf[1,y_n]==T)
  coverage[i,3] <- I(m_qda[1,y_n]==T)
  coverage[i,4] <- I(m_lda[1,y_n]==T)
  
  if(i%%100==0){
    cat("Iter:", i, "\n")
  } 
}


# Adahedge ------
ada_alg <- adahedge(losses)
cov_ada <- matrix(NA, nrow = NROW(ada_alg$weights)-1, ncol = 2)
for(i in 1:NROW(cov_ada)){
  c_n <- coverage[i,]
  w_n <- ada_alg$weights[i,]
  cov_ada[i,1] <- sum(c_n*w_n) > runif(1, 0.5, 1)
  cov_ada[i,2] <- sum(c_n*w_n) > 0.5
}
colMeans(cov_ada)

data.plot <- data.frame(
  iter = 1:(N/2-1),
  local.m = stats::filter(cov_ada[,2], rep(1/100, 100)),
  local.wm = stats::filter(cov_ada[,1], rep(1/100, 100))
)

pM1<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = local.m, color = "Weighted Majority"), linetype = "solid") +
  geom_line(aes(y = local.wm, color = "Rand. Weighted Majority"), linetype = "dashed") +
  geom_hline(yintercept = 0.80, color = "black", linetype = "dashed") +
  labs(title = expression(paste("Adaptive ", eta)), x = "Iter", y = "Local Level Coverage", color = "") +
  scale_color_manual(values = c("Weighted Majority" = "forestgreen", "Rand. Weighted Majority" = "orange", "Bern 0.9" = "orange")) +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0.5, 1)
pM1

data.plot1 <- as.matrix(ada_alg$weights)
colnames(data.plot1) <- c("NN", "RF", "QDA", "LDA")
data.plot1 <- as.data.frame(data.plot1)
data.plot1$iter <- 1:nrow(data.plot1)

p1<-ggplot(data.plot1, aes(x = iter)) +
  geom_line(aes(y = NN, color = "NN"), linetype = "solid", size = 0.75) +
  geom_line(aes(y = RF, color = "RF"), linetype = "dashed", size = 0.75) +
  geom_line(aes(y = QDA, color = "QDA"), linetype = "dotted", size = 0.75) +
  geom_line(aes(y = LDA, color = "LDA"), linetype = "solid", size = 0.75) +
  labs(title = expression(paste("Adaptive ", eta)), x = "Iter", y = "Weights", color = "") +
  scale_color_manual(values = c("NN" = "blue",  "RF" = "red", "QDA" = "orange",  "LDA" = "forestgreen")) +
  theme_minimal() + theme(legend.position = "bottom") 
p1

# Hedge -----
hed_alg <- hedge(losses, eta = 0.01)
cov_hed <- matrix(NA, nrow = NROW(ada_alg$weights)-1, ncol = 2)
for(i in 1:NROW(cov_ada)){
  c_n <- coverage[i,]
  w_n <- hed_alg$weights[i,]
  cov_hed[i,1] <- sum(c_n*w_n) > runif(1, 0.5, 1)
  cov_hed[i,2] <- sum(c_n*w_n) > 0.5
}
colMeans(cov_hed)

data.plot <- data.frame(
  iter = 1:(N/2-1),
  local.m = stats::filter(cov_hed[,2], rep(1/100, 100)),
  local.wm = stats::filter(cov_hed[,1], rep(1/100, 100))
)

pM2<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = local.m, color = "Weighted Majority"), linetype = "solid") +
  geom_line(aes(y = local.wm, color = "Rand. Weighted Majority"), linetype = "dashed") +
  geom_hline(yintercept = 0.80, color = "black", linetype = "dashed") +
  labs(title = expression(eta == 0.01), x = "Iter", y = "Local Level Coverage", color = "") +
  scale_color_manual(values = c("Weighted Majority" = "forestgreen", "Rand. Weighted Majority" = "orange", "Bern 0.9" = "orange")) +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0.5, 1)
pM2

data.plot2 <- as.matrix(hed_alg$weights)
colnames(data.plot2) <- c("NN", "RF", "QDA", "LDA")
data.plot2 <- as.data.frame(data.plot2)
data.plot2$iter <- 1:nrow(data.plot2)

p2<-ggplot(data.plot2, aes(x = iter)) +
  geom_line(aes(y = NN, color = "NN"), linetype = "solid", size = 0.75) +
  geom_line(aes(y = RF, color = "RF"), linetype = "dashed", size = 0.75) +
  geom_line(aes(y = QDA, color = "QDA"), linetype = "dotted", size = 0.75) +
  geom_line(aes(y = LDA, color = "LDA"), linetype = "solid", size = 0.75) +
  labs(title = expression(eta == 0.01), x = "Iter", y = "Weights", color = "") +
  scale_color_manual(values = c("NN" = "blue",  "RF" = "red", "QDA" = "orange",  "LDA" = "forestgreen")) +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0,1)
p2

# Sums covs 
# Sum of covariances -----
sum_covs <- rep(NA, NROW(losses))

for(i in 10:NROW(losses)){
  sum_covs[i] <- cov(coverage[1:i,1], ada_alg$weights[1:i,1]) + cov(coverage[1:i,2], ada_alg$weights[1:i,2]) + 
    cov(coverage[1:i,3], ada_alg$weights[1:i,3]) + cov(coverage[1:i], ada_alg$weights[1:i,4])
}


data.cov1 <- data.frame(
  sum_covs = sum_covs,
  iter = 1:NROW(sum_covs)
)

p_cov1 <-ggplot(data.cov1, aes(x = iter)) +
  geom_line(aes(y = sum_covs), linetype = "solid", size = 0.75) +
  labs(title = expression(paste("Adaptive ", eta)), x = "Iter", y = "Sum of covariances", color = "") +
  theme_minimal() + theme(legend.position = "bottom") + ylim(-1, 1)

sum_covs <- rep(NA, NROW(losses))

for(i in 10:NROW(losses)){
  sum_covs[i] <- cov(coverage[1:i,1], hed_alg$weights[1:i,1]) + cov(coverage[1:i,2], hed_alg$weights[1:i,2]) + 
    cov(coverage[1:i,3], hed_alg$weights[1:i,3]) + cov(coverage[1:i], hed_alg$weights[1:i,4])
}


data.cov2 <- data.frame(
  sum_covs = sum_covs,
  iter = 1:NROW(sum_covs)
)

p_cov2 <-ggplot(data.cov2, aes(x = iter)) +
  geom_line(aes(y = sum_covs), linetype = "solid", size = 0.75) +
  labs(title = expression(paste(eta==0.01)), x = "Iter", y = "Sum of covariances", color = "") +
  theme_minimal() + theme(legend.position = "bottom") + ylim(-1, 1)

row1 <- grid.arrange(pM1, p1, p_cov1, ncol = 3)
row2 <- grid.arrange(pM2, p2, p_cov2, ncol = 3)
grid.arrange(row1, row2)
