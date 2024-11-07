rm(list = ls())
library(ggplot2)
library(gridExtra)
load("sims_cor.RData")

# Correlation 
w_alg1_c <- w_alg1
for(i in 1:NROW(w_alg1_c)){
  w_alg1_c[i,] <- w_alg1[i,] - mean(w_alg1[i,])
}

w_alg2_c <- w_alg2
for(i in 1:NROW(w_alg2_c)){
  w_alg2_c[i,] <- w_alg2[i,] - mean(w_alg2[i,])
}

w_alg3_c <- w_alg3
for(i in 1:NROW(w_alg3_c)){
  w_alg3_c[i,] <- w_alg3[i,] - mean(w_alg3[i,])
}

w_alg4_c <- w_alg4
for(i in 1:NROW(w_alg4_c)){
  w_alg4_c[i,] <- w_alg4[i,] - mean(w_alg4[i,])
}

m_alg1_c <- m_alg1
for(i in 1:NROW(m_alg1_c)){
  m_alg1_c[i,] <- m_alg1[i,] - mean(m_alg1[i,])
}

m_alg2_c <- m_alg2
for(i in 1:NROW(m_alg2_c)){
  m_alg2_c[i,] <- m_alg2[i,] - mean(m_alg2[i,])
}

m_alg3_c <- m_alg3
for(i in 1:NROW(m_alg3_c)){
  m_alg3_c[i,] <- m_alg3[i,] - mean(m_alg3[i,])
}

m_alg4_c <- m_alg4
for(i in 1:NROW(m_alg4_c)){
  m_alg4_c[i,] <- m_alg4[i,] - mean(m_alg4[i,])
}

sums_wm <- w_alg1_c*m_alg1_c + w_alg2_c*m_alg2_c + w_alg3_c*m_alg3_c + w_alg4_c*m_alg4_c

sum_cov <- rowMeans(sums_wm)
sum_lo  <- apply(sums_wm, 1, function(x) mean(x) - sd(x))
sum_up  <- apply(sums_wm, 1, function(x) mean(x) + sd(x))

plot.frame <- data.frame(
  iter = 100:200,
  sum_cov = sum_cov,
  sum_lo  = sum_lo,
  sum_up = sum_up
)

p1 <- ggplot(plot.frame, aes(x=iter)) + 
  geom_line(aes(y = sum_cov), color = "black", linetype = "solid") + 
  geom_line(aes(y = sum_lo), color = "black", linetype = "dashed") + 
  geom_line(aes(y = sum_up), color = "black", linetype = "dashed") + 
  labs(title = "", x = "Iter", y = "Sum of covariances", color = "") +
  theme_minimal() + theme(legend.position = "bottom") + ylim(-1, 1)


# Weights
plot.frame <- data.frame(
  "iter" = 100:200,
  "w1" = rowMeans(w_alg1),
  "w2" = rowMeans(w_alg2),
  "w3" = rowMeans(w_alg3),
  "w4" = rowMeans(w_alg4)
)

p2 <- ggplot(plot.frame, aes(x=iter)) + 
  geom_line(aes(y = w1, color = "w1"), linetype = "solid", linewidth = 0.75) +
  geom_line(aes(y = w2, color = "w2"), linetype = "dashed", linewidth = 0.75) +
  geom_line(aes(y = w3, color = "w3"), linetype = "dotted", linewidth = 0.75) +
  geom_line(aes(y = w4, color = "w4"), linetype = "longdash", linewidth = 0.75) +
  scale_color_manual(
    values = c("w1" = "blue", "w2" = "red", "w3" = "forestgreen", "w4" = "orange"),
    labels = c("w1" = expression(lambda[1]), "w2" = expression(lambda[2]), "w3" = expression(lambda[3]), "w4" = expression(lambda[4]))) + 
  labs(title = "", x = "Iter", y = "Weigths", color = "") +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0, 1)

# coverage
plot.frame <- data.frame(
  "iter" = 100:200,
  "w1" = 1-rowMeans(m_alg1),
  "w2" = 1-rowMeans(m_alg2),
  "w3" = 1-rowMeans(m_alg3),
  "w4" = 1-rowMeans(m_alg4)
)

p3 <- ggplot(plot.frame, aes(x=iter)) + 
  geom_line(aes(y = w1, color = "w1"), linetype = "solid", linewidth = 0.75) +
  geom_line(aes(y = w2, color = "w2"), linetype = "dashed", linewidth = 0.75) +
  geom_line(aes(y = w3, color = "w3"), linetype = "dotted", linewidth = 0.75) +
  geom_line(aes(y = w4, color = "w4"), linetype = "longdash", linewidth = 0.75) +
  geom_hline(yintercept = 0.9, linetype = "dashed") +
  scale_color_manual(
    values = c("w1" = "blue", "w2" = "red", "w3" = "forestgreen", "w4" = "orange"),
    labels = c("w1" = expression(lambda[1]), "w2" = expression(lambda[2]), "w3" = expression(lambda[3]), "w4" = expression(lambda[4]))) + 
  labs(title = "", x = "Iter", y = "Coverage", color = "") +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0.5, 1) 


# Coverage mv -------
misc_mv <- (1 - m_alg1) * w_alg1 + (1 - m_alg2) * w_alg2 + (1 - m_alg3) * w_alg3 + (1 - m_alg4) * w_alg4
misc_mv_s <- misc_mv_u <- I(misc_mv>1/2)
set.seed(1)
for(i in 1:nrow(misc_mv)){
  for(j in 1:ncol(misc_mv)){
    misc_mv_u[i, j] <- I(misc_mv[i,j]>1/2 + runif(1)/2)
  }
} 

plot.frame <- data.frame(
  "iter" = 100:200,
  "mv" = rowMeans(misc_mv_s),
  "rmv" = rowMeans(misc_mv_u)
)

p4 <- ggplot(plot.frame, aes(x=iter)) + 
  geom_line(aes(y = mv, color = "Maj. vote"), linetype = "solid", linewidth = 0.75) +
  geom_line(aes(y = rmv, color = "Rand. maj. vote"), linetype = "dashed", linewidth = 0.75) +
  scale_color_manual(values = c("Maj. vote" = "forestgreen", "Rand. maj. vote" = "orange")) + 
  labs(title = "", x = "Iter", y = "Coverage", color = "") +
  geom_hline(yintercept = 0.8, linetype = "dashed") +
  theme_minimal() + theme(legend.position = "bottom") + ylim(0.5, 1) 
  
  

grid.arrange(p1, p2, p3, p4, ncol = 4)

