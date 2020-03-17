
# Figure 1: Aspirations and Payoffs ---------------------------------------

# Parameters
sig <- 0.65
del <- 0.95
pi <- 1.1

# Payoff functions
util <- function(c) {c ^ (1 - sig)}
w0 <- function(z) {del * z ^ (1 - sig)}
w1 <- function(e) {del * pi * e ^ (1 - sig)}
fig1_payoff <- function(z, a) {
  w0(z) + w1(max(c(z - a, 0)))
}
fig1_payoff <- Vectorize(fig1_payoff, vectorize.args = "z")

layout(matrix(1:2, nrow=1))
curve(fig1_payoff(x, a = 1), from = 0, to = 2)
curve(fig1_payoff(x, a = 1), from = 0, to = 2)
curve(fig1_payoff(x, a = 1.25), from = 0, to = 2, add = T)

# Figure 2: Costs and Benefits --------------------------------------------

rho <- 1.01
prod <- function(z) rho * z
finv <- function(z) z / rho # called 'k' in the paper
fig2_cost <- function(y, z){
  util(y) - util(y - finv(z))
}
fig2_cost <- Vectorize(fig2_cost, vectorize.args = "z")
layout(matrix(1:1))
curve(fig2_cost(y = 2, x), from = 0, to = 2, ylim = c(0, 3), xlab = "z",
      ylab = "Payoffs and Costs", main = "Fig 2")
curve(fig1_payoff(x, a = 1), from = 0, to = 2, add = T)


# Fig 3: Wealth and Aspirations -------------------------------------------

# Derivatives
util_p <- function(c) {(1 - sig) * c ^ -sig}
w0_p <- function(z) {del * (1 - sig) * z ^ -sig}
w1_p <- function(e) {del * pi * (1 - sig) * e ^ -sig}

zopt0 <- function(z, y) { # z < a
  w0_p(z) - (util_p(y - finv(z)) / rho)
}
zopt0 <- Vectorize(zopt0, vectorize.args = "z")

zopt1 <- function(z, y, a) { # z > a
  w0_p(z) + w1_p(z - a) - (util_p(y - finv(z)) / rho)
}
zopt1 <- Vectorize(zopt1, vectorize.args = "z")

# Checking for roots
curve(zopt0(x, y = 2), 0, 2)
abline(h = 0)
curve(zopt1(x, y = 2, a = 1), 0, 2)
abline(h = 0) # both have a unique soln. 

find_z <- function(z, y, a){
  # Find the roots
  z0 = uniroot(zopt0, interval = c(0, 2), y = y, extendInt = "yes")$root
  z1 = uniroot(zopt1, interval = c(1.5, 1.6), y = y, a = a, extendInt = "yes")$root
  
  # Which is better?
  u0 = util(y - finv(z0)) + w0(z0)
  u1 = util(y - finv(z1)) + w0(z0) + w1(max(c(z0 - a, 0)))
  
  if (u0 > u1) {
    return(z0)
  } else {
    return(z1)
  }
}

x <- seq(0, 1.5, length.out = 1000)
opt_z <- lapply(x, function(x) find_z(y = 2, a = x))
plot(x, unlist(opt_z), ylim = c(0, 2), type = "l")


# Figure 4: Growth and Aspirations ----------------------------------------


