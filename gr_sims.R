
# Summary and Setup -------------------------------------------------------

# Replicates model from Genicot and Ray (2017)

library(purrr)

# Setup and Params --------------------------------------------------------

# Payoff params
del <- 0.8
sig <- 0.8
pi <- 1.0

# Various production functions
big_a <- 4
rho <- 1.6
bet <- 0.55
theta <- 1.0

# Functions ---------------------------------------------------------------

# Payoff functions
util <- function(c) {c ^ (1 - sig)}
w0 <- function(z) {del * z ^ (1 - sig)}
w1 <- function(e) {del * pi * e ^ (1 - sig)}

# Production functions
prod <- function(k) theta * (big_a / bet) * (k ^ bet)
kfunc <- function(z) (bet * z / (theta * big_a)) ^ (1 / bet) # inverse of prod()

# Total Payoff
total_payoff <- function(y, z, a) {
  util(y - kfunc(z)) + w0(z) + w1(max(c(z - a, 0)))
}

# Derivatives
util_p <- function(c) {(1 - sig) * c ^ -sig}
w0_p <- function(z) {del * (1 - sig) * z ^ -sig}
w1_p <- function(e) {del * pi * (1 - sig) * e ^ -sig}
f_p <- function(k) {theta * big_a * k ^ (bet - 1)}

# Optimal z
zopt0 <- function(z, y) { # z < a
  w0_p(z) - (util_p(y - kfunc(z)) / f_p(kfunc(z)))
}

zopt1 <- function(z, y, a) { # z > a
  w0_p(z) + w1_p(z - a) - (util_p(y - kfunc(z)) / f_p(kfunc(z)))
}

find_interv <- function(y, a, both = T) {
  # Smarter way
  # both: do we evaluate optimal z on both sides of a (a < f(y)),
  # or not (a > f(y))?
  
  # z0 (<a) have to find this in either case
  zvals0 = seq(0, prod(y), length.out = 600)
  z_cands0 = zopt0(zvals0, y)
  z_cands0 = zvals0[z_cands0 < Inf & z_cands0 > -Inf & !is.na(z_cands0)]
  interv0 = range(z_cands0)
  
  if (min(interv0) == -Inf | max(interv0) == Inf) {
    cat(y, "\n")
    stop("No candidate z values found for zopt0")
  }
  
  if (min(interv0) == max(interv0)) {
    interv0 = interv0 * c(0.999, 1.001) # if one point, extend around there
  }
  
  if (both) {
    # z1 (>a)
    zvals1 = seq(a, prod(y), length.out = 600)
    z_cands1 = zopt1(zvals1, y, a)
    z_cands1 = zvals1[z_cands1 < Inf & z_cands1 > -Inf & !is.na(z_cands1)]
    interv1 = range(z_cands1)
    if (min(interv1) == -Inf | max(interv1) == Inf) {
      cat("y = ", y, "a = ", a, "\n")
      stop("No candidate z values found for zopt1")
    }
    
    return(list(interv0 = interv0, interv1 = interv1)) 
    
  } else {
    return(list(interv0 = interv0))
  }
}

find_z <- function(y, a) {
  # Check which z's are feasible
  if (a >= prod(y)) { # aspirations will be frustrated regardless
    interv0 = find_interv(y, a, both = F)$interv0
    z0 = uniroot(zopt0, interval = interv0, y = y, extendInt = "yes")$root
    return(z0)
  } else { # have to check both options
    intervs = find_interv(y, a)
    z0 = uniroot(zopt0, interval = intervs$interv0, 
                 y = y, extendInt = "yes")$root
    z1 = uniroot(zopt1, interval = intervs$interv1, 
                 y = y, a = a, extendInt = "yes")$root
    
    # Which is better?
    u0 = util(y - kfunc(z0)) + w0(z0)
    u1 = util(y - kfunc(z1)) + w0(z1) + w1(max(c(z1 - a, 0)))
    
    if (u0 > u1) {
      return(z0)
    } else {
      return(z1)
    }
  }
}

# Example 1 ---------------------------------------------------------------

# Deterministic production
y_init <- seq(10.0, 130.0, length.out = 100) # uniform dist
yvals <- y_init

# Find steady state
dif <- 10.0
tol <- 1e-3
its <- 0
maxit <- 100

while (dif > tol & its < maxit) {
  avg_inc <- mean(yvals)
  a_y <- 0.5 * (yvals + avg_inc)

  yvals_next <- map2_dbl(yvals, a_y, find_z)
  
  dif <- max(abs(yvals - yvals_next))
  its <- its + 1
  yvals <- yvals_next
}

plot(y_init, yvals)
hist(yvals, breaks = length(y_init))
p_new <- mean(round(yvals, 2) == min(round(yvals, 2)))

# whether incomes center around one or two values depends on the spread 
# of the initial distribution
# as does p_new

# Example 2: Introduce Heterogeneity --------------------------------------

# Stochastic production
y_init <- seq(10.0, 1000.0, length.out = 100) # uniform dist
yvals <- y_init

# Find steady state
dif <- 10.0
tol <- 1e-3
its <- 0
maxit <- 100
p_old <- 1

while (dif > tol & its < maxit) {
  avg_inc <- mean(yvals)
  a_y <- 0.5 * (yvals + avg_inc) # aspirations
  theta <- rlnorm(1, 1, 0.04) # random theta (aggregate)
  
  yvals_next <- map2_dbl(yvals, a_y, find_z)
  
  
  print(dif)
  its <- its + 1
  yvals <- yvals_next
  p_old <- p_next
}
